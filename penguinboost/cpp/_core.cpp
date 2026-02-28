/**
 * PenguinBoost C++ acceleration module (pybind11)
 *
 * Exports:
 *   build_histogram        – single-pass gradient/hessian histogram building
 *   find_best_split_basic  – fast split finding (no adaptive_reg/monotone)
 *   predict_trees          – batch prediction over all trees (flat arrays)
 *   set_num_threads        – set the number of OpenMP threads
 *   get_num_threads        – get the current effective thread count
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#else
// Stubs so the rest of the file compiles without #ifdef guards everywhere.
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// ---- Global thread count (-1 means "use all available") ----
static int g_num_threads = -1;

static int resolve_num_threads() {
#ifdef _OPENMP
    return (g_num_threads > 0) ? g_num_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

// ============================================================
// 1. build_histogram
//
// Single-pass over sample rows.  For each sample i we iterate
// over every feature j and accumulate into hist[j, bin].
// This avoids n_features separate np.bincount calls and gives
// better cache usage because X_binned is row-major.
//
// OpenMP: samples are split across threads; each thread owns a
// private flat histogram to avoid false sharing. After the
// parallel region the per-thread histograms are merged serially.
// ============================================================

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<int64_t>>
build_histogram(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> X_binned,
    py::array_t<double,   py::array::c_style | py::array::forcecast> gradients,
    py::array_t<double,   py::array::c_style | py::array::forcecast> hessians,
    py::object sample_indices_obj,
    int max_bins)
{
    auto X = X_binned.unchecked<2>();
    auto g = gradients.unchecked<1>();
    auto h = hessians.unchecked<1>();

    const int n_features = static_cast<int>(X.shape(1));
    const int n_bins     = max_bins + 1;   // +1 for NaN bin

    // ---- Allocate output arrays ----
    py::array_t<double>  grad_hist ({n_features, n_bins});
    py::array_t<double>  hess_hist ({n_features, n_bins});
    py::array_t<int64_t> count_hist({n_features, n_bins});

    auto gh = grad_hist .mutable_unchecked<2>();
    auto hh = hess_hist .mutable_unchecked<2>();
    auto ch = count_hist.mutable_unchecked<2>();

    // Zero-initialise output (numpy does not guarantee this)
    for (int j = 0; j < n_features; ++j)
        for (int b = 0; b < n_bins; ++b) { gh(j,b)=0.0; hh(j,b)=0.0; ch(j,b)=0; }

    // ---- Resolve sample indices ----
    std::vector<int64_t> idx_buf;
    const int64_t* idx_ptr;
    int64_t n_sub;

    if (sample_indices_obj.is_none()) {
        n_sub = X.shape(0);
        idx_buf.resize(n_sub);
        for (int64_t k = 0; k < n_sub; ++k) idx_buf[k] = k;
        idx_ptr = idx_buf.data();
    } else {
        auto si = py::cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(
                      sample_indices_obj);
        n_sub = si.shape(0);
        idx_buf.resize(n_sub);
        auto si_r = si.unchecked<1>();
        for (int64_t k = 0; k < n_sub; ++k) idx_buf[k] = si_r(k);
        idx_ptr = idx_buf.data();
    }

    // ---- Thread-local histogram accumulation ----
    const int nthreads_req = resolve_num_threads();
    const int stride       = n_features * n_bins;

    // One flat histogram per thread, zero-initialised
    std::vector<double>  tl_gh((std::size_t)nthreads_req * stride, 0.0);
    std::vector<double>  tl_hh((std::size_t)nthreads_req * stride, 0.0);
    std::vector<int64_t> tl_ch((std::size_t)nthreads_req * stride, 0);

    int nthreads_used = 1;

#pragma omp parallel num_threads(nthreads_req)
    {
        const int tid  = omp_get_thread_num();
        double*  my_gh = tl_gh.data() + (std::size_t)tid * stride;
        double*  my_hh = tl_hh.data() + (std::size_t)tid * stride;
        int64_t* my_ch = tl_ch.data() + (std::size_t)tid * stride;

#pragma omp for schedule(static)
        for (int64_t k = 0; k < n_sub; ++k) {
            const int64_t i  = idx_ptr[k];
            const double  gi = g(i);
            const double  hi = h(i);
            const uint8_t* row = &X(i, 0);
            for (int j = 0; j < n_features; ++j) {
                const int b = row[j];
                my_gh[j * n_bins + b] += gi;
                my_hh[j * n_bins + b] += hi;
                my_ch[j * n_bins + b] += 1;
            }
        }

#pragma omp single
        { nthreads_used = omp_get_num_threads(); }
    }  // implicit barrier – nthreads_used is visible here

    // ---- Serial merge into output ----
    for (int t = 0; t < nthreads_used; ++t) {
        const double*  src_gh = tl_gh.data() + (std::size_t)t * stride;
        const double*  src_hh = tl_hh.data() + (std::size_t)t * stride;
        const int64_t* src_ch = tl_ch.data() + (std::size_t)t * stride;
        for (int j = 0; j < n_features; ++j) {
            for (int b = 0; b < n_bins; ++b) {
                gh(j, b) += src_gh[j * n_bins + b];
                hh(j, b) += src_hh[j * n_bins + b];
                ch(j, b) += src_ch[j * n_bins + b];
            }
        }
    }

    return {grad_hist, hess_hist, count_hist};
}

// ============================================================
// 2. find_best_split_basic
//
// Covers the common case (no adaptive_reg, no monotone constraints).
// Sweeps through (feature, bin, nan_dir) in a tight C++ loop,
// avoiding the large temporary 3-D numpy arrays the Python version creates.
//
// NaN convention (matches Python):
//   nan_dir=0 → NaN samples go LEFT
//   nan_dir=1 → NaN samples go RIGHT
//
// For a split at bin b:
//   nd=0: gL = cumL + g_nan,  gR = g_total - cumL
//   nd=1: gL = cumL,          gR = g_total - cumL + g_nan
//
// OpenMP: features split across threads; each thread tracks its own
// best and updates the global best once via a critical section.
// ============================================================

std::tuple<int, int, double, int>
find_best_split_basic(
    py::array_t<double,  py::array::c_style | py::array::forcecast> grad_hist,
    py::array_t<double,  py::array::c_style | py::array::forcecast> hess_hist,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> count_hist,
    int    max_bins,
    double reg_lambda,
    double reg_alpha,
    double min_child_weight,
    int    min_child_samples)
{
    auto gh = grad_hist .unchecked<2>();
    auto hh = hess_hist .unchecked<2>();
    auto ch = count_hist.unchecked<2>();

    const int n_features = static_cast<int>(grad_hist.shape(0));
    const int mb         = max_bins;   // valid bins: 0 .. mb-1

    double best_gain    = 0.0;
    int    best_feature = -1;
    int    best_bin     = -1;
    int    best_nan_dir = 0;

    // Inline score function (L1+L2 regularisation).
    // reg_alpha and reg_lambda are captured by reference but are read-only
    // throughout the parallel region, so concurrent access is safe.
    auto score = [&](double g_s, double h_s, double lam) -> double {
        if (reg_alpha == 0.0) {
            return g_s * g_s / (h_s + lam);
        }
        const double abs_g = std::abs(g_s);
        if (abs_g <= reg_alpha) return 0.0;
        const double g_adj = g_s - std::copysign(reg_alpha, g_s);
        return g_adj * g_adj / (h_s + lam);
    };

#pragma omp parallel num_threads(resolve_num_threads())
    {
        // Thread-local best (stack variables – no false sharing)
        double tl_best_gain    = 0.0;
        int    tl_best_feature = -1;
        int    tl_best_bin     = -1;
        int    tl_best_nan_dir = 0;

#pragma omp for schedule(static)
        for (int j = 0; j < n_features; ++j) {
            // Compute totals for valid bins
            double  g_total = 0.0, h_total = 0.0;
            int64_t c_total = 0;
            for (int b = 0; b < mb; ++b) {
                g_total += gh(j, b);
                h_total += hh(j, b);
                c_total += ch(j, b);
            }
            const double  g_nan = gh(j, mb);
            const double  h_nan = hh(j, mb);
            const int64_t c_nan = ch(j, mb);

            const double g_sum = g_total + g_nan;
            const double h_sum = h_total + h_nan;
            const double s_total = score(g_sum, h_sum, reg_lambda);

            // Sweep over nan_dir in the outer loop so inner loop is just bins
            for (int nd = 0; nd < 2; ++nd) {
                double  cumL_g = 0.0, cumL_h = 0.0;
                int64_t cumL_c = 0;

                for (int b = 0; b < mb; ++b) {
                    cumL_g += gh(j, b);
                    cumL_h += hh(j, b);
                    cumL_c += ch(j, b);

                    double  gL, hL, gR, hR;
                    int64_t cL, cR;

                    if (nd == 0) {   // NaN → left
                        gL = cumL_g + g_nan;
                        hL = cumL_h + h_nan;
                        cL = cumL_c + c_nan;
                        gR = g_total - cumL_g;
                        hR = h_total - cumL_h;
                        cR = c_total - cumL_c;
                    } else {         // NaN → right
                        gL = cumL_g;
                        hL = cumL_h;
                        cL = cumL_c;
                        gR = g_total - cumL_g + g_nan;
                        hR = h_total - cumL_h + h_nan;
                        cR = c_total - cumL_c + c_nan;
                    }

                    // Validity checks
                    if (hL < min_child_weight || hR < min_child_weight)  continue;
                    if (cL < (int64_t)min_child_samples ||
                        cR < (int64_t)min_child_samples)                  continue;

                    const double gain = 0.5 * (score(gL, hL, reg_lambda) +
                                               score(gR, hR, reg_lambda) - s_total);

                    if (gain > tl_best_gain) {
                        tl_best_gain    = gain;
                        tl_best_feature = j;
                        tl_best_bin     = b;
                        tl_best_nan_dir = nd;
                    }
                }
            }
        }

        // Merge thread-local best into global best (entered at most nthreads times)
#pragma omp critical
        {
            if (tl_best_gain > best_gain) {
                best_gain    = tl_best_gain;
                best_feature = tl_best_feature;
                best_bin     = tl_best_bin;
                best_nan_dir = tl_best_nan_dir;
            }
        }
    }

    return {best_feature, best_bin, best_gain, best_nan_dir};
}

// ============================================================
// 3. predict_trees
//
// All trees are stored as concatenated flat arrays (pre-order DFS layout).
// Feature indices stored in the tree nodes are already absolute
// (col_indices remapping is done in Python's DecisionTree.to_arrays).
//
// Tree node layout:
//   feature[i]   = split feature (-1 means leaf)
//   threshold[i] = bin threshold
//   nan_dir[i]   = 0→NaN goes left, 1→NaN goes right
//   value[i]     = leaf prediction (only read when feature[i] < 0)
//   left[i]      = index of left child  (-1 for leaf)
//   right[i]     = index of right child (-1 for leaf)
//
// tree_offsets[t] = index of root node of tree t in the flat arrays.
//
// OpenMP: loop order swapped to samples-outer / trees-inner.
// Each sample accumulates into a scalar p, written to pred[i] once
// at the end to minimise false sharing. Tree arrays are read-only.
// ============================================================

py::array_t<double>
predict_trees(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_features,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_thresholds,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_nan_dirs,
    py::array_t<double,  py::array::c_style | py::array::forcecast> all_values,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_lefts,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> all_rights,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> tree_offsets,
    py::array_t<double,  py::array::c_style | py::array::forcecast> tree_lrs,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X_binned,
    double base_score,
    int    n_trees,
    int    max_bins)
{
    // Raw pointer access for maximum speed in the inner loop
    const int32_t* feat    = all_features  .data();
    const int32_t* thresh  = all_thresholds.data();
    const int32_t* nan_dir = all_nan_dirs  .data();
    const double*  val     = all_values    .data();
    const int32_t* lchild  = all_lefts     .data();
    const int32_t* rchild  = all_rights    .data();
    const int32_t* offsets = tree_offsets  .data();
    const double*  lrs     = tree_lrs      .data();

    const int n_samples  = static_cast<int>(X_binned.shape(0));
    const int n_features = static_cast<int>(X_binned.shape(1));
    const uint8_t* Xptr  = X_binned.data();
    const uint8_t nan_bin = static_cast<uint8_t>(max_bins);

    auto predictions = py::array_t<double>(n_samples);
    double* pred = predictions.mutable_data();

    // Samples-outer / trees-inner: each sample is fully independent.
    // pred[i] is written exactly once per sample → no false sharing.
#pragma omp parallel for schedule(static) num_threads(resolve_num_threads())
    for (int i = 0; i < n_samples; ++i) {
        const uint8_t* row = Xptr + (int64_t)i * n_features;
        double p = base_score;
        for (int t = 0; t < n_trees; ++t) {
            int node = offsets[t];
            while (feat[node] >= 0) {
                const uint8_t bin = row[feat[node]];
                bool go_left;
                if (bin == nan_bin) {
                    go_left = (nan_dir[node] == 0);
                } else {
                    go_left = (bin <= (uint8_t)thresh[node]);
                }
                node = go_left ? lchild[node] : rchild[node];
            }
            p += lrs[t] * val[node];
        }
        pred[i] = p;
    }

    return predictions;
}

// ============================================================
// Thread control
// ============================================================

void set_num_threads(int n) {
    g_num_threads = (n > 0) ? n : -1;
}

int get_num_threads() {
    return resolve_num_threads();
}

// ============================================================
// Module registration
// ============================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "PenguinBoost C++ acceleration (histogram, split finding, tree prediction)";

    m.def("set_num_threads", &set_num_threads, "n"_a,
          "Set number of OpenMP threads used by PenguinBoost kernels. "
          "Pass n <= 0 to restore automatic (all available cores).");

    m.def("get_num_threads", &get_num_threads,
          "Return the current effective number of threads "
          "(1 when built without OpenMP).");

    m.def("build_histogram", &build_histogram,
          "X_binned"_a, "gradients"_a, "hessians"_a,
          "sample_indices"_a, "max_bins"_a,
          R"doc(
Build gradient/hessian/count histograms in a single row-major pass.

Parameters
----------
X_binned        : uint8 array (n_samples, n_features)
gradients       : float64 array (n_samples,)
hessians        : float64 array (n_samples,)
sample_indices  : int64 array (n_sub,) or None
max_bins        : int

Returns
-------
(grad_hist, hess_hist, count_hist) each of shape (n_features, max_bins+1)
)doc");

    m.def("find_best_split_basic", &find_best_split_basic,
          "grad_hist"_a, "hess_hist"_a, "count_hist"_a,
          "max_bins"_a, "reg_lambda"_a, "reg_alpha"_a,
          "min_child_weight"_a, "min_child_samples"_a,
          R"doc(
Find best split (feature, bin, gain, nan_dir) without adaptive_reg or monotone constraints.

Returns
-------
(best_feature, best_bin, best_gain, best_nan_dir)
Returns (-1, -1, 0.0, 0) when no valid split is found.
)doc");

    m.def("predict_trees", &predict_trees,
          "all_features"_a, "all_thresholds"_a, "all_nan_dirs"_a,
          "all_values"_a, "all_lefts"_a, "all_rights"_a,
          "tree_offsets"_a, "tree_lrs"_a,
          "X_binned"_a, "base_score"_a, "n_trees"_a, "max_bins"_a,
          R"doc(
Batch prediction across all trees using pre-order DFS flat arrays.

Feature indices in tree nodes must already be remapped to absolute column
indices (i.e., col_indices remapping is done in Python before calling this).

Parameters
----------
all_features   : int32 (total_nodes,)
all_thresholds : int32 (total_nodes,)
all_nan_dirs   : int32 (total_nodes,)
all_values     : float64 (total_nodes,)
all_lefts      : int32 (total_nodes,)
all_rights     : int32 (total_nodes,)
tree_offsets   : int32 (n_trees,)  root index of each tree
tree_lrs       : float64 (n_trees,)
X_binned       : uint8 (n_samples, n_features)
base_score     : float64
n_trees        : int
max_bins       : int

Returns
-------
predictions : float64 (n_samples,)
)doc");
}
