# PenguinBoost ドキュメント

**AROGB (Adversarial Regularized Ordered Gradient Boosting)**

LightGBM・CatBoost・XGBoost の技術を融合した独自の勾配ブースティングライブラリ。
金融データでの過学習耐性を最優先に設計。

---

## 目次

- [1. インストール](#1-インストール)
- [2. クイックスタート](#2-クイックスタート)
  - [2.1 回帰](#21-回帰)
  - [2.2 二値分類](#22-二値分類)
  - [2.3 多クラス分類](#23-多クラス分類)
  - [2.4 ランキング](#24-ランキング)
  - [2.5 生存分析](#25-生存分析)
  - [2.6 分位点回帰（VaR/CVaR）](#26-分位点回帰varcvar)
  - [2.7 Numerai スタイルのリターン予測](#27-numerai-スタイルのリターン予測)
- [3. パラメータリファレンス](#3-パラメータリファレンス)
  - [3.1 基本パラメータ](#31-基本パラメータ)
  - [3.2 サンプリング](#32-サンプリング)
  - [3.3 カテゴリカル特徴量](#33-カテゴリカル特徴量)
  - [3.4 DART（Dropout Trees）](#34-dartdropout-trees)
  - [3.5 勾配摂動](#35-勾配摂動)
  - [3.6 ベイズ適応正則化](#36-ベイズ適応正則化)
  - [3.7 単調制約](#37-単調制約)
  - [3.8 時間的正則化](#38-時間的正則化)
  - [3.9 ハイブリッド木成長](#39-ハイブリッド木成長)
  - [3.10 多重順列 Ordered Boosting](#310-多重順列-ordered-boosting)
  - [3.11 直交勾配射影](#311-直交勾配射影)
  - [3.12 Era Boosting](#312-era-boosting)
  - [3.13 特徴量露出ペナルティ](#313-特徴量露出ペナルティ)
  - [3.14 制御パラメータ](#314-制御パラメータ)
- [4. API クラスリファレンス](#4-api-クラスリファレンス)
  - [4.1 PenguinBoostRegressor](#41-penguinboostregressor)
  - [4.2 PenguinBoostClassifier](#42-penguinboostclassifier)
  - [4.3 PenguinBoostRanker](#43-penguinboostranker)
  - [4.4 PenguinBoostSurvival](#44-penguinboostsurvival)
  - [4.5 PenguinBoostQuantileRegressor](#45-penguinboostquantileregressor)
  - [4.6 FeatureNeutralizer](#46-featureneutralizer)
  - [4.7 OrthogonalGradientProjector](#47-orthogonalgradientprojector)
  - [4.8 EraBoostingReweighter](#48-eraboostingreweighter)
  - [4.9 EraMetrics](#49-erametrics)
  - [4.10 SpearmanObjective](#410-spearmanobjective)
  - [4.11 MaxSharpeEraObjective](#411-maxsharpeeraobjective)
  - [4.12 FeatureExposurePenalizedObjective](#412-featureexposurepenalizedobjective)
- [5. 詳細ガイド](#5-詳細ガイド)
  - [5.1 ハイブリッド木成長](#51-ハイブリッド木成長symmetric--leaf-wise)
  - [5.2 DART](#52-dartdropout-trees)
  - [5.3 勾配摂動](#53-勾配摂動gradient-perturbation)
  - [5.4 ベイズ適応型分割ゲイン](#54-ベイズ適応型分割ゲイン)
  - [5.5 多重順列 Ordered Boosting](#55-多重順列-ordered-boosting中央値集約)
  - [5.6 単調制約](#56-単調制約monotone-constraints)
  - [5.7 時間的正則化](#57-時間的正則化temporal-regularization)
  - [5.8 分位点回帰（VaR / CVaR）](#58-分位点回帰var--cvar)
- [6. 金融特化機能の詳細ガイド](#6-金融特化機能の詳細ガイド)
  - [6.1 特徴量中立化（Feature Neutralization）](#61-特徴量中立化feature-neutralization)
  - [6.2 直交勾配射影（Orthogonal Gradient Projection）](#62-直交勾配射影orthogonal-gradient-projection)
  - [6.3 Era Boosting（Era-aware Reweighting）](#63-era-boostingera-aware-reweighting)
  - [6.4 Spearman 目的関数](#64-spearman-目的関数)
  - [6.5 MaxSharpe Era 目的関数](#65-maxsharpe-era-目的関数)
  - [6.6 特徴量露出ペナルティ目的関数](#66-特徴量露出ペナルティ目的関数)
- [7. 金融ユーティリティ](#7-金融ユーティリティ)
  - [7.1 Purged K-Fold CV](#71-purged-k-fold-cv)
  - [7.2 レジーム検出](#72-レジーム検出)
  - [7.3 金融メトリクス](#73-金融メトリクス)
- [8. 評価メトリクス一覧](#8-評価メトリクス一覧)
- [9. scikit-learn 連携](#9-scikit-learn-連携)
  - [9.1 パイプライン統合](#91-パイプライン統合)
  - [9.2 クロスバリデーション](#92-クロスバリデーション)
  - [9.3 パラメータ操作](#93-パラメータ操作)
- [10. ユースケース別レシピ](#10-ユースケース別レシピ)
  - [10.1 Numerai スタイルリターン予測（全機能）](#101-numerai-スタイルリターン予測全機能)
  - [10.2 金融リターン予測（全部盛り）](#102-金融リターン予測全部盛り)
  - [10.3 信用スコアリング（単調制約）](#103-信用スコアリング単調制約)
  - [10.4 リスク管理（VaR/CVaR）](#104-リスク管理varcvar)
  - [10.5 大規模データ高速学習](#105-大規模データ高速学習)
  - [10.6 過学習が激しい小データ](#106-過学習が激しい小データ)
- [11. チューニングガイド](#11-チューニングガイド)
  - [11.1 パラメータ優先順位](#111-パラメータ優先順位)
  - [11.2 目的別推奨設定](#112-目的別推奨設定)
  - [11.3 機能の組み合わせ相性](#113-機能の組み合わせ相性)
- [12. アーキテクチャ](#12-アーキテクチャ)
  - [12.1 ファイル構成](#121-ファイル構成)
  - [12.2 学習フロー](#122-学習フロー)
  - [12.3 数学的背景](#123-数学的背景)

---

## 1. インストール

```bash
# 開発モード（ソースから）
git clone <repository-url>
cd penguinboost
pip install -e .

# テスト実行
pip install -e ".[dev]"
pytest tests/
```

**動作要件:**

| 要件 | バージョン |
|---|---|
| Python | >= 3.9 |
| NumPy | >= 1.21 |
| scikit-learn | >= 1.0 |
| pytest（開発時） | >= 7.0 |

---

## 2. クイックスタート

### 2.1 回帰

```python
from penguinboost import PenguinBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = PenguinBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**損失関数の選択:**

```python
# MSE（デフォルト）
model = PenguinBoostRegressor(objective="mse")

# MAE（外れ値に頑健）
model = PenguinBoostRegressor(objective="mae")

# Huber（MSEとMAEの中間）
model = PenguinBoostRegressor(objective="huber", huber_delta=1.5)
```

### 2.2 二値分類

```python
from penguinboost import PenguinBoostClassifier

clf = PenguinBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
)
clf.fit(X_train, y_train)

labels = clf.predict(X_test)          # クラスラベル
proba = clf.predict_proba(X_test)     # 確率 shape=(n, 2)
```

### 2.3 多クラス分類

```python
from sklearn.datasets import load_iris

iris = load_iris()
clf = PenguinBoostClassifier(n_estimators=100, random_state=42)
clf.fit(iris.data, iris.target)

proba = clf.predict_proba(iris.data)  # shape=(n, 3)
labels = clf.predict(iris.data)       # 0, 1, 2
print(clf.classes_)                   # array([0, 1, 2])
```

多クラス分類は内部で One-vs-All 方式のバイナリ分類器を各クラスに対して学習します。

### 2.4 ランキング

```python
from penguinboost import PenguinBoostRanker

ranker = PenguinBoostRanker(n_estimators=50, max_depth=4, random_state=42)
ranker.fit(
    X_train,
    y_train,                   # 関連度ラベル（0, 1, 2, 3, ...）
    group=[100, 100, 100],     # クエリごとのサンプル数
)
scores = ranker.predict(X_test)  # スコアが高い=上位
```

内部で LambdaRank（NDCG最適化）を使用します。

### 2.5 生存分析

```python
from penguinboost import PenguinBoostSurvival

surv = PenguinBoostSurvival(n_estimators=50, max_depth=3, random_state=42)
surv.fit(
    X_train,
    times=event_times,       # イベント/打切り時刻
    events=event_indicators, # 1=イベント発生, 0=打切り
)
risk = surv.predict(X_test)         # リスクスコア（高い=高リスク）
risk = surv.predict_hazard(X_test)  # predict() のエイリアス
```

Cox 比例ハザードモデルの偏尤度を最適化します。

### 2.6 分位点回帰（VaR/CVaR）

```python
from penguinboost import PenguinBoostQuantileRegressor

# 5%分位点の予測（VaR推定）
var_model = PenguinBoostQuantileRegressor(
    objective="quantile",
    alpha=0.05,
    n_estimators=200,
    random_state=42,
)
var_model.fit(X_train, y_train)
var_5 = var_model.predict(X_test)

# CVaR (Expected Shortfall) の推定
cvar_model = PenguinBoostQuantileRegressor(
    objective="cvar",
    alpha=0.05,
    n_estimators=200,
    random_state=42,
)
cvar_model.fit(X_train, y_train)
cvar_5 = cvar_model.predict(X_test)
```

### 2.7 Numerai スタイルのリターン予測

```python
import numpy as np
from penguinboost import (
    PenguinBoostRegressor,
    SpearmanObjective,
    FeatureNeutralizer,
)
from penguinboost.core.boosting import BoostingEngine

# データ準備: eras = 時間期間ラベル
# X.shape = (n_samples, n_features)
# y = 株式リターン（rank-normalized）
# eras = 各サンプルの属する時間期間 ID

# ── 1. SpearmanObjective を使ったエンジンレベルの学習 ──────────────
from penguinboost.core.boosting import BoostingEngine

engine = BoostingEngine(
    n_estimators=300,
    learning_rate=0.05,
    use_era_boosting=True,
    era_boosting_method='hard_era',
    use_orthogonal_gradients=True,
    orthogonal_strength=0.3,
)
obj = SpearmanObjective(corr_correction=0.5)
engine.fit(X_train, y_train, obj, era_indices=eras_train)
raw_pred = engine.predict(X_test)

# ── 2. sklearn API 経由（高レベル） ────────────────────────────────
model = PenguinBoostRegressor(
    n_estimators=300,
    learning_rate=0.05,
    use_era_boosting=True,
    era_boosting_method='hard_era',
    use_feature_exposure_penalty=True,
    feature_exposure_lambda=0.1,
    random_state=42,
)
model.fit(X_train, y_train, era_indices=eras_train)
pred = model.predict(X_test)

# ── 3. 学習後の特徴量中立化 ────────────────────────────────────────
neutralized = model.neutralize(pred, X_test, proportion=0.5)

# 特徴量露出の確認
exposures = model.feature_exposure(X_test)
print(f"Max feature exposure: {np.max(np.abs(exposures)):.4f}")
```

---

## 3. パラメータリファレンス

全ての高レベルAPI（`PenguinBoostRegressor`, `PenguinBoostClassifier`, `PenguinBoostRanker`, `PenguinBoostSurvival`, `PenguinBoostQuantileRegressor`）で共通のパラメータです。

### 3.1 基本パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `n_estimators` | int | 100 | ブースティングラウンド数（木の数） |
| `learning_rate` | float | 0.1 | 学習率（shrinkage）。小さいほど頑健だが収束が遅い |
| `max_depth` | int | 6 | 木の最大深度 |
| `max_leaves` | int | 31 | 木の最大葉数（leaf-wise 成長で使用） |
| `growth` | str | `"leafwise"` | 木の成長戦略。`"leafwise"`, `"symmetric"`, `"depthwise"`, `"hybrid"` |
| `reg_lambda` | float | 1.0 | L2 正則化（葉の重みに対する） |
| `reg_alpha` | float | 0.0 | L1 正則化（スパース性を促進） |
| `min_child_weight` | float | 1.0 | 葉ノードの最小ヘシアン合計 |
| `min_child_samples` | int | 1 | 葉ノードの最小サンプル数 |
| `max_bins` | int | 255 | ヒストグラムの最大ビン数（1〜255） |
| `early_stopping_rounds` | int/None | None | 検証スコアが改善しないラウンド数で学習停止 |
| `importance_type` | str | `"gain"` | 特徴量重要度の種類。`"gain"` または `"split"` |

### 3.2 サンプリング

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `subsample` | float | 1.0 | 行サブサンプリング率。1.0=全サンプル使用 |
| `colsample_bytree` | float | 1.0 | 列サブサンプリング率（木ごと） |
| `use_goss` | bool | False | GOSS（Gradient-based One-Side Sampling）を使用 |
| `goss_top_rate` | float | 0.2 | GOSS: 勾配上位の保持割合 |
| `goss_other_rate` | float | 0.1 | GOSS: 残りからのランダムサンプリング割合 |
| `efb_threshold` | float | 0.0 | Exclusive Feature Bundling 衝突閾値（0=無効） |

### 3.3 カテゴリカル特徴量

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `cat_features` | list/None | None | カテゴリカル特徴量のインデックスリスト |
| `cat_smoothing` | float | 10.0 | Ordered Target Statistics のスムージング係数 |

### 3.4 DART（Dropout Trees）

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_dart` | bool | False | DART を有効化 |
| `dart_drop_rate` | float | 0.1 | 各既存木のドロップ確率 |
| `dart_skip_drop` | float | 0.0 | イテレーション単位でドロップ自体をスキップする確率 |

### 3.5 勾配摂動

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_gradient_perturbation` | bool | False | 勾配摂動を有効化 |
| `gradient_clip_tau` | float | 5.0 | 勾配のクリッピング閾値 |
| `gradient_noise_eta` | float | 0.1 | ノイズスケール（勾配の標準偏差に対する比率） |

### 3.6 ベイズ適応正則化

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_adaptive_reg` | bool | False | 適応正則化を有効化 |
| `adaptive_alpha` | float | 0.5 | イテレーション進行に伴うλ増加率 |
| `adaptive_mu` | float | 1.0 | ベイズ子ノード罰則係数（1/√n で作用） |

### 3.7 単調制約

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `monotone_constraints` | dict/None | None | `{特徴量index: +1/-1}` の辞書 |

### 3.8 時間的正則化

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_temporal_reg` | bool | False | 時間的正則化を有効化 |
| `temporal_rho` | float | 0.1 | 時間的平滑化ペナルティ強度 |

### 3.9 ハイブリッド木成長

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `symmetric_depth` | int | 3 | ハイブリッド成長で対称木を使う深度 |

`growth="hybrid"` 時のみ有効。

### 3.10 多重順列 Ordered Boosting

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_ordered_boosting` | bool | False | Ordered Boosting を有効化 |
| `n_permutations` | int | 4 | 順列数 K。K≥2 で中央値集約が有効化 |

### 3.11 直交勾配射影

勾配から特徴量に線形説明できる成分を除去し、モデルが真の非線形 alpha を学習するよう誘導します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_orthogonal_gradients` | bool | False | 直交勾配射影を有効化 |
| `orthogonal_strength` | float | 1.0 | 射影の強度。0.0=なし、1.0=完全直交化 |
| `orthogonal_eps` | float | 1e-4 | (X^TX + εI)^{-1} の Tikhonov 正則化強度 |
| `orthogonal_features` | list/None | None | 対象とする特徴量インデックス。None=全特徴量 |

### 3.12 Era Boosting

era（時間期間）ごとの Spearman 相関に基づいてサンプルの重みを動的に調整します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_era_boosting` | bool | False | Era Boosting を有効化 |
| `era_boosting_method` | str | `'hard_era'` | 重み付け戦略: `'hard_era'`, `'sharpe_reweight'`, `'proportional'` |
| `era_boosting_temp` | float | 1.0 | Softmax 温度。低いほど最悪 era に集中 |

**`era_boosting_method` の違い:**

| 値 | 動作 |
|---|---|
| `'hard_era'` | Spearman 相関が低い era を softmax(-ρ/T) で重点化 |
| `'sharpe_reweight'` | Sharpe 比を向上させる era を優先（era 間の分散低減） |
| `'proportional'` | 1 - \|ρ\| に比例した重み（全 era を常にサンプリング） |

**注意:** `fit()` に `era_indices` を渡す必要があります。

### 3.13 特徴量露出ペナルティ

学習中、予測値と特徴量の相関（特徴量露出）を直接ペナルティとして勾配に加算します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_feature_exposure_penalty` | bool | False | 特徴量露出ペナルティを有効化 |
| `feature_exposure_lambda` | float | 0.1 | ペナルティ強度 λ |
| `exposure_penalty_features` | list/None | None | 対象とする特徴量インデックス。None=全特徴量 |

### 3.14 制御パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `verbose` | int | 0 | 出力レベル。0=なし、1=各イテレーション |
| `random_state` | int/None | None | 乱数シード |

---

## 4. API クラスリファレンス

### 4.1 PenguinBoostRegressor

```python
from penguinboost import PenguinBoostRegressor
```

**追加パラメータ:**

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `objective` | str | `"mse"` | 損失関数: `"mse"`, `"mae"`, `"huber"` |
| `huber_delta` | float | 1.0 | Huber損失のδパラメータ |

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X, y, eval_set=None, era_indices=None)` | 学習。`era_indices` は era ラベル配列（Era Boosting / MaxSharpe 目的関数に必要） |
| `predict(X)` | 予測値を返す |
| `neutralize(predictions, X, proportion=1.0, per_era=False, eras=None, features=None)` | 予測値から特徴量露出を除去 |
| `feature_exposure(X, predictions=None, features=None)` | 予測値と各特徴量の Pearson 相関を返す |
| `feature_importances_` | 特徴量重要度（プロパティ） |
| `get_params()` | 全パラメータを辞書で取得 |
| `set_params(**params)` | パラメータを設定 |

### 4.2 PenguinBoostClassifier

```python
from penguinboost import PenguinBoostClassifier
```

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X, y, eval_set=None)` | 学習。2クラス/多クラス自動判定 |
| `predict(X)` | クラスラベルを返す |
| `predict_proba(X)` | クラス確率を返す。shape=(n_samples, n_classes) |
| `feature_importances_` | 特徴量重要度 |

**属性（学習後）:**

| 属性 | 説明 |
|---|---|
| `classes_` | クラスラベル配列 |
| `n_classes_` | クラス数 |

### 4.3 PenguinBoostRanker

```python
from penguinboost import PenguinBoostRanker
```

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X, y, group, eval_set=None, eval_group=None)` | 学習。`group` はクエリあたりのサンプル数リスト |
| `predict(X)` | ランキングスコアを返す |

### 4.4 PenguinBoostSurvival

```python
from penguinboost import PenguinBoostSurvival
```

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X, times, events, eval_set=None)` | 学習。`times`: イベント時刻、`events`: 1=イベント/0=打切り |
| `predict(X)` | リスクスコア（exp(raw)）を返す |
| `predict_hazard(X)` | `predict()` のエイリアス |

### 4.5 PenguinBoostQuantileRegressor

```python
from penguinboost import PenguinBoostQuantileRegressor
```

**追加パラメータ:**

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `objective` | str | `"quantile"` | `"quantile"`（pinball損失）または `"cvar"` |
| `alpha` | float | 0.5 | 分位点レベル。0.05 = 5%VaR |

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X, y, eval_set=None)` | 学習 |
| `predict(X)` | 条件付き分位点予測を返す |

### 4.6 FeatureNeutralizer

```python
from penguinboost import FeatureNeutralizer
```

学習後に予測値から特徴量に線形説明できる成分を除去します（Numerai 標準手法）。

**コンストラクタ:**

```python
FeatureNeutralizer(eps=1e-5)
```

| パラメータ | 説明 |
|---|---|
| `eps` | (X^TX + εI)^{-1} の Tikhonov 正則化強度 |

**メソッド:**

| メソッド | 説明 |
|---|---|
| `neutralize(predictions, X, features=None, proportion=1.0, per_era=False, eras=None)` | 予測値を中立化して返す |
| `feature_exposure(predictions, X, features=None)` | 各特徴量との Pearson 相関を返す。shape=(n_features,) |
| `max_feature_exposure(predictions, X, features=None)` | 最大絶対値露出（スカラー） |
| `feature_exposure_per_era(predictions, X, eras, features=None)` | era 平均の特徴量露出 |

### 4.7 OrthogonalGradientProjector

```python
from penguinboost import OrthogonalGradientProjector
```

学習中に各イテレーションの勾配から特徴量線形成分を除去します。`fit()` で事前計算し、`project()` で適用します。

**コンストラクタ:**

```python
OrthogonalGradientProjector(strength=1.0, eps=1e-4, features=None)
```

| パラメータ | 説明 |
|---|---|
| `strength` | 射影強度。0=なし、1=完全直交化 |
| `eps` | Tikhonov 正則化 |
| `features` | 対象特徴量インデックス。None=全特徴量 |

**メソッド:**

| メソッド | 説明 |
|---|---|
| `fit(X)` | (X^TX + εI)^{-1} を事前計算（学習前に1回呼ぶ） |
| `project(gradients)` | 勾配から特徴量線形成分を除去して返す |

### 4.8 EraBoostingReweighter

```python
from penguinboost import EraBoostingReweighter
```

era ごとの Spearman 相関に基づいてサンプル重みを計算します。

**コンストラクタ:**

```python
EraBoostingReweighter(method='hard_era', temperature=1.0, min_era_weight=0.02)
```

| パラメータ | 説明 |
|---|---|
| `method` | `'hard_era'`, `'sharpe_reweight'`, `'proportional'` |
| `temperature` | Softmax 温度（低いほど最悪 era に集中） |
| `min_era_weight` | 各 era の最低重みフロア（飢餓防止） |

**メソッド:**

| メソッド | 説明 |
|---|---|
| `compute_sample_weights(predictions, targets, eras)` | サンプル重みを返す。sum=n_samples |
| `era_stats(predictions, targets, eras)` | era ごとの Spearman 相関を辞書で返す |

### 4.9 EraMetrics

```python
from penguinboost import EraMetrics
```

学習中の era ごとのパフォーマンスをトラッキングします。

**コンストラクタ:**

```python
EraMetrics(eras)
```

**メソッド:**

| メソッド | 説明 |
|---|---|
| `update(predictions, targets)` | 現在のイテレーションの era 相関を記録 |
| `mean_corr(iteration=-1)` | 指定イテレーションの era 平均 Spearman 相関 |
| `sharpe(iteration=-1)` | era 相関の Sharpe 比（mean/std） |
| `worst_era(iteration=-1)` | 最低 Spearman 相関の era ラベル |

### 4.10 SpearmanObjective

```python
from penguinboost import SpearmanObjective
```

ランク相関を近似的に最大化する目的関数。ランク MSE + Pearson 補正。

**コンストラクタ:**

```python
SpearmanObjective(corr_correction=0.5)
```

| パラメータ | 説明 |
|---|---|
| `corr_correction` | Pearson 相関補正の強度 [0, 1]。0=純粋ランクMSE |

**インターフェース:** `init_score(y)`, `gradient(y, pred)`, `hessian(y, pred)`, `loss(y, pred)`

### 4.11 MaxSharpeEraObjective

```python
from penguinboost import MaxSharpeEraObjective
```

era ごとの Spearman 相関の Sharpe 比（mean/std）を最大化する目的関数。

**コンストラクタ:**

```python
MaxSharpeEraObjective(fallback_to_spearman=True, corr_eps=1e-9)
```

**メソッド（目的関数インターフェース以外）:**

| メソッド | 説明 |
|---|---|
| `set_era_indices(eras)` | era ラベル配列を設定（学習前に必須） |

**注:** `PenguinBoostRegressor.fit(..., era_indices=eras)` を渡すと自動的に `set_era_indices` が呼ばれます。`BoostingEngine` を直接使う場合も同様。

### 4.12 FeatureExposurePenalizedObjective

```python
from penguinboost import FeatureExposurePenalizedObjective
```

任意の目的関数に特徴量露出ペナルティを追加するラッパー。

**コンストラクタ:**

```python
FeatureExposurePenalizedObjective(
    base_objective,   # 基底目的関数（例: SpearmanObjective()）
    X_ref,            # 学習用特徴量行列 shape=(n_samples, n_features)
    lambda_fe=0.1,    # ペナルティ強度
    features=None,    # 対象特徴量インデックス
)
```

**メソッド:**

| メソッド | 説明 |
|---|---|
| `gradient(y, pred)` | 基底勾配 + 露出ペナルティ勾配 |
| `feature_exposure(pred)` | 現在の予測の per-feature 露出 |

---

## 5. 詳細ガイド

### 5.1 ハイブリッド木成長（Symmetric → Leaf-wise）

**概要:** CatBoostの対称木（正則化効果大）で木の上部を構築し、途中からLightGBMのleaf-wise分割（表現力大）に切り替えます。

**数学的背景:**
- レベル 0 ~ `symmetric_depth-1`: 各レベルで全ノードのヒストグラムを集約し、1つの分割ルールを共有（oblivious tree）
- レベル `symmetric_depth` 以降: 各葉ノードを独立にbest-first分割

```python
model = PenguinBoostRegressor(
    growth="hybrid",
    symmetric_depth=3,   # 浅い部分は対称木（安定）
    max_depth=8,         # 深い部分はleaf-wise（柔軟）
    max_leaves=63,
)
```

**チューニング:**

| `symmetric_depth` | 効果 |
|---|---|
| 1〜2 | ほぼleaf-wise。柔軟性を優先 |
| 3〜4 | バランスの良い設定（推奨） |
| 5以上 | 正則化が強い。過学習しやすいデータ向け |

**`growth` の4種類の比較:**

| 戦略 | 由来 | 特徴 |
|---|---|---|
| `"leafwise"` | LightGBM | ゲイン最大の葉を優先分割。表現力高、過学習しやすい |
| `"symmetric"` | CatBoost | 全ノードが同じ分割ルール。正則化効果大、表現力は限定的 |
| `"depthwise"` | XGBoost | レベルごとに全ノードを分割。leafwiseとsymmetricの中間 |
| `"hybrid"` | PenguinBoost独自 | 浅い部分はsymmetric、深い部分はleafwise |

### 5.2 DART（Dropout Trees）

**概要:** 各ブースティングイテレーションで、既存の木を確率的にドロップアウトし、ドロップされた木の分だけ勾配を再計算します。新しい木にはスケーリング `1/(1+n_dropped)` を適用します。

**数学:**
```
ドロップ後の予測: F'(x) = F(x) - Σ_{k∈dropped} η·f_k(x)
新しい木のスケール: η_new = η / (1 + |dropped|)
```

```python
model = PenguinBoostRegressor(
    use_dart=True,
    dart_drop_rate=0.1,     # 各木を10%確率でドロップ
    dart_skip_drop=0.0,     # ドロップをスキップする確率
    n_estimators=300,       # DARTは多めの木が必要
    learning_rate=0.1,
)
```

**注意点:**
- DARTは木が多いほど効果的。`n_estimators=200+` を推奨
- 学習が遅くなる（ドロップ→再計算のオーバーヘッド）
- `dart_skip_drop=0.1` でたまにドロップをスキップすると安定する場合がある

### 5.3 勾配摂動（Gradient Perturbation）

**概要:** 勾配にクリッピングとガウスノイズを加えることで、暗黙の正則化効果を得ます。

**数学:**
```
g_perturbed = clip(g, -τ, τ) + ε
ε ~ N(0, (η · std(g))²)
```

```python
model = PenguinBoostRegressor(
    use_gradient_perturbation=True,
    gradient_clip_tau=5.0,      # 勾配を[-5, 5]にクリッピング
    gradient_noise_eta=0.05,    # std(g)の5%のノイズ
)
```

**チューニング:**

| パラメータ | 小さい値 | 大きい値 |
|---|---|---|
| `gradient_clip_tau` | 外れ値を強く抑制 | クリッピングなしに近い |
| `gradient_noise_eta` | ノイズ小→元の勾配に近い | ノイズ大→正則化が強い |

**推奨:**
- 金融データ（外れ値多い）: `tau=3.0〜5.0`, `eta=0.03〜0.1`
- 通常データ: `tau=10.0`, `eta=0.01〜0.05`

### 5.4 ベイズ適応型分割ゲイン

**概要:** 標準のXGBoost分割ゲインを拡張し、子ノードごとに異なるλ値を適用します。

**数学:**
```
標準:    Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)]

改良版: Gain = ½[G_L²/(H_L+λ_L) + G_R²/(H_R+λ_R) - G²/(H+λ)]

         λ_L = λ_base·(1 + α·t/T) + μ/√n_L
         λ_R = λ_base·(1 + α·t/T) + μ/√n_R
```

- `α·t/T`: イテレーションが進むほど正則化が強くなる（学習後半の過学習を防止）
- `μ/√n`: サンプル数が少ないノードほどペナルティが大きい（ベイズ的事前分布）

```python
model = PenguinBoostRegressor(
    use_adaptive_reg=True,
    reg_lambda=1.0,           # ベースλ
    adaptive_alpha=0.5,       # 最終イテレーションでλが1.5倍
    adaptive_mu=1.0,          # 小ノードペナルティ
)
```

### 5.5 多重順列 Ordered Boosting（中央値集約）

**概要:** CatBoostのOrdered Boostingを拡張し、K個の順列で勾配を推定後、中央値で集約します。

**数学:**
```
K個の順列 π₁..πK で各サンプルの勾配を計算:
  g_i = median(g_i^(π₁), ..., g_i^(πK))
  h_i = median(h_i^(π₁), ..., h_i^(πK))
```

```python
model = PenguinBoostRegressor(
    use_ordered_boosting=True,
    n_permutations=4,          # K=4（推奨）
)
```

### 5.6 単調制約（Monotone Constraints）

**概要:** 特徴量と予測値の間に単調性（増加/減少）を強制します。制約に違反する分割は棄却されます。

```python
model = PenguinBoostRegressor(
    monotone_constraints={
        0: +1,   # 特徴量0が増加 → 予測も増加
        2: -1,   # 特徴量2が増加 → 予測は減少
        # 指定なしの特徴量は制約なし
    },
)
```

### 5.7 時間的正則化（Temporal Regularization）

**概要:** 時間的に隣接するサンプルの予測値の急激な変動にペナルティを加えます。

**数学:**
```
Ω_temporal = ρ · Σ_t (F(x_t) - F(x_{t-1}))²

追加勾配:
  内部点: g_temp[t] = 2ρ · (2F_t - F_{t-1} - F_{t+1})
  境界点: g_temp[0] = 2ρ · (F_0 - F_1)
```

**前提:** データが時間順にソートされていること。

```python
model = PenguinBoostRegressor(
    use_temporal_reg=True,
    temporal_rho=0.01,
)
```

### 5.8 分位点回帰（VaR / CVaR）

#### Quantile（VaR推定）

```
損失: L(y, f) = α·max(y-f, 0) + (1-α)·max(f-y, 0)
勾配: g_i = α - 1{y_i < pred_i}
```

#### CVaR（Expected Shortfall）

```
勾配: g_i = 1 - 1{y_i < pred_i} / α
```

---

## 6. 金融特化機能の詳細ガイド

### 6.1 特徴量中立化（Feature Neutralization）

**概要:** 学習後の後処理として、予測値から特徴量の線形効果を除去します。Numerai で標準的に使われる手法。

**数学:**
```
p_neutralized = p - proportion · X(X^TX + εI)^{-1}X^T p
               = p - proportion · proj_X(p)
```

`proportion=1.0` で予測値が特徴量空間と完全に直交化されます。

```python
from penguinboost import PenguinBoostRegressor, FeatureNeutralizer

model = PenguinBoostRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ── 方法1: モデルのメソッドを使う（推奨） ──
neutralized = model.neutralize(
    pred, X_test,
    proportion=0.5,       # 50%の露出を除去
    per_era=True,         # era 内で独立に中立化（Numerai 推奨）
    eras=eras_test,
)

# ── 方法2: FeatureNeutralizer を直接使う ──
fn = FeatureNeutralizer(eps=1e-5)
neutralized = fn.neutralize(pred, X_test, proportion=1.0)

# 露出の確認
exposures = fn.feature_exposure(pred, X_test)
print(f"Max exposure before: {np.max(np.abs(exposures)):.4f}")

exposures_after = fn.feature_exposure(neutralized, X_test)
print(f"Max exposure after:  {np.max(np.abs(exposures_after)):.4f}")
```

**`proportion` のチューニング:**

| 値 | 効果 |
|---|---|
| 0.0 | 中立化なし（元の予測） |
| 0.5 | 露出を半減。予測の情報量を保ちながら露出低減 |
| 1.0 | 完全中立化。最大限の露出除去（Numerai 標準） |

**`per_era=True` の推奨理由:**
各 era で特徴量の分布が異なる可能性があるため、era 内で独立に中立化することで era-specific な線形効果を除去できます。

### 6.2 直交勾配射影（Orthogonal Gradient Projection）

**概要:** 学習中の各イテレーションで、勾配から特徴量に線形説明できる成分を除去します。これにより、モデルが特徴量の線形効果を再学習することを防ぎ、真の非線形 alpha シグナルを発見するよう誘導します。

**数学:**
```
g_orth = g - strength · X(X^TX + εI)^{-1}X^T g
```

`(X^TX + εI)^{-1}` は学習前に1回だけ計算（計算効率良好）。

```python
model = PenguinBoostRegressor(
    n_estimators=300,
    use_orthogonal_gradients=True,
    orthogonal_strength=0.5,   # 勾配の50%の特徴量成分を除去
    orthogonal_eps=1e-4,       # 数値安定性
    random_state=42,
)
model.fit(X_train, y_train)
```

**特徴量中立化との違い:**

| 手法 | タイミング | 効果 |
|---|---|---|
| `FeatureNeutralizer` | 学習後（後処理） | 最終予測から線形露出を除去 |
| `use_orthogonal_gradients` | 学習中（各イテレーション） | モデルが線形効果を学ばないように誘導 |

両方を組み合わせることも可能です。

### 6.3 Era Boosting（Era-aware Reweighting）

**概要:** 金融 ML（特に Numerai）では、データが時間期間（era）に分割されています。Era Boosting は各 era の Spearman 相関を計算し、パフォーマンスが低い era を重点的に学習させます。

**目的:** 特定の era で壊滅的に失敗するモデルではなく、全時間期間で安定して機能するモデルを作る。

```python
import numpy as np

# era_indices の作成例（各サンプルが属する週番号など）
n_eras = 10
n_per_era = 50
eras = np.repeat(np.arange(n_eras), n_per_era)

model = PenguinBoostRegressor(
    n_estimators=300,
    use_era_boosting=True,
    era_boosting_method='hard_era',
    era_boosting_temp=0.5,   # 低温→最悪eraに集中
    random_state=42,
)
model.fit(X_train, y_train, era_indices=eras_train)
```

**`EraBoostingReweighter` を単独で使う:**

```python
from penguinboost import EraBoostingReweighter

rew = EraBoostingReweighter(method='sharpe_reweight', temperature=1.0)
weights = rew.compute_sample_weights(predictions, targets, eras)

# era 統計の確認
stats = rew.era_stats(predictions, targets, eras)
for era, corr in stats.items():
    print(f"Era {era}: Spearman = {corr:.4f}")
```

**`EraMetrics` でトレーニング中のモニタリング:**

```python
from penguinboost import EraMetrics

em = EraMetrics(eras_train)
# トレーニングループ内で各イテレーション後に呼ぶ
em.update(current_predictions, y_train)
print(f"Mean corr: {em.mean_corr():.4f}")
print(f"Sharpe:    {em.sharpe():.4f}")
print(f"Worst era: {em.worst_era()}")
```

### 6.4 Spearman 目的関数

**概要:** ランク相関（Spearman）を近似的に最大化する目的関数。ランク正規化ターゲットに対する MSE として定式化。

**数学:**
```
L(P, Y) = Σ_i (P_i - r_i)²

where r_i = rank_normalize(Y_i) ∈ [-1, 1]
```

Pearson 補正（`corr_correction > 0`）を有効にすると、予測スケールのミスマッチを補正し、より正確なランク相関勾配を得られます。

```python
from penguinboost import SpearmanObjective
from penguinboost.core.boosting import BoostingEngine

obj = SpearmanObjective(corr_correction=0.5)

engine = BoostingEngine(
    n_estimators=300,
    learning_rate=0.05,
    use_era_boosting=True,
)
engine.fit(X_train, y_train, obj, era_indices=eras_train)
pred = engine.predict(X_test)
```

**`corr_correction` のチューニング:**

| 値 | 動作 |
|---|---|
| 0.0 | 純粋なランク MSE（シンプル、安定） |
| 0.5 | 補正あり（推奨バランス） |
| 1.0 | 完全 Pearson 補正（スケール感度高） |

### 6.5 MaxSharpe Era 目的関数

**概要:** era ごとの Spearman 相関の Sharpe 比を最大化します。平均相関を維持しながら era 間の分散を低減し、安定した収益を目指します。

**数学（Sharpe の chain rule）:**
```
maximize  Sharpe(ρ) = μ_era(ρ) / σ_era(ρ)

∂Sharpe/∂P_i = Σ_e w_e · ∂ρ_e/∂P_i

where w_e = (σ² - μ(ρ_e - μ)) / (n_eras · σ³)   [Sharpe gradient weight]
      ∂ρ_e/∂P_i ≈ (r_ei - r̄_e) / (n_e · σ_P · σ_r)  [Pearson approx]
```

era 平均より低い相関の era に対して大きな重みが与えられ、それらを改善することで Sharpe が上昇します。

```python
from penguinboost import MaxSharpeEraObjective
from penguinboost.core.boosting import BoostingEngine

obj = MaxSharpeEraObjective()
# era_indices は fit() に渡すと自動設定される
engine = BoostingEngine(n_estimators=300, learning_rate=0.05)
engine.fit(X_train, y_train, obj, era_indices=eras_train)
```

**SpearmanObjective との使い分け:**

| 目的関数 | 最適化指標 | 向いているシナリオ |
|---|---|---|
| `SpearmanObjective` | 全体のランク相関 | 高い平均リターンを重視 |
| `MaxSharpeEraObjective` | era 相関の Sharpe | era 間の安定性を重視（ドローダウン低減） |

### 6.6 特徴量露出ペナルティ目的関数

**概要:** 任意の目的関数をラップし、学習中に予測値と特徴量の相関（特徴量露出）を直接ペナルティとして加えます。

**数学（露出ペナルティの勾配）:**
```
R(P) = λ · Σ_k corr(P, X_k)²

∂R/∂P_i = 2λ · Σ_k ρ_k · [(X_ki - X̄_k)/(n·σ_P·σ_k) - ρ_k·(P_i - P̄)/(n·σ_P²)]
```

```python
from penguinboost import (
    FeatureExposurePenalizedObjective,
    SpearmanObjective,
)
from penguinboost.core.boosting import BoostingEngine

base = SpearmanObjective(corr_correction=0.5)
obj = FeatureExposurePenalizedObjective(
    base_objective=base,
    X_ref=X_train,
    lambda_fe=0.2,        # 露出ペナルティ強度
    features=None,        # None=全特徴量
)

engine = BoostingEngine(n_estimators=300, learning_rate=0.05)
engine.fit(X_train, y_train, obj, era_indices=eras_train)
```

**`use_feature_exposure_penalty` パラメータとの違い:**

| 方法 | 使い方 | 注意点 |
|---|---|---|
| `FeatureExposurePenalizedObjective` | 任意の目的関数にラップ可能 | エンジンレベルの API |
| `use_feature_exposure_penalty=True` | `PenguinBoostRegressor` パラメータ | sklearn API から簡単に使える |

---

## 7. 金融ユーティリティ

### 7.1 Purged K-Fold CV

時系列の金融データで情報リークを防ぐクロスバリデーション。テスト区間の直後にエンバーゴ期間を設け、その期間のサンプルを訓練から除外します。

```python
from penguinboost.core.financial import PurgedKFold
from penguinboost import PenguinBoostRegressor
import numpy as np

# データは時間順にソート済みと仮定
cv = PurgedKFold(
    n_splits=5,
    embargo_pct=0.02,   # 全サンプルの2%をエンバーゴ
)

scores = []
for train_idx, test_idx in cv.split(X):
    model = PenguinBoostRegressor(n_estimators=100, random_state=42)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    scores.append(np.mean((y[test_idx] - pred) ** 2))

print(f"Purged CV MSE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
```

**パラメータ:**

| パラメータ | 説明 |
|---|---|
| `n_splits` | フォールド数 |
| `embargo_pct` | テスト直後に除外するサンプルの割合（0.01 = 1%） |

### 7.2 レジーム検出

ローリングボラティリティに基づいてマーケットレジーム（低/中/高ボラティリティ）を分類します。

```python
from penguinboost.core.financial import RegimeDetector
import numpy as np

returns = np.diff(np.log(prices))

detector = RegimeDetector(window=20, n_regimes=3)
detector.fit(returns)
regimes = detector.predict(returns)
# regimes: 0=低ボラ, 1=中ボラ, 2=高ボラ

# レジーム別にモデルを学習する例
for regime in range(3):
    mask = regimes == regime
    if mask.sum() > 50:
        model = PenguinBoostRegressor(n_estimators=100, random_state=42)
        model.fit(X[mask], y[mask])
```

### 7.3 金融メトリクス

```python
from penguinboost.metrics.metrics import sharpe_ratio, max_drawdown, quantile_loss
```

#### sharpe_ratio

```python
sr = sharpe_ratio(daily_returns, risk_free_rate=0.0)
```

年率換算シャープレシオ: `(mean(r - rf) / std(r - rf)) * sqrt(252)`

#### max_drawdown

```python
mdd = max_drawdown(cumulative_portfolio_value)
```

最大ドローダウン: ピークからの最大下落率

#### quantile_loss

```python
ql = quantile_loss(y_true, y_pred, alpha=0.05)
```

Pinball損失（分位点回帰の評価指標）。

---

## 8. 評価メトリクス一覧

```python
from penguinboost.metrics.metrics import METRIC_REGISTRY
```

| 名前 | 関数 | 用途 |
|---|---|---|
| `"rmse"` | `rmse(y_true, y_pred)` | 回帰 |
| `"mae"` | `mae(y_true, y_pred)` | 回帰（外れ値に頑健） |
| `"r2"` | `r2_score(y_true, y_pred)` | 回帰（決定係数） |
| `"logloss"` | `logloss(y_true, y_pred)` | 二値分類 |
| `"accuracy"` | `accuracy(y_true, y_pred)` | 分類 |
| `"auc"` | `auc(y_true, y_pred)` | 二値分類 |
| `"ndcg"` | `ndcg_at_k(y_true, y_pred, k)` | ランキング |
| `"c_index"` | `concordance_index(times, events, risk)` | 生存分析 |
| `"sharpe_ratio"` | `sharpe_ratio(returns, rf)` | 金融 |
| `"max_drawdown"` | `max_drawdown(cum_returns)` | 金融 |
| `"quantile_loss"` | `quantile_loss(y_true, y_pred, alpha)` | 分位点回帰 |

---

## 9. scikit-learn 連携

全クラスが `sklearn.base.BaseEstimator` を継承しているため、scikit-learn のエコシステムとシームレスに統合できます。

### 9.1 パイプライン統合

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", PenguinBoostRegressor(n_estimators=100)),
])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

### 9.2 クロスバリデーション

```python
from sklearn.model_selection import cross_val_score

model = PenguinBoostRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"CV MSE: {-scores.mean():.4f} +/- {scores.std():.4f}")
```

金融データの場合は `PurgedKFold` を使用:

```python
from penguinboost.core.financial import PurgedKFold

cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
```

### 9.3 パラメータ操作

```python
model = PenguinBoostRegressor(
    n_estimators=100,
    use_dart=True,
    use_era_boosting=True,
    use_orthogonal_gradients=True,
)

# 全パラメータの取得
params = model.get_params()
print(params["use_era_boosting"])          # True
print(params["orthogonal_strength"])       # 1.0

# パラメータの動的変更
model.set_params(
    n_estimators=200,
    era_boosting_method='sharpe_reweight',
)
```

---

## 10. ユースケース別レシピ

### 10.1 Numerai スタイルリターン予測（全機能）

Numerai Tournament を想定した最もロバストな構成。

```python
import numpy as np
from penguinboost import (
    PenguinBoostRegressor,
    MaxSharpeEraObjective,
    FeatureExposurePenalizedObjective,
    FeatureNeutralizer,
)
from penguinboost.core.boosting import BoostingEngine
from penguinboost.core.financial import PurgedKFold

# ── 戦略1: sklearn API（シンプル）──────────────────────────────────
model = PenguinBoostRegressor(
    # 基本
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    max_leaves=31,

    # 過学習防止
    use_dart=True, dart_drop_rate=0.05,
    use_adaptive_reg=True, adaptive_alpha=0.3,
    subsample=0.8, colsample_bytree=0.7,

    # Era Boosting（era 間の安定性）
    use_era_boosting=True,
    era_boosting_method='sharpe_reweight',
    era_boosting_temp=1.0,

    # 直交勾配射影（特徴量線形効果を排除）
    use_orthogonal_gradients=True,
    orthogonal_strength=0.3,

    # 特徴量露出ペナルティ（学習中の露出抑制）
    use_feature_exposure_penalty=True,
    feature_exposure_lambda=0.05,

    verbose=1,
    random_state=42,
)

# era ラベルを渡して学習
model.fit(X_train, y_train, era_indices=eras_train)

# 予測と後処理
pred_test = model.predict(X_test)

# 学習後中立化（per-era で適用）
pred_neutralized = model.neutralize(
    pred_test, X_test,
    proportion=0.5,
    per_era=True,
    eras=eras_test,
)

# 露出確認
exp = model.feature_exposure(X_test, pred_neutralized)
print(f"Max feature exposure: {np.max(np.abs(exp)):.4f}")

# ── 戦略2: エンジン API + MaxSharpe 目的関数（上級）────────────────
obj = MaxSharpeEraObjective()
obj = FeatureExposurePenalizedObjective(
    base_objective=obj,
    X_ref=X_train,
    lambda_fe=0.1,
)

engine = BoostingEngine(
    n_estimators=500,
    learning_rate=0.03,
    use_era_boosting=True,
    era_boosting_method='hard_era',
    use_orthogonal_gradients=True,
    orthogonal_strength=0.2,
)
engine.fit(X_train, y_train, obj, era_indices=eras_train)
pred_engine = engine.predict(X_test)

# 後処理: per-era 中立化
fn = FeatureNeutralizer()
pred_final = fn.neutralize(
    pred_engine, X_test,
    proportion=1.0,
    per_era=True,
    eras=eras_test,
)
```

### 10.2 金融リターン予測（全部盛り）

```python
from penguinboost import PenguinBoostRegressor
from penguinboost.core.financial import PurgedKFold

model = PenguinBoostRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    max_leaves=63,
    growth="hybrid",
    symmetric_depth=3,
    use_dart=True, dart_drop_rate=0.1,
    use_gradient_perturbation=True,
    gradient_clip_tau=5.0, gradient_noise_eta=0.03,
    use_adaptive_reg=True,
    adaptive_alpha=0.5, adaptive_mu=1.0,
    use_ordered_boosting=True, n_permutations=4,
    use_temporal_reg=True, temporal_rho=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    verbose=1,
    random_state=42,
)

cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
for train_idx, test_idx in cv.split(X):
    split = int(len(train_idx) * 0.8)
    t_idx = train_idx[:split]
    v_idx = train_idx[split:]
    model.fit(
        X[t_idx], y[t_idx],
        eval_set=(X[v_idx], y[v_idx]),
    )
    pred = model.predict(X[test_idx])
```

### 10.3 信用スコアリング（単調制約）

```python
from penguinboost import PenguinBoostClassifier

# 特徴量: 0=年収, 1=勤続年数, 2=延滞回数, 3=借入残高
clf = PenguinBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    monotone_constraints={
        0: -1,   # 年収↑ → デフォルト確率↓
        1: -1,   # 勤続年数↑ → デフォルト確率↓
        2: +1,   # 延滞回数↑ → デフォルト確率↑
        3: +1,   # 借入残高↑ → デフォルト確率↑
    },
    use_adaptive_reg=True,
    use_ordered_boosting=True,
    random_state=42,
)
clf.fit(X_train, y_train)
default_prob = clf.predict_proba(X_test)[:, 1]
```

### 10.4 リスク管理（VaR/CVaR）

```python
from penguinboost import PenguinBoostQuantileRegressor
import numpy as np

models = {}
for alpha in [0.01, 0.05, 0.10]:
    models[alpha] = PenguinBoostQuantileRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=300, learning_rate=0.03,
        use_gradient_perturbation=True,
        gradient_clip_tau=3.0,
        random_state=42,
    )
    models[alpha].fit(X_train, y_train)

var_5pct = models[0.05].predict(X_test)

cvar_model = PenguinBoostQuantileRegressor(
    objective="cvar", alpha=0.05,
    n_estimators=300, learning_rate=0.03,
    random_state=42,
)
cvar_model.fit(X_train, y_train)
cvar_5pct = cvar_model.predict(X_test)

print(f"5% VaR:  {np.mean(var_5pct):.4f}")
print(f"5% CVaR: {np.mean(cvar_5pct):.4f}")
```

### 10.5 大規模データ高速学習

```python
model = PenguinBoostRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    use_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    efb_threshold=0.05,
    colsample_bytree=0.7,
    random_state=42,
)
```

### 10.6 過学習が激しい小データ

```python
model = PenguinBoostRegressor(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    max_leaves=8,
    growth="symmetric",
    reg_lambda=5.0,
    reg_alpha=1.0,
    min_child_weight=10.0,
    min_child_samples=20,
    use_dart=True, dart_drop_rate=0.2,
    use_gradient_perturbation=True,
    gradient_noise_eta=0.1,
    use_adaptive_reg=True,
    adaptive_alpha=1.0, adaptive_mu=3.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
)
```

---

## 11. チューニングガイド

### 11.1 パラメータ優先順位

効果が大きい順にチューニングすることを推奨します。

| 優先度 | パラメータ | チューニング範囲 |
|---|---|---|
| 1 | `n_estimators` + `learning_rate` | (100, 0.1), (300, 0.05), (500, 0.03) |
| 2 | `max_depth` | 3〜10 |
| 3 | `subsample` + `colsample_bytree` | 0.6〜1.0 |
| 4 | `reg_lambda` | 0.1〜10.0 |
| 5 | `min_child_weight` | 1〜50 |
| 6 | `growth` | "leafwise", "hybrid", "symmetric" |
| 7 | 拡張機能群 | 後述 |
| 8 | 金融特化機能群 | 後述 |

### 11.2 目的別推奨設定

| 目的 | 推奨パラメータ |
|---|---|
| 過学習が激しい | `use_dart=True`, `use_adaptive_reg=True`, `adaptive_mu=2〜5` |
| 外れ値が多い | `use_gradient_perturbation=True`, `objective="huber"`, `n_permutations=4` |
| 時系列の安定性 | `use_temporal_reg=True`, `temporal_rho=0.005〜0.05` |
| ドメイン知識あり | `monotone_constraints={...}` |
| リスク管理 | `PenguinBoostQuantileRegressor(objective="cvar", alpha=0.05)` |
| Numerai / era 安定性 | `use_era_boosting=True`, `era_boosting_method='sharpe_reweight'` |
| 特徴量露出を下げたい（学習中） | `use_orthogonal_gradients=True`, `use_feature_exposure_penalty=True` |
| 特徴量露出を下げたい（後処理） | `model.neutralize(pred, X, proportion=0.5, per_era=True, eras=eras)` |
| ランク相関最適化 | `SpearmanObjective(corr_correction=0.5)` |
| era Sharpe 最大化 | `MaxSharpeEraObjective()` |
| 計算効率重視 | `use_goss=True`, `efb_threshold=0.05`, `max_bins=64` |

### 11.3 機能の組み合わせ相性

| 組み合わせ | 相性 | 備考 |
|---|---|---|
| DART + 適応正則化 | 良好 | 二重の過学習抑制 |
| 勾配摂動 + Ordered Boosting | 良好 | 外れ値耐性の相乗効果 |
| 時間的正則化 + ハイブリッド成長 | 良好 | 金融データの安定構成 |
| Era Boosting + MaxSharpe 目的関数 | 良好 | Sharpe 最大化を2重に推進 |
| 直交勾配 + 特徴量露出ペナルティ | 良好 | 学習中の露出抑制を二重に行う |
| 直交勾配 + 学習後中立化 | 優秀 | 学習中と後処理の両方で露出を制御 |
| DART + GOSS | 注意 | 両方でサンプリングが発生。片方だけで十分な場合も |
| 直交勾配射影 + GOSS | 注意 | GOSS で異なるサブセットが選ばれるため射影精度が低下する可能性 |
| 全拡張機能の組み合わせ | 注意 | 正則化が強すぎて学習不足になりうる。各パラメータを控えめに |

---

## 12. アーキテクチャ

### 12.1 ファイル構成

```
penguinboost/
├── __init__.py                 # v0.3.0, 12クラスをexport
├── sklearn_api.py              # scikit-learn互換API（5クラス）
├── utils.py                    # 入力バリデーション
├── core/
│   ├── boosting.py             # BoostingEngine（全機能統合）
│   ├── tree.py                 # DecisionTree（4種成長戦略、バッチ予測）
│   ├── histogram.py            # HistogramBuilder（完全ベクトル化）
│   ├── binning.py              # FeatureBinner + EFB
│   ├── sampling.py             # GOSSSampler
│   ├── categorical.py          # OrderedTargetEncoder
│   ├── regularization.py       # AdaptiveRegularizer, GradientPerturber
│   ├── dart.py                 # DARTManager
│   ├── monotone.py             # MonotoneConstraintChecker
│   ├── financial.py            # PurgedKFold, TemporalRegularizer, RegimeDetector
│   ├── neutralization.py       # FeatureNeutralizer, OrthogonalGradientProjector
│   └── era_boost.py            # EraBoostingReweighter, EraMetrics
├── objectives/
│   ├── __init__.py             # OBJECTIVE_REGISTRY
│   ├── regression.py           # MSE, MAE, Huber
│   ├── classification.py       # BinaryLogloss, Softmax
│   ├── ranking.py              # LambdaRank
│   ├── survival.py             # Cox PH
│   ├── quantile.py             # Quantile, CVaR
│   └── corr.py                 # SpearmanObjective, MaxSharpeEraObjective,
│                                 #   FeatureExposurePenalizedObjective
├── metrics/
│   ├── __init__.py
│   └── metrics.py              # 11種のメトリクス + METRIC_REGISTRY
└── tests/
    └── test_penguinboost.py     # 98テスト
```

### 12.2 学習フロー

```
fit() 呼び出し
  │
  ├─ カテゴリカルエンコーディング (OrderedTargetEncoder)
  ├─ ビニング + EFB (FeatureBinner)
  ├─ 初期予測値の設定 (objective.init_score)
  │
  ├─ OrthogonalGradientProjector.fit() — (X^TX+εI)^{-1} を事前計算
  ├─ EraBoostingReweighter の初期化
  ├─ 特徴量露出ペナルティの事前計算 (X_centered, X_std)
  ├─ MaxSharpeEraObjective に era_indices を設定
  │
  └─ for iteration in range(n_estimators):
       │
       ├─ [DART] 既存木をドロップ → 予測値を調整
       │
       ├─ 勾配・ヘシアン計算 (objective.gradient/hessian)
       │
       ├─ [時間的正則化] 勾配に temporal gradient を加算
       │
       ├─ [Era Boosting] era ごとの Spearman で勾配を重み付け
       │
       ├─ [特徴量露出ペナルティ] 露出削減勾配を加算
       │
       ├─ [直交勾配射影] 特徴量線形成分を除去
       │
       ├─ [勾配摂動] クリッピング + ガウスノイズ
       │
       ├─ [GOSS] 勾配ベースサンプリング
       │   or [サブサンプリング] ランダム行サンプリング
       │
       ├─ [Ordered Boosting] K個の順列で勾配推定 → 中央値集約
       │
       ├─ [列サブサンプリング] 特徴量のランダム選択
       │
       ├─ 木の構築 (DecisionTree.build)
       │   ├─ ヒストグラム構築 (bincount 完全ベクトル化)
       │   ├─ [適応正則化] 子ノードごとの適応λで分割ゲイン計算
       │   ├─ [単調制約] 制約違反の分割を棄却
       │   └─ [ハイブリッド成長] symmetric → leafwise切り替え
       │
       ├─ [DART] 新しい木にスケールファクター適用
       │
       ├─ 予測値を更新 (バッチベクトル化)
       │
       └─ [早期停止] バリデーションスコア監視
```

### 12.3 数学的背景

#### 分割ゲイン（Bayesian Adaptive Split Gain）

```
Gain = ½ [ G_L² / (H_L + λ_L) + G_R² / (H_R + λ_R) - G² / (H + λ) ] - γ

where:
  λ_L = λ_base · (1 + α · t/T) + μ / √n_L
  λ_R = λ_base · (1 + α · t/T) + μ / √n_R
```

#### 特徴量中立化

```
p_neutral = p - proportion · X(X^TX + εI)^{-1}X^T p

  proportion: 露出除去割合 [0, 1]
  ε:          Tikhonov 正則化 (neutralization.eps)
```

#### 直交勾配射影

```
g_orth = g - strength · X(X^TX + εI)^{-1}X^T g

  strength: 射影強度 [0, 1]
  ε:        Tikhonov 正則化 (orthogonal_eps)
```

#### Era Boosting 重み

```
[hard_era]:       w_e = softmax(-ρ_e / T)
[sharpe_reweight]: w_e = softmax(∂Sharpe/∂ρ_e / T)
                  ∂Sharpe/∂ρ_e = (σ² - μ·(ρ_e - μ)) / (n_eras · σ³)
[proportional]:   w_e ∝ 1 - |ρ_e|

サンプル重み: sample_w_i = era_w_e · n_samples / n_in_era   (e = era of sample i)
```

#### Spearman 目的関数

```
L(P, Y) = Σ_i (P_i - r_i)²
r_i = rank_normalize(Y_i) = 2·(rank_i - 1)/(n - 1) - 1  ∈ [-1, 1]

勾配: g_i = (P_i - r_i) - α·(P_i - P̄)·corr(P, r) / (σ_P · σ_r)
  α: corr_correction
```

#### MaxSharpe Era 目的関数

```
maximize  Sharpe(ρ) = μ_era / σ_era

∂Sharpe/∂ρ_e = (σ_era² - μ_era·(ρ_e - μ_era)) / (n_eras · σ_era³)

∂ρ_e/∂P_i ≈ (r_ei - r̄_e) / (n_e · σ_{P_e} · σ_{r_e})   [Pearson approximation]
```

#### 特徴量露出ペナルティ

```
R(P) = λ · Σ_k corr(P, X_k)²

∂R/∂P_i = 2λ · Σ_k ρ_k · [(X_ki - X̄_k)/(n·σ_P·σ_k) - ρ_k·(P_i - P̄)/(n·σ_P²)]

  ρ_k = corr(P, X_k): 現在の予測と特徴量 k の相関
```

#### DART スケーリング

```
η_new = η / (1 + |dropped|)
```

#### Quantile / CVaR 損失

```
[Quantile]: g_i = α - 1{y_i < f_i}
[CVaR]:     g_i = 1 - 1{y_i < f_i} / α
```
