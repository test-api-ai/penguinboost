# PenguinBoost


## 必要環境

| 要件 | バージョン |
|---|---|
| Python | >= 3.9 |
| NumPy | >= 1.21 |
| scikit-learn | >= 1.0 |
| C++ コンパイラ | C++17 対応（GCC 7+ / Clang 5+ / MSVC 2017+） |
| pybind11 | >= 2.10（ビルド時のみ） |

> **注意:** コアの計算エンジンは C++ で実装されており、インストール時に C++ のコンパイルが行われます。コンパイラが事前にインストールされている必要があります。
> - macOS: `xcode-select --install`
> - Ubuntu/Debian: `sudo apt install build-essential`
> - Windows: Visual Studio Build Tools（C++ ワークロード付き）

## インストール

```bash
# pybind11 を先にインストール
pip install pybind11

# ソースからインストール（C++ 拡張モジュールがビルドされます）
git clone https://github.com/test-api-ai/penguinboost.git
cd penguinboost
pip install .
```

開発用（編集可能インストール）:

```bash
pip install -e ".[dev]"
```

## クイックスタート

### 回帰

```python
from penguinboost import PenguinBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PenguinBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, pred):.4f}")
```

### クラス分類

```python
from penguinboost import PenguinBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = PenguinBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")

# クラス確率を取得
proba = clf.predict_proba(X_test)  # shape=(n_samples, n_classes)
```

## 主な機能

- **多様な学習タスク**: 回帰・二値/多クラス分類・ランキング・生存分析・分位点回帰
- **4種の木成長戦略**: `leafwise`（LightGBM 式）、`symmetric`（CatBoost 式）、`depthwise`、`hybrid`
- **過学習対策**: DART、勾配摂動、ベイズ適応正則化、単調制約
- **金融特化機能**: Era Boosting、特徴量中立化、直交勾配射影、MaxSharpe 目的関数
- **scikit-learn 互換**: `Pipeline`・`cross_val_score` 等とシームレスに連携

詳細は [`docs/USAGE.md`](docs/USAGE.md) を参照してください。

## テスト

```bash
pip install -e ".[dev]"
pytest tests/
```

## ライセンス

MIT License — Copyright (c) 2026 S.I

詳細は [LICENSE](LICENSE) を参照してください。

---

## 使用ライブラリ・著作権表示

PenguinBoost は以下のサードパーティライブラリを使用しています。

### ランタイム依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Copyright (c) 2005-2024, NumPy Developers |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause | Copyright (c) 2007-2024, scikit-learn developers |

### ビルド時依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [pybind11](https://github.com/pybind/pybind11) | BSD 3-Clause | Copyright (c) 2016 Wenzel Jakob and others |

---

## アルゴリズム参照・著作権表示

PenguinBoost は以下のプロジェクトで発表されたアルゴリズムや手法を参考に実装しています。
これらのプロジェクトのソースコードは使用していません。

### XGBoost

- **Project**: https://github.com/dmlc/xgboost
- **License**: Apache 2.0
- **Reference**: Tianqi Chen and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD 2016. https://arxiv.org/abs/1603.02754
- **参考にした手法**: L1/L2 正則化付き葉の重み計算式、ヘシアンベースの分割ゲイン、最小子ノード制約

### LightGBM

- **Project**: https://github.com/microsoft/LightGBM
- **License**: MIT
- **Reference**: Guolin Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- **参考にした手法**: ヒストグラムベース分割探索・差分トリック、Leaf-wise 木成長、GOSS（勾配ベースサンプリング）、EFB（排他的特徴量バンドリング）

### CatBoost

- **Project**: https://github.com/catboost/catboost
- **License**: Apache 2.0
- **Reference**: Liudmila Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features." NeurIPS 2018. https://arxiv.org/abs/1706.09516
- **参考にした手法**: Ordered Boosting（予測シフト低減）、Ordered Target Encoding（カテゴリカル特徴量）、対称木（Oblivious Tree）成長

### Numerai

- **Project**: https://numer.ai
- **GitHub**: https://github.com/numerai
- **参考にした手法**: Era ベースのクロスバリデーションと per-era 性能加重、特徴量中立化（線形特徴量成分の除去）、特徴量露出ペナルティ、Era 間 Sharpe 比最大化
