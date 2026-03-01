# 金融回帰の詳細リファレンス

**PenguinBoost — AROGB: Adversarial Regularized Ordered Gradient Boosting**

金融時系列・Numerai 型データに特化した回帰手法の数学的解説と実装ガイド。

---

## 目次

1. [背景と問題設定](#1-背景と問題設定)
2. [損失関数](#2-損失関数)
   - 2.1 [Spearman 相関目的関数](#21-spearman-相関目的関数)
   - 2.2 [MaxSharpe Era 目的関数](#22-maxsharpe-era-目的関数)
   - 2.3 [特徴量露出ペナルティ付き目的関数](#23-特徴量露出ペナルティ付き目的関数)
   - 2.4 [中立化対応目的関数](#24-中立化対応目的関数-neutralizationawareobjective)
   - 2.5 [非対称 Huber 損失](#25-非対称-huber-損失-asymmetrichuberobject)
   - 2.6 [CVaR (Conditional Value at Risk)](#26-cvar-conditional-value-at-risk)
3. [サンプリング戦略](#3-サンプリング戦略)
   - 3.1 [標準 GOSS](#31-標準-goss)
   - 3.2 [時間的重み付き TW-GOSS](#32-時間的重み付き-tw-goss)
4. [勾配の前処理](#4-勾配の前処理)
   - 4.1 [直交勾配射影 (OGP)](#41-直交勾配射影-ogp)
   - 4.2 [Era 適応型勾配クリッピング](#42-era-適応型勾配クリッピング)
5. [木の構築](#5-木の構築)
   - 5.1 [標準分割利得](#51-標準分割利得)
   - 5.2 [Era 敵対的分割基準](#52-era-敵対的分割基準)
6. [DART (Dropout Additive Regression Trees)](#6-dart-dropout-additive-regression-trees)
   - 6.1 [標準 DART](#61-標準-dart)
   - 6.2 [Era 対応 DART](#62-era-対応-dart)
7. [早期終了](#7-早期終了)
   - 7.1 [標準損失ベース早期終了](#71-標準損失ベース早期終了)
   - 7.2 [Sharpe 比ベース早期終了](#72-sharpe-比ベース早期終了)
   - 7.3 [Sharpe 木正則化](#73-sharpe-木正則化)
8. [Era Boosting リウェイティング](#8-era-boosting-リウェイティング)
9. [マルチターゲット補助学習](#9-マルチターゲット補助学習)
10. [共形予測 (Conformal Prediction)](#10-共形予測-conformal-prediction)
11. [特徴量中立化 (事後処理)](#11-特徴量中立化-事後処理)
12. [パラメータリファレンス](#12-パラメータリファレンス)
13. [Numerai 完全ワークフロー例](#13-numerai-完全ワークフロー例)
14. [ハイパーパラメータ チューニングガイド](#14-ハイパーパラメータ-チューニングガイド)

---

## 1. 背景と問題設定

### 1.1 金融回帰の特殊性

一般的な回帰問題と金融回帰の根本的な違いは以下の通り：

| 特性 | 一般回帰 | 金融回帰 |
|------|----------|---------|
| 評価指標 | MSE / MAE | Spearman 相関 / Sharpe 比 |
| 分布 | 任意 | ヘビーテール、歪み |
| 時間構造 | 独立同分布 | エラ（期間）依存 |
| 特徴量リスク | なし | 露出過剰でモデル崩壊 |
| 外れ値 | 一般的な除外処理 | 過剰予測は危険 |
| 評価 | hold-out set | Purged K-Fold + Embargo |

### 1.2 Numerai 型データの定式化

訓練データを次のように定義する：

$$\mathcal{D} = \{(x_i, y_i, e_i)\}_{i=1}^{n}$$

- $x_i \in \mathbb{R}^p$：特徴量ベクトル
- $y_i \in [0,1]$：ターゲット（Numerai では 5 分位に正規化）
- $e_i \in \{1,\ldots,T\}$：エラ（市場週）のラベル

モデルは加法的木の集合 $F(x) = \sum_{m=1}^{M} \eta \cdot h_m(x)$ を学習する。ここで $\eta$ は学習率、$h_m$ は第 $m$ 番目の弱学習器（決定木）。

### 1.3 目標：過学習耐性のある Spearman 相関の最大化

主目標：
$$\text{maximize} \quad \mathbb{E}_{e}\left[\rho_S\!\left(F(X_e), Y_e\right)\right]$$

ここで $\rho_S$ はエラ内 Spearman 順位相関であり、$\mathbb{E}_e$ はエラ間の期待値（= 平均）。

安定性の補助条件：
$$\text{maximize} \quad \text{SR}(F) = \frac{\bar{\rho}}{\sigma_\rho}, \quad \bar{\rho} = \frac{1}{T}\sum_{e=1}^T \rho_e, \quad \sigma_\rho = \text{std}_e(\rho_e)$$

---

## 2. 損失関数

### 2.1 Spearman 相関目的関数

#### 数学的定式化

Spearman 相関を直接最適化するため、**ソフトランク近似**（Blondel et al., 2020）を用いる。

サンプル $\{(y_i, \hat{y}_i)\}_{i=1}^n$ に対して：

**ソフトランク関数**（温度パラメータ $\tau > 0$）：

$$\hat{r}_i(\hat{y}) = \frac{1}{n-1} \sum_{j \neq i} \sigma\!\left(\frac{\hat{y}_i - \hat{y}_j}{\tau}\right)$$

ここで $\sigma(z) = 1/(1+e^{-z})$ はシグモイド関数。$\hat{r}_i \in (0,1)$ は $[0,1]$ に正規化されたソフトランク。

真のランク $r_i = \text{rank}(y_i)/(n-1)$（0-indexed 正規化）に対して：

$$L_{\text{Spearman}} = 1 - \frac{\sum_i (\hat{r}_i - \bar{\hat{r}})(r_i - \bar{r})}{\sqrt{\sum_i (\hat{r}_i - \bar{\hat{r}})^2} \cdot \sqrt{\sum_i (r_i - \bar{r})^2}}$$

**勾配の計算**（自動微分または解析的導出）：

$$g_i = \frac{\partial L}{\partial \hat{y}_i} = \sum_j \frac{\partial L}{\partial \hat{r}_j} \cdot \frac{\partial \hat{r}_j}{\partial \hat{y}_i}$$

$\partial \hat{r}_j / \partial \hat{y}_i$ の計算：

$$\frac{\partial \hat{r}_j}{\partial \hat{y}_i} = \frac{1}{n-1} \cdot \frac{1}{\tau} \sigma'\!\left(\frac{\hat{y}_j - \hat{y}_i}{\tau}\right), \quad (i \neq j)$$

$$\frac{\partial \hat{r}_i}{\partial \hat{y}_i} = -\frac{1}{n-1} \sum_{j \neq i} \frac{1}{\tau} \sigma'\!\left(\frac{\hat{y}_i - \hat{y}_j}{\tau}\right)$$

#### 実装例

```python
from penguinboost.objectives.corr import SpearmanObjective

obj = SpearmanObjective(temperature=0.5, n_pairs_subsample=1000)

# カスタム目的関数として使用
from penguinboost import PenguinBoostRegressor

model = PenguinBoostRegressor(
    objective="spearman",         # 組み込みキーワード
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
)
model.fit(X_train, y_train, era_indices=era_train)
```

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `temperature` | `0.5` | ソフトランク温度 $\tau$。小さいほど真の Spearman に近いが勾配が不安定 |
| `n_pairs_subsample` | `None` | ペア数サブサンプル。$n^2/2$ が大きい時に速度改善 |

---

### 2.2 MaxSharpe Era 目的関数

#### 数学的定式化

エラ $e$ でのモデル予測と真値の Spearman 相関：

$$\rho_e = \rho_S\!\left(\hat{Y}_e, Y_e\right)$$

目標：Sharpe 比の最大化：

$$\text{maximize} \quad \text{SR} = \frac{\mu_\rho}{\sigma_\rho + \epsilon}$$

ここで $\mu_\rho = \frac{1}{T}\sum_e \rho_e$、$\sigma_\rho = \sqrt{\frac{1}{T}\sum_e (\rho_e - \mu_\rho)^2}$。

損失関数（最小化形式）：

$$L = -\frac{\mu_\rho}{\sigma_\rho + \epsilon}$$

勾配（連鎖律）：

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{\sigma_\rho + \epsilon} \cdot \frac{\partial \mu_\rho}{\partial \hat{y}_i} + \frac{\mu_\rho}{(\sigma_\rho + \epsilon)^2} \cdot \frac{\partial \sigma_\rho}{\partial \hat{y}_i}$$

各エラ内の Spearman 勾配 $\frac{\partial \rho_e}{\partial \hat{y}_i}$ はソフトランク近似で計算し、エラ間で集約：

$$\frac{\partial \mu_\rho}{\partial \hat{y}_i} = \frac{1}{T} \cdot \frac{\partial \rho_{e_i}}{\partial \hat{y}_i}$$

$$\frac{\partial \sigma_\rho}{\partial \hat{y}_i} = \frac{\rho_{e_i} - \mu_\rho}{T \cdot \sigma_\rho} \cdot \frac{\partial \rho_{e_i}}{\partial \hat{y}_i}$$

#### 実装例

```python
from penguinboost.objectives.corr import MaxSharpeEraObjective
import numpy as np

obj = MaxSharpeEraObjective(eps=1e-6)

# era_indices: エララベルの配列（整数 or 文字列）
model = PenguinBoostRegressor(
    objective="max_sharpe",
    n_estimators=500,
    learning_rate=0.03,
)
model.fit(X_train, y_train, era_indices=era_train)
```

---

### 2.3 特徴量露出ペナルティ付き目的関数

#### 数学的定式化

モデル予測の各特徴量 $j$ への露出（線形相関）を抑制する：

$$\text{exposure}_j = \left|\text{corr}(\hat{y}, X_{\cdot j})\right|$$

ペナルティ付き損失（$\lambda_{\text{exp}} > 0$）：

$$L = L_{\text{Spearman}} + \lambda_{\text{exp}} \sum_{j=1}^{p} \left|\text{corr}(\hat{y}, X_{\cdot j})\right|^2$$

露出ペナルティの勾配（特徴量 $j$ に対して）：

$$\frac{\partial}{\partial \hat{y}_i} \left|\text{corr}(\hat{y}, X_{\cdot j})\right|^2 = 2 \cdot \text{corr}(\hat{y}, X_{\cdot j}) \cdot \frac{\partial \text{corr}(\hat{y}, X_{\cdot j})}{\partial \hat{y}_i}$$

Pearson 相関の偏微分：

$$\frac{\partial \text{corr}(\hat{y}, x_j)}{\partial \hat{y}_i} = \frac{x_{ij} - \bar{x}_j - \text{corr}(\hat{y}, x_j)(\hat{y}_i - \bar{\hat{y}})}{(n-1) \cdot \text{std}(\hat{y}) \cdot \text{std}(x_j)}$$

#### 実装例

```python
from penguinboost.objectives.corr import FeatureExposurePenalizedObjective

obj = FeatureExposurePenalizedObjective(
    exposure_lambda=0.1,    # ペナルティ強度
    max_exposure=0.1,       # 露出がこの値を超えた特徴量のみペナルティ
)

model = PenguinBoostRegressor(objective=obj, n_estimators=500)
model.fit(X_train, y_train)
```

---

### 2.4 中立化対応目的関数 (NeutralizationAwareObjective)

#### 数学的定式化

**問題**: 特徴量 $X \in \mathbb{R}^{n \times p}$ に対して、予測 $\hat{y}$ が特徴量空間と線形依存している場合、Numerai での評価は「中立化後の相関」 $\rho_S(\hat{y}_\perp, y)$ で行われる。

**中立化（正射影）**:

リッジ正則化付き射影行列：

$$H_X = X(X^\top X + \lambda I)^{-1} X^\top \in \mathbb{R}^{n \times n}$$

中立化予測：

$$\hat{y}_\perp = (I - H_X)\hat{y} = P_\perp \hat{y}$$

ここで $P_\perp = I - H_X$ は特徴量空間の直交補空間への射影。

**損失関数**:

$$L = -\rho_S(\hat{y}_\perp, y) = -\rho_S(P_\perp \hat{y}, y)$$

**勾配の計算（連鎖律）**:

$$\frac{\partial L}{\partial \hat{y}} = \frac{\partial L}{\partial \hat{y}_\perp} \cdot \frac{\partial \hat{y}_\perp}{\partial \hat{y}} = P_\perp^\top \cdot g_\perp$$

ここで $g_\perp = \frac{\partial L}{\partial \hat{y}_\perp}$ は Spearman 損失の中立化予測に対する勾配。

$P_\perp$ は対称行列（$P_\perp^\top = P_\perp$）なので：

$$g = P_\perp g_\perp$$

**ウッドバリー公式による効率的な計算**（$p < n$ の場合）:

直接 $H_X$ を計算すると $O(n^2 p)$。代わりに：

$$P_\perp \hat{y} = \hat{y} - X \underbrace{(X^\top X + \lambda I)^{-1} X^\top \hat{y}}_{v}$$

$v = (X^\top X + \lambda I)^{-1}(X^\top \hat{y})$ をリッジ回帰として解く（$O(p^2 n + p^3)$）。

#### 実装例

```python
from penguinboost.objectives.corr import NeutralizationAwareObjective
import numpy as np

# X_ref: 中立化に使う特徴量行列（訓練データ）
obj = NeutralizationAwareObjective(
    X_ref=X_train,
    lambda_ridge=1e-4,    # リッジ正則化強度
    features=None,         # None = 全特徴量使用。特定列のインデックスリスト可
)

model = PenguinBoostRegressor(objective=obj, n_estimators=500)
model.fit(X_train, y_train)

# 予測も中立化する（推論時）
pred_raw = model.predict(X_test)
pred_neutral = obj.neutralize(pred_raw, X_test)  # 事後中立化
```

#### 計算コスト

| ステップ | 計算量 |
|---------|--------|
| $P_\perp$ の初期化（$X^\top X + \lambda I$）| $O(np^2)$（1回のみ） |
| $v = (X^\top X + \lambda I)^{-1} X^\top g$ の計算 | $O(p^2 + np)$ |
| $P_\perp g = g - Xv$ の計算 | $O(np)$ |

---

### 2.5 非対称 Huber 損失 (AsymmetricHuberObjective)

#### 数学的定式化

**設計思想**: 金融予測では「過剰予測（overprediction）」が過少予測よりも危険なケースがある（例：リスク管理、ポジションサイジング）。非対称 Huber 損失は $\hat{y} > y$ の方向に強いペナルティを課す。

残差を $r = \hat{y} - y$（予測 $-$ 真値）と定義。

**損失関数**（$\delta > 0$、$\kappa \geq 1$）:

$$L_{\text{AH}}(r) = \begin{cases}
\dfrac{1}{2} r^2 & \text{if } r \leq \delta \quad (\text{二次領域})\\[8pt]
\kappa \left(\delta |r| - \dfrac{1}{2}\delta^2\right) & \text{if } r > \delta \quad (\text{線形領域・過剰予測})
\end{cases}$$

**勾配（$\partial L / \partial \hat{y}$）**:

$$g(r) = \begin{cases}
r & \text{if } r \leq \delta \\[4pt]
+\kappa \delta & \text{if } r > \delta
\end{cases}$$

**ヘッセアン（疑似ヘッセ）**:

$$h(r) = \begin{cases}
1 & \text{if } r \leq \delta \\[4pt]
10^{-3} & \text{if } r > \delta
\end{cases}$$

**解釈**:
- $r \leq \delta$（二次領域）: 通常の二乗誤差、勾配 $= r$
- $r > \delta$（線形・過剰予測ゾーン）: 定数勾配 $= \kappa\delta > 0$、モデルを下方修正方向に押す
- $\kappa = 1$: 標準 Huber（対称）
- $\kappa = 2$: 過剰予測に 2 倍の勾配強度

**標準 Huber との比較**:

通常の Huber 損失（対称）は $|r| > \delta$ で $\text{sign}(r) \cdot \delta$ の勾配を与えるため、過少予測も同等にペナルティを受ける。非対称版は $r < -\delta$（過少予測）では二次損失のまま。

#### 実装例

```python
from penguinboost.objectives.regression import AsymmetricHuberObjective

obj = AsymmetricHuberObjective(
    delta=1.0,   # 線形化閾値
    kappa=2.0,   # 過剰予測ペナルティ倍率（>= 1.0）
)

# 直接損失関数として使用
gradients = obj.gradient(y_true, y_pred)
hessians = obj.hessian(y_true, y_pred)

# sklearn API 経由
model = PenguinBoostRegressor(
    objective="asymmetric_huber",
    asymmetric_delta=1.0,
    asymmetric_kappa=2.0,
    n_estimators=300,
)
model.fit(X_train, y_train)
```

#### いつ使うか

- リスク管理モデル：過剰予測によるポジション過大化を防ぐ
- 低流動性銘柄の予測：上振れ予測による損失が非対称なケース
- 欠損・スパースデータ：MSE は外れ値に脆弱、Huber で安定化

---

### 2.6 CVaR (Conditional Value at Risk)

#### 数学的定式化

信頼水準 $\alpha \in (0,1)$（典型的に 0.95）における CVaR（Expected Shortfall）：

$$\text{CVaR}_\alpha(L) = \min_{v \in \mathbb{R}} \left\{ v + \frac{1}{(1-\alpha)n} \sum_{i=1}^{n} \max(L_i - v, 0) \right\}$$

ここで $L_i$ は第 $i$ サンプルの損失、$v^* = \text{VaR}_\alpha(L)$（分位点）。

（Rockafellar & Uryasev, 2000 の同値定式化）

**分位点回帰との関係**:

CVaR 最適化は $\alpha$ 分位点の Pinball 損失（分位点回帰損失）の最小化と一致：

$$L_{\tau}(r) = \begin{cases}
(1-\tau) \cdot (-r) & \text{if } r < 0 \\
\tau \cdot r & \text{if } r \geq 0
\end{cases}$$

ここで $r = y - \hat{y}$、$\tau = \alpha$。

**使用方法**:

```python
from penguinboost import PenguinBoostQuantileRegressor

# VaR の推定（95% 分位）
model_var = PenguinBoostQuantileRegressor(
    quantile=0.95,
    n_estimators=300,
    learning_rate=0.05,
)
model_var.fit(X_train, y_train)
var_pred = model_var.predict(X_test)

# CVaR は VaR 超過損失の平均として事後計算
mask = y_test > var_pred
cvar = y_test[mask].mean() if mask.any() else var_pred.mean()
```

---

## 3. サンプリング戦略

### 3.1 標準 GOSS

**GOSS (Gradient-based One-Side Sampling)** は勾配の大きいサンプルを優先選択する。

アルゴリズム:
1. 全サンプルを $|g_i|$ の降順にソート
2. 上位 $a \times 100\%$ を確定選択（高勾配集合 $A$）
3. 残りから $b \times 100\%$ をランダムサンプリング（低勾配集合 $B$）
4. $B$ の勾配に補正係数 $\frac{1-a}{b}$ を乗算（不偏性の維持）

```python
model = PenguinBoostRegressor(
    goss=True,
    goss_top_rate=0.2,     # a: 上位選択率
    goss_other_rate=0.1,   # b: 低勾配サンプリング率
)
```

### 3.2 時間的重み付き TW-GOSS

#### 数学的定式化

標準 GOSS は時間構造を無視する。金融データでは**最近のエラ**の情報が将来予測により関連性が高い。

**複合重み**（勾配強度 × 時間新鮮度）：

$$w_i = |g_i| \cdot \exp\!\left(-\lambda (t_{\max} - t_i)\right)$$

ここで：
- $|g_i|$：勾配の絶対値（情報量の代理指標）
- $t_i \in \{1,\ldots,T\}$：サンプル $i$ のエラ番号（時間インデックス）
- $t_{\max} = \max_i t_i$：最新エラのインデックス
- $\lambda > 0$：時間減衰率

アルゴリズム:
1. $w_i = |g_i| \cdot \exp(-\lambda(t_{\max} - t_i))$ を計算
2. 上位 $a \times 100\%$ を $w_i$ の降順で選択（集合 $A$）
3. 残りから重み比例サンプリングで $b \times 100\%$ 選択（集合 $B$）
4. 補正係数 $\frac{1-a}{b}$ を $B$ に適用

**$\lambda$ の選択指針**:
- $\lambda = 0$：標準 GOSS（時間非依存）
- $\lambda = 0.01$：穏やかな時間減衰（典型的設定）
- $\lambda = 0.1$：強い時間重み付け（少エラ数・近年データ重視）

#### 実装例

```python
from penguinboost import PenguinBoostRegressor

model = PenguinBoostRegressor(
    use_tw_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    tw_goss_decay=0.01,    # λ: 時間減衰率
    n_estimators=500,
)
model.fit(X_train, y_train, era_indices=era_train)
```

**注意**: `use_tw_goss=True` と `goss=True` は同時使用不可（`use_tw_goss` が優先）。

---

## 4. 勾配の前処理

### 4.1 直交勾配射影 (OGP)

#### 数学的定式化

特徴量行列 $X$ の列空間に対して勾配を直交化する。目標：勾配が特徴量線形結合と相関しないようにする（特徴量露出の低減）。

射影行列（ランク $k$ 近似、$k \ll p$）：

$$P_k = I - \sum_{j=1}^{k} u_j u_j^\top$$

ここで $u_1,\ldots,u_k$ は $X^\top X$ の上位 $k$ 固有ベクトル（主成分軸）。

直交化勾配：

$$g_\perp = P_k g = g - \sum_{j=1}^{k} u_j (u_j^\top g)$$

**テクニカルノート**: 標準化した $X$（各列を $\sqrt{n}$ でスケーリング）に対して SVD を計算するとより安定。

#### 実装例

```python
model = PenguinBoostRegressor(
    use_orthogonal_gradient=True,
    ogp_components=10,      # 直交化する主成分数 k
    n_estimators=500,
)
```

---

### 4.2 Era 適応型勾配クリッピング

#### 数学的定式化

エラごとに外れ値勾配を除去し、エラ間の勾配スケールを均一化する。

**エラ $e$ 内の MAD（Median Absolute Deviation）**:

$$\text{MAD}_e = \text{median}_{i: e_i = e}\left(|g_i - \text{median}_e(g)|\right)$$

**クリッピング**（クリップ係数 $c > 0$）:

$$g_i^{\text{clip}} = \text{sign}(g_i) \cdot \min\!\left(|g_i|,\ c \cdot \text{MAD}_{e_i} + \epsilon\right)$$

$\epsilon > 0$ は数値安定性のための最小値（例：$10^{-8}$）。

**MAD vs. 標準偏差の優位性**:

MAD はガウス分布の $\sigma$ と比較して外れ値に対してロバストであり（$\sigma/\text{MAD} \approx 1.4826$）、金融リターン分布のヘビーテール特性に適する。

#### 実装例

```python
from penguinboost.core.regularization import EraAdaptiveGradientClipper

# 単体使用
clipper = EraAdaptiveGradientClipper(
    clip_multiplier=4.0,   # c: MAD の何倍でクリップするか
    min_mad=1e-8,
)
g_clipped = clipper.clip(gradients, era_indices)

# モデル統合
model = PenguinBoostRegressor(
    use_era_gradient_clipping=True,
    era_clip_multiplier=4.0,
    n_estimators=500,
)
model.fit(X_train, y_train, era_indices=era_train)
```

#### 統計情報の取得

```python
g_clipped, stats = clipper.clip_with_stats(gradients, era_indices)
for era_id, info in stats.items():
    print(f"Era {era_id}: MAD={info['mad']:.4f}, clip_fraction={info['clip_fraction']:.2%}")
```

---

## 5. 木の構築

### 5.1 標準分割利得

決定木の各ノードにおいて、分割 $(k, b)$（特徴量 $k$、ビン境界 $b$）の利得：

$$\text{Gain}(k,b) = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma$$

ここで：
- $G_L = \sum_{i \in L} g_i$、$H_L = \sum_{i \in L} h_i$（左子ノードの勾配・ヘッセ集計）
- $G_R, H_R$：右子ノード同様
- $\lambda$：L2 正則化（葉の値の縮小）
- $\gamma$：分割に対するペナルティ（木の複雑度正則化）

最適葉値：

$$v^* = -\frac{G}{H + \lambda}$$

### 5.2 Era 敵対的分割基準

#### 数学的定式化

**問題**: 標準の利得最大化は特定のエラにのみ有効な分割を選択してしまう（時代特有のパターンへの過学習）。

**エラ間の分散によるペナルティ**:

エラ $e$ での左ノード最適葉値：

$$v^*_{e,L} = -\frac{G_{e,L}}{H_{e,L} + \lambda}$$

エラ間での分散：

$$\text{Var}_e(v^*_L) = \frac{1}{T} \sum_{e=1}^{T} \left(v^*_{e,L} - \bar{v}^*_L\right)^2$$

**修正利得**（$\beta > 0$）:

$$\text{Gain}_{\text{robust}}(k,b) = \text{Gain}(k,b) - \beta \left[\text{Var}_e(v^*_L) + \text{Var}_e(v^*_R)\right]$$

**直感**: 分割後の各ノードの「最適予測値」がエラ間で大きく変動するほど、その分割は時代依存的であり、将来汎化しにくい。$\beta$ でそのペナルティ強度を制御。

#### アルゴリズム

1. 各ノードについてエラ別ヒストグラム $\{(G_{e,k,b}, H_{e,k,b})\}$ を構築（計算量 $O(T \cdot p \cdot B)$）
2. $v^*_{e,k,b,L/R}$ を全 $(k,b)$ について計算
3. $\text{Var}_e(v^*_L) + \text{Var}_e(v^*_R)$ を計算
4. 修正利得で最良分割を選択

#### 実装例

```python
model = PenguinBoostRegressor(
    use_era_adversarial_split=True,
    era_adversarial_beta=0.05,   # β: 分散ペナルティ強度
    n_estimators=300,
)
model.fit(X_train, y_train, era_indices=era_train)
```

#### $\beta$ の選択指針

| $\beta$ | 効果 |
|---------|------|
| `0.0` | 通常の分割（ペナルティなし） |
| `0.01` | 軽微な時代依存性の抑制 |
| `0.05` | 標準的な設定 |
| `0.2` | 強い汎化・精度低下のトレードオフ |

---

## 6. DART (Dropout Additive Regression Trees)

### 6.1 標準 DART

**アイデア**: ニューラルネットワークの Dropout を勾配ブースティングに適用。各反復でランダムに木を「ドロップ」（無効化）して次の木を学習することで過学習を抑制。

アルゴリズム（1 反復）:
1. 既存の $m$ 本の木からドロップ集合 $\mathcal{D} \subseteq [m]$ をサンプリング（$|\mathcal{D}| \sim \text{Binomial}(m, p_{\text{drop}})$）
2. ドロップ木を除いた予測で残差を計算
3. 新しい木 $h_{m+1}$ を学習
4. スケール調整：$h_{m+1}$ に $\frac{|\mathcal{D}|}{|\mathcal{D}|+1}$ を乗算、ドロップ木に $\frac{1}{|\mathcal{D}|+1}$ を乗算して再追加

```python
model = PenguinBoostRegressor(
    dart=True,
    drop_rate=0.1,     # p_drop: 各木のドロップ確率
    skip_drop=0.5,     # この確率でドロップ自体をスキップ
)
```

### 6.2 Era 対応 DART

#### 数学的定式化

**問題**: 標準 DART では全ての木を同一確率でドロップするが、特定エラに特化した木（エラ間分散が大きい木）を優先してドロップすべきである。

**木 $m$ のエラ間分散**（Spearman 相関ベース）:

木 $m$ の単体予測 $h_m(X)$ について、エラ $e$ での Spearman 相関：

$$\rho_{m,e} = \rho_S(h_m(X_e), Y_e)$$

エラ間分散：

$$V_m = \frac{1}{T} \sum_{e=1}^T (\rho_{m,e} - \bar{\rho}_m)^2$$

**ドロップ確率**（中心化シグモイド）:

$$p_{\text{drop}}(m) = \sigma\!\left(s \cdot (V_m - \text{median}_m(V))\right)$$

ここで：
- $\sigma(z) = 1/(1+e^{-z})$：シグモイド関数
- $s > 0$：スケール（感度）パラメータ
- $\text{median}_m(V)$：全木の $V_m$ の中央値

**中心化の重要性**: $V_m$ の値は小さい（例：$0.001$〜$0.01$）ため、$\sigma(s \cdot V_m)$ はほぼ $0.5$ に集中してしまう。中央値を引くことで確率が $[0,1]$ に分散する。

直感：
- $V_m > \text{median}(V)$：エラ依存的な木 → 高ドロップ確率
- $V_m < \text{median}(V)$：汎化的な木 → 低ドロップ確率

#### 実装例

```python
from penguinboost.core.dart import EraAwareDARTManager

# 単体使用
dart = EraAwareDARTManager(
    drop_rate=0.1,
    skip_drop=0.0,
    era_var_scale=20.0,  # s: シグモイドスケール
)

# モデル統合
model = PenguinBoostRegressor(
    use_era_aware_dart=True,
    drop_rate=0.1,
    era_dart_var_scale=20.0,
    n_estimators=500,
)
model.fit(X_train, y_train, era_indices=era_train)
```

---

## 7. 早期終了

### 7.1 標準損失ベース早期終了

検証データの損失が `early_stopping_rounds` 反復間改善しなければ停止。

```python
model = PenguinBoostRegressor(
    early_stopping_rounds=50,
    n_estimators=1000,
)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
print(f"最良反復: {model.engine_.best_iteration_}")
```

### 7.2 Sharpe 比ベース早期終了

#### 数学的定式化

**問題**: 損失ベース早期終了は平均性能を最適化するが、金融では「安定した」モデル（高 Sharpe 比）が望ましい。損失が改善しても Sharpe 比が低下することがある。

**エラ別検証 Spearman 相関**:

$$\rho_e^{\text{val}} = \rho_S(F(X_e^{\text{val}}), Y_e^{\text{val}})$$

**検証 Sharpe 比**:

$$\text{SR}^{\text{val}} = \frac{\bar{\rho}^{\text{val}}}{\sigma_{\rho}^{\text{val}} + \epsilon}$$

**早期終了条件**: `sharpe_es_patience` 回連続で $\text{SR}^{\text{val}}$ が改善しなければ停止。

#### 実装例

```python
model = PenguinBoostRegressor(
    use_sharpe_early_stopping=True,
    sharpe_es_patience=30,     # 改善なしの許容回数
    n_estimators=1000,
    learning_rate=0.05,
)
model.fit(
    X_train, y_train,
    era_indices=era_train,
    X_val=X_val, y_val=y_val,
    era_val=era_val,
)
```

### 7.3 Sharpe 木正則化

#### 数学的定式化

各木の寄与をエラ Sharpe 比で重み付けする。

木 $m$ の単体エラ Spearman 相関：

$$\rho_{m,e} = \rho_S(\eta \cdot h_m(X_e^{\text{train}}), Y_e^{\text{train}})$$

木 $m$ の Sharpe 比：

$$\text{SR}_m = \frac{\bar{\rho}_m}{\sigma_{\rho,m} + \epsilon}$$

**スケール調整**（閾値 $\theta$）:

$$\tilde{h}_m = \begin{cases}
h_m & \text{if } \text{SR}_m \geq \theta \\[4pt]
h_m \cdot \dfrac{\text{SR}_m}{\theta} & \text{if } \text{SR}_m < \theta
\end{cases}$$

直感：Sharpe 比が閾値を下回る木は寄与を縮小し、過学習の木の影響を低減。

#### 実装例

```python
model = PenguinBoostRegressor(
    use_sharpe_tree_reg=True,
    sharpe_reg_threshold=0.0,   # θ: Sharpe 閾値（負でも可）
    n_estimators=500,
)
model.fit(X_train, y_train, era_indices=era_train)
```

---

## 8. Era Boosting リウェイティング

#### 数学的定式化

エラ間の不均一な性能を補正する。「苦手なエラ」のサンプルに高い重みを与える。

**パフォーマンス指標**（エラ $e$ のスコア $s_e$）:

$$s_e = \rho_S(F(X_e), Y_e)$$

**再重み付け**（AdaBoost 類似）:

$$w_i^{(t+1)} \propto \exp(-\alpha \cdot s_{e_i})$$

ここで $\alpha > 0$ は増幅率。パフォーマンスが低いエラ（$s_e$ が小さい）のサンプルに高い重みが割り当てられる。

#### 実装例

```python
from penguinboost import EraBoostingReweighter, EraMetrics

reweighter = EraBoostingReweighter(
    alpha=1.0,              # 増幅強度
    normalize=True,         # 重みの合計が n になるよう正規化
    clip_ratio=10.0,        # 最大/最小重みの比率上限
)

# 現在の予測からサンプル重みを計算
era_metrics = EraMetrics()
weights = reweighter.compute_weights(
    predictions=F_pred,
    targets=y_train,
    era_indices=era_train,
)

# 次ラウンドの訓練に使用
model.fit(X_train, y_train, sample_weight=weights)
```

---

## 9. マルチターゲット補助学習

#### 数学的定式化

**問題**: メインターゲット（例：Numerai `target`）のサンプル数が少ない場合、補助ターゲット（例：`target_jerome`, `target_david` など相関のある将来ターン）を活用して汎化を改善する。

**マルチターゲット構造**:

- メインターゲット $y \in \mathbb{R}^n$
- 補助ターゲット行列 $Y_{\text{aux}} \in \mathbb{R}^{n \times K}$（$K$ 個の補助ターゲット）

**混合勾配**:

$$g_{\text{mixed}} = \alpha(t) \cdot g_{\text{main}} + (1-\alpha(t)) \cdot \frac{1}{K} \sum_{k=1}^{K} g_{\text{aux},k}$$

ここで：
- $g_{\text{main}} = \nabla_{\hat{y}} L_{\text{main}}(y, \hat{y})$：メイン損失の勾配
- $g_{\text{aux},k} = \nabla_{\hat{y}} L_{\text{aux},k}(y_{\text{aux},k}, \hat{y})$：$k$ 番目補助損失の勾配
- $\alpha(t)$：反復 $t$ における混合係数

**動的スケジュール**（`use_schedule=True` の場合）:

$$\alpha(t) = \alpha_{\text{start}} + (\alpha_{\text{end}} - \alpha_{\text{start}}) \cdot \left(\frac{t}{M}\right)^{p}$$

ここで $M$ は総反復数、$p > 0$ はスケジュールのべき乗（$p=1$ でリニア、$p>1$ で遅い増加）。

典型的設定：
- 初期（$t \approx 0$）：$\alpha = 0.3$（補助ターゲットを重視し、共通特徴を学習）
- 後期（$t \approx M$）：$\alpha = 0.9$（メインターゲットに集中）

#### 実装例

```python
from penguinboost import PenguinBoostRegressor, MultiTargetAuxiliaryObjective
from penguinboost.objectives.corr import SpearmanObjective

# メイン目的関数
main_obj = SpearmanObjective(temperature=0.5)

# マルチターゲット目的関数（動的スケジュール付き）
multi_obj = MultiTargetAuxiliaryObjective(
    main_objective=main_obj,
    alpha=0.7,              # 固定混合係数（スケジュールなし）
    use_schedule=True,
    alpha_start=0.3,        # 初期 α
    n_estimators=500,
    schedule_power=1.0,     # スケジュールのべき乗
)

# 補助ターゲットを設定
aux_objectives = [SpearmanObjective() for _ in range(K)]
multi_obj.set_aux_targets(Y_aux=Y_aux_train, aux_objectives=aux_objectives)

# モデル訓練
model = PenguinBoostRegressor(
    objective=multi_obj,
    n_estimators=500,
    learning_rate=0.05,
)
model.fit(X_train, y_main_train)
```

#### 実用的な注意点

1. 補助ターゲットの相関を事前確認（Spearman 相関 > 0.3 を目安）
2. `alpha_start` を低くしすぎると補助ターゲットに引っ張られてメインが低下
3. `schedule_power=2.0` で後半にメイン集中を加速できる

---

## 10. 共形予測 (Conformal Prediction)

#### 数学的定式化

**目標**: 点予測 $\hat{y}$ に対して、有効な信頼区間 $[L_i, U_i]$ を提供する。

$$P(y_i \in [L_i, U_i]) \geq 1 - \alpha$$

**スプリット共形予測**（Split Conformal）:

1. データを訓練セット $\mathcal{D}_{\text{train}}$ とキャリブレーションセット $\mathcal{D}_{\text{cal}}$ に分割
2. モデルを $\mathcal{D}_{\text{train}}$ で訓練
3. キャリブレーションスコアを計算：$s_i = |y_i - \hat{y}_i|$
4. **有限サンプル補正**付き分位点：
   $$q = \text{Quantile}\!\left(\{s_i\}_{i \in \mathcal{D}_{\text{cal}}},\ \frac{\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil}{n_{\text{cal}}}\right)$$
5. 予測区間：$[L_i, U_i] = [\hat{y}_i - q, \hat{y}_i + q]$

**有限サンプル補正の必要性**: 標準の $(1-\alpha)$ 分位点では有限標本でのカバレッジが保証されない。$\lceil(n+1)(1-\alpha)\rceil/n$ 補正でカバレッジ $\geq 1-\alpha$ が数学的に保証される（Vovk et al., 2005）。

**非対称区間**（`asymmetric=True`）:

上限・下限を別々に計算：

$$s_i^+ = \max(y_i - \hat{y}_i, 0), \quad s_i^- = \max(\hat{y}_i - y_i, 0)$$

$$q^+ = \text{Quantile}(s^+,\ \cdot),\quad q^- = \text{Quantile}(s^-,\ \cdot)$$

$$[L_i, U_i] = [\hat{y}_i - q^-, \hat{y}_i + q^+]$$

#### 実装例

```python
from penguinboost.core.conformal import ConformalPredictor, EraConformalPredictor

# 訓練・キャリブレーション分割
model.fit(X_train, y_train)
pred_cal = model.predict(X_cal)

# 標準共形予測
cp = ConformalPredictor(alpha=0.1, asymmetric=False)
cp.calibrate(y_cal, pred_cal)

# テスト予測区間
pred_test = model.predict(X_test)
lower, upper = cp.predict_interval(pred_test)
coverage = cp.empirical_coverage(y_test, pred_test)
print(f"実測カバレッジ: {coverage:.1%} (目標: {1-cp.alpha:.1%})")
```

#### Era 別共形予測

金融データではエラごとに予測誤差の分布が異なる。`EraConformalPredictor` はエラ別にキャリブレーションを行う：

```python
cp_era = EraConformalPredictor(alpha=0.1, min_era_samples=20)
cp_era.calibrate(y_cal, pred_cal, era_cal)

lower, upper = cp_era.predict_interval(pred_test, era_test)
```

エラが未知の場合（新しいエラ）は全体のキャリブレーション分位点にフォールバック。

---

## 11. 特徴量中立化 (事後処理)

#### 数学的定式化

モデル出力を特定の特徴量に対して直交化する（予測が特徴量の線形結合に依存しないようにする）。

**中立化変換**:

$$\hat{y}_\perp = \hat{y} - X_{\text{neutral}} \cdot (X_{\text{neutral}}^\top X_{\text{neutral}})^{-1} X_{\text{neutral}}^\top \hat{y}$$

数値安定化のためリッジ正則化を加える：

$$\hat{y}_\perp = \hat{y} - X(X^\top X + \lambda I)^{-1} X^\top \hat{y}$$

**部分中立化**（強度パラメータ $\gamma \in [0,1]$）:

$$\hat{y}_\perp^{(\gamma)} = \gamma \hat{y}_\perp + (1-\gamma) \hat{y}$$

$\gamma=0$：中立化なし、$\gamma=1$：完全中立化。

#### 実装例

```python
from penguinboost import FeatureNeutralizer
import numpy as np

# モデル訓練
pred_test = model.predict(X_test)

# 事後中立化
neutralizer = FeatureNeutralizer(
    proportion=0.5,      # γ: 中立化強度
    by_era=True,         # エラ別に中立化（推奨）
)
pred_neutral = neutralizer.neutralize(
    predictions=pred_test,
    features=X_test,
    era_indices=era_test,
)
```

---

## 12. パラメータリファレンス

### PenguinBoostRegressor 金融機能パラメータ

#### 損失関数

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `objective` | `str/obj` | `"mse"` | `"spearman"`, `"max_sharpe"`, `"asymmetric_huber"`, カスタム目的関数オブジェクト |
| `asymmetric_delta` | `float` | `1.0` | AsymmetricHuber の $\delta$（線形化閾値） |
| `asymmetric_kappa` | `float` | `2.0` | AsymmetricHuber の $\kappa \geq 1$（過剰予測ペナルティ倍率） |

#### サンプリング

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `goss` | `bool` | `False` | 標準 GOSS を有効化 |
| `goss_top_rate` | `float` | `0.2` | 高勾配上位選択率 $a$ |
| `goss_other_rate` | `float` | `0.1` | 低勾配サンプリング率 $b$ |
| `use_tw_goss` | `bool` | `False` | 時間的重み付き GOSS を有効化（`goss` より優先） |
| `tw_goss_decay` | `float` | `0.01` | 時間減衰率 $\lambda$ |

#### 勾配前処理

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `use_orthogonal_gradient` | `bool` | `False` | 直交勾配射影を有効化 |
| `ogp_components` | `int` | `10` | 直交化する主成分数 $k$ |
| `use_era_gradient_clipping` | `bool` | `False` | Era MAD クリッピングを有効化 |
| `era_clip_multiplier` | `float` | `4.0` | クリップ係数 $c$ |

#### 木の構築

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `use_era_adversarial_split` | `bool` | `False` | Era 敵対的分割を有効化 |
| `era_adversarial_beta` | `float` | `0.05` | エラ間分散ペナルティ強度 $\beta$ |

#### DART

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `dart` | `bool` | `False` | 標準 DART を有効化 |
| `use_era_aware_dart` | `bool` | `False` | Era 対応 DART を有効化（`dart` より優先） |
| `drop_rate` | `float` | `0.1` | 基本ドロップ確率 |
| `skip_drop` | `float` | `0.0` | ドロップをスキップする確率 |
| `era_dart_var_scale` | `float` | `20.0` | シグモイドスケール $s$ |

#### 早期終了・正則化

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `use_sharpe_early_stopping` | `bool` | `False` | Sharpe 比ベース早期終了を有効化 |
| `sharpe_es_patience` | `int` | `30` | 改善なし許容反復数 |
| `use_sharpe_tree_reg` | `bool` | `False` | Sharpe 木正則化を有効化 |
| `sharpe_reg_threshold` | `float` | `0.0` | Sharpe 閾値 $\theta$ |

---

## 13. Numerai 完全ワークフロー例

```python
import numpy as np
import pandas as pd
from penguinboost import (
    PenguinBoostRegressor,
    FeatureNeutralizer,
    MultiTargetAuxiliaryObjective,
    EraConformalPredictor,
)
from penguinboost.objectives.corr import SpearmanObjective, NeutralizationAwareObjective

# ============================================================
# 1. データ準備
# ============================================================
# Numerai データを想定: features, target, era 列を含む DataFrame
# df_train, df_val はエラ列 "era" を持つ

feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
target_col = "target"
era_col = "era"

X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train[target_col].values.astype(np.float32)
era_train = df_train[era_col].values

X_val = df_val[feature_cols].values.astype(np.float32)
y_val = df_val[target_col].values.astype(np.float32)
era_val = df_val[era_col].values

# 補助ターゲット（複数ターゲット訓練）
aux_target_cols = ["target_jerome", "target_david", "target_waldo"]
Y_aux = df_train[aux_target_cols].values.astype(np.float32)

# ============================================================
# 2. 目的関数の構築
# ============================================================
# 選択肢A: 中立化対応 Spearman（Numerai でおすすめ）
main_obj = NeutralizationAwareObjective(
    X_ref=X_train,
    lambda_ridge=1e-4,
)

# 選択肢B: 標準 Spearman
# main_obj = SpearmanObjective(temperature=0.5)

# マルチターゲット包含
multi_obj = MultiTargetAuxiliaryObjective(
    main_objective=main_obj,
    use_schedule=True,
    alpha_start=0.3,    # 初期: 補助ターゲット重視
    n_estimators=600,
    schedule_power=1.5,
)
multi_obj.set_aux_targets(
    Y_aux=Y_aux,
    aux_objectives=[SpearmanObjective() for _ in range(len(aux_target_cols))],
)

# ============================================================
# 3. モデル構築（全金融機能を組み合わせ）
# ============================================================
model = PenguinBoostRegressor(
    # 基本パラメータ
    objective=multi_obj,
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    reg_lambda=0.1,

    # サンプリング: TW-GOSS
    use_tw_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    tw_goss_decay=0.01,

    # 勾配前処理
    use_era_gradient_clipping=True,
    era_clip_multiplier=4.0,
    use_orthogonal_gradient=True,
    ogp_components=10,

    # Era 敵対的分割
    use_era_adversarial_split=True,
    era_adversarial_beta=0.05,

    # Era 対応 DART
    use_era_aware_dart=True,
    drop_rate=0.1,
    era_dart_var_scale=20.0,

    # 早期終了
    use_sharpe_early_stopping=True,
    sharpe_es_patience=40,
    early_stopping_rounds=50,

    # Sharpe 木正則化
    use_sharpe_tree_reg=True,
    sharpe_reg_threshold=0.0,

    # 再現性
    random_state=42,
)

# ============================================================
# 4. 訓練
# ============================================================
model.fit(
    X_train, y_train,
    era_indices=era_train,
    X_val=X_val,
    y_val=y_val,
    era_val=era_val,
)
print(f"Best iteration: {model.engine_.best_iteration_}")

# ============================================================
# 5. 予測と後処理
# ============================================================
pred_val_raw = model.predict(X_val)

# 事後中立化（Numerai 推奨）
neutralizer = FeatureNeutralizer(proportion=0.5, by_era=True)
pred_val = neutralizer.neutralize(pred_val_raw, X_val, era_val)

# ============================================================
# 6. 共形予測区間（信頼区間）
# ============================================================
# キャリブレーション = 検証データを再利用
cp = EraConformalPredictor(alpha=0.1, min_era_samples=20)
cp.calibrate(y_val, pred_val_raw, era_val)

# テストデータに適用
X_test = df_test[feature_cols].values.astype(np.float32)
era_test = df_test[era_col].values
pred_test_raw = model.predict(X_test)
lower, upper = cp.predict_interval(pred_test_raw, era_test)

pred_test = neutralizer.neutralize(pred_test_raw, X_test, era_test)

# ============================================================
# 7. 評価
# ============================================================
from scipy.stats import spearmanr

era_scores = {}
for era in np.unique(era_val):
    mask = era_val == era
    rho, _ = spearmanr(pred_val[mask], y_val[mask])
    era_scores[era] = rho

mean_corr = np.mean(list(era_scores.values()))
std_corr = np.std(list(era_scores.values()))
sharpe = mean_corr / (std_corr + 1e-8)
print(f"Mean Corr: {mean_corr:.4f}, Std: {std_corr:.4f}, Sharpe: {sharpe:.4f}")
```

---

## 14. ハイパーパラメータ チューニングガイド

### 14.1 優先度の高いパラメータ

最初に調整すべきパラメータ（影響が大きい順）：

| 優先度 | パラメータ | 探索範囲 | 指針 |
|--------|-----------|---------|------|
| 1 | `learning_rate` | [0.01, 0.1] | 小さいほど汎化・遅い収束 |
| 2 | `max_depth` | [4, 8] | 金融データは 5-6 が多い |
| 3 | `num_leaves` | [15, 127] | 2^(max_depth) を超えないこと |
| 4 | `min_child_samples` | [10, 50] | 大きいほど過学習抑制 |
| 5 | `reg_lambda` | [0.01, 1.0] | L2 正則化（葉の値） |

### 14.2 金融特有パラメータの推奨初期値

| パラメータ | 推奨初期値 | 調整方向 |
|-----------|-----------|---------|
| `tw_goss_decay` | `0.01` | データのエラ数が少ない → 大きく |
| `era_clip_multiplier` | `4.0` | ヘビーテール → 小さく（2.0〜3.0） |
| `ogp_components` | `10` | 特徴量数の 10〜20% |
| `era_adversarial_beta` | `0.05` | 汎化重視 → 大きく（0.1〜0.2） |
| `era_dart_var_scale` | `20.0` | エラ数が少ない → 小さく |
| `sharpe_es_patience` | `30-50` | `n_estimators` の 5〜10% |

### 14.3 Optuna による自動チューニング例

```python
import optuna

def objective(trial):
    params = {
        "objective": "spearman",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "use_tw_goss": True,
        "tw_goss_decay": trial.suggest_float("tw_goss_decay", 0.001, 0.1, log=True),
        "use_era_gradient_clipping": True,
        "era_clip_multiplier": trial.suggest_float("era_clip_multiplier", 2.0, 6.0),
        "use_era_adversarial_split": True,
        "era_adversarial_beta": trial.suggest_float("era_adversarial_beta", 0.01, 0.2),
        "use_sharpe_early_stopping": True,
        "sharpe_es_patience": 30,
        "random_state": 42,
    }
    model = PenguinBoostRegressor(**params)
    model.fit(X_train, y_train, era_indices=era_train,
              X_val=X_val, y_val=y_val, era_val=era_val)

    pred_val = model.predict(X_val)
    # エラ Spearman Sharpe を最大化
    era_corrs = []
    for era in np.unique(era_val):
        mask = era_val == era
        rho, _ = spearmanr(pred_val[mask], y_val[mask])
        era_corrs.append(rho)
    sharpe = np.mean(era_corrs) / (np.std(era_corrs) + 1e-8)
    return sharpe  # maximize

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(f"Best Sharpe: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 14.4 機能の組み合わせガイドライン

| シナリオ | 推奨機能組み合わせ |
|---------|-----------------|
| Numerai 標準 | `SpearmanObjective` + `use_tw_goss` + `use_era_gradient_clipping` |
| 汎化最優先 | 上記 + `use_era_adversarial_split` + `use_era_aware_dart` |
| 安定性重視 | 上記 + `use_sharpe_early_stopping` + `use_sharpe_tree_reg` |
| 完全最適化 | 上記 + `NeutralizationAwareObjective` + `MultiTargetAuxiliaryObjective` |
| リスク管理 | `AsymmetricHuberObjective` (kappa=3) + `EraConformalPredictor` |

---

## 参考文献

1. **Blondel et al. (2020)**: "Fast Differentiable Sorting and Ranking" — ソフトランク近似
2. **Rockafellar & Uryasev (2000)**: "Optimization of Conditional Value-at-Risk" — CVaR 定式化
3. **Rashmi & Gilad-Bachrach (2015)**: "DART: Dropouts meet Multiple Additive Regression Trees" — DART
4. **Ke et al. (2017)**: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" — GOSS, ヒストグラム法
5. **Vovk et al. (2005)**: "Algorithmic Learning in a Random World" — 共形予測の理論
6. **Shafer & Vovk (2008)**: "A Tutorial on Conformal Prediction" — 有限サンプル補正
7. **Caruana (1997)**: "Multitask Learning" — 補助学習
