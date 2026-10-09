# 第07章 矩陣微分、VJP與反向傳播

## 學習目標與先備知識

本章把「函數可微」轉化為「模型可以有效訓練」。完成本章後，讀者應能：

1. 區分導數、Jacobian、梯度、Jacobian-vector product（JVP）與 vector-Jacobian product（VJP）。
2. 使用微分與 Frobenius 內積推導矩陣運算的梯度。
3. 依計算圖的反向拓撲順序執行反向傳播。
4. 在一個中間值被多條路徑使用時，正確累加所有梯度貢獻。
5. 不建立大型完整 Jacobian，直接計算 VJP。
6. 用中央有限差分及內積對偶檢查解析梯度。
7. 明確區分訓練損失、驗證指標與測試結果，不把低訓練誤差直接解讀為泛化能力。

先備知識包括矩陣乘法、轉置、基本偏導數及 NumPy 陣列操作。本卷約定矩陣橫列為 row、縱行為 column。批次資料採

$$
X\in\mathbb{R}^{B\times D_{\mathrm{in}}},
$$

每筆樣本儲存在一個 row；仿射映射寫成

$$
Y=XW+b,
$$

其中

$$
W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}},
\qquad
b\in\mathbb{R}^{D_{\mathrm{out}}}.
$$

偏置 $b$ 沿 batch 軸廣播。這與微分教材常把單一向量寫成 column 向量並不衝突；關鍵在於每一式的 shape 必須一致，而且梯度與被微分變數具有相同 shape。

---

## 問題與直覺

設一個模型依序進行

$$
X\longrightarrow Z=XW+b\longrightarrow A=\tanh Z\longrightarrow L.
$$

若參數 $W$ 有百萬個元素，輸出 $A$ 也有大量元素，直接建立「每個輸出對每個參數的偏導數」會產生巨大 Jacobian。訓練通常不需要這個完整矩陣，因為最終目標是純量損失 $L$。我們真正需要的是：

> 損失的一個微小變化，如何沿計算圖反向分配到每個參數？

反向傳播是鏈式法則的有效實作。每個節點接收來自下游的梯度，利用局部導數計算對其輸入的 VJP，再把結果傳向上游。若同一值流向多個分支，各分支都會對它的梯度作出貢獻，因此必須相加，而不是覆寫。

![反向傳播示意](../figures/backprop.svg)

### 模型、目標與證據契約

為避免把「程式能執行」誤當成「模型具備能力」，本章採以下實驗契約：

- **資料生成**：只使用程式產生的合成二分類資料，不下載模型、語料或 tokenizer。
- **生成規則**：特徵 $x\in\mathbb{R}^2$ 由標準常態抽樣，條件標籤由
  $P(y=1\mid x)=\sigma(x_1-0.7x_2+0.2)$ 定義。
- **機率域**：$\sigma(s)=1/(1+e^{-s})\in(0,1)$，抽樣標籤屬於 $\{0,1\}$。有限樣本中的類別比例不是條件真分布本身。
- **固定種子的範圍**：種子只固定該次 NumPy 隨機數流程，不保證跨 NumPy 版本、平台或浮點實作逐位一致。
- **資料切分**：訓練、驗證、測試索引互不重疊。只用訓練集更新參數；驗證集可供模型選擇；測試集不參與學習率、步數或架構調整。
- **損失平均**：二元交叉熵沿 batch 軸取 mean，反向時只除以 $B$ 一次。
- **基線**：以訓練集多數類別作固定預測，再分別評估驗證集及測試集。
- **梯度驗收**：逐參數有限差分或方向導數若超過明列容差，程式必須中止，不得繼續訓練。
- **故障測試**：非法標籤、空 batch、非有限 logit 若未被拒絕，測試本身必須失敗。
- **泛化限制**：低訓練損失只表示模型適合訓練資料，不保證驗證、測試或分布外資料表現。

本章程式沒有執行紀錄，因此後文只能陳述預期行為，不能聲稱測試已通過或模型已訓練。

---

## 定義、定理與推導

### 微分、梯度與 Frobenius 內積

對純量函數 $f(X)$，其中 $X\in\mathbb{R}^{m\times n}$，梯度定義為與 $X$ 同 shape 的矩陣 $\nabla_Xf$，使一階微分可寫成

$$
df=\operatorname{tr}\left((\nabla_Xf)^T\,dX\right).
$$

矩陣的 Frobenius 內積定義為

$$
\langle A,B\rangle_F
=\operatorname{tr}(A^TB)
=\sum_{i,j}A_{ij}B_{ij}.
$$

因此

$$
df=\langle\nabla_Xf,dX\rangle_F.
$$

這個表示法把「每個元素的偏導數」組成與 $X$ 同 shape 的矩陣。若 $W$ 的 shape 是 $(D_{\mathrm{in}},D_{\mathrm{out}})$，則 $\nabla_Wf$ 也必須是同一 shape。

### Jacobian、JVP與VJP

考慮向量函數

$$
y=f(x),
\qquad
x\in\mathbb{R}^n,
\quad
y\in\mathbb{R}^m.
$$

其 Jacobian 為

$$
J_f(x)_{ij}
=
\frac{\partial y_i}{\partial x_j},
\qquad
J_f(x)\in\mathbb{R}^{m\times n}.
$$

給定輸入方向 $u\in\mathbb{R}^n$，JVP 是

$$
J_f(x)u\in\mathbb{R}^m.
$$

它描述輸入沿 $u$ 微小移動時，輸出的一階變化方向。若 $f$ 可微，則

$$
f(x+\epsilon u)
=
f(x)+\epsilon J_f(x)u+o(\epsilon).
$$

給定輸出端向量 $v\in\mathbb{R}^m$，VJP 是

$$
v^TJ_f(x)\in\mathbb{R}^{1\times n}.
$$

若梯度統一表示成 column 向量，反向傳播通常寫成

$$
\nabla_xL
=
J_f(x)^T\nabla_yL.
$$

這正是 VJP 的轉置表示。反向實作只需要知道如何把上游梯度 $\nabla_yL$ 映回 $\nabla_xL$，不必配置完整的 $m\times n$ Jacobian。

### 小命題：JVP與VJP的內積對偶

**命題。** 若 $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ 在 $x$ 可微，則對任意 $u\in\mathbb{R}^n$ 與 $v\in\mathbb{R}^m$，

$$
\langle v,J_f(x)u\rangle
=
\langle J_f(x)^Tv,u\rangle.
$$

**證明。**

由歐氏內積定義，

$$
\langle v,J_f(x)u\rangle
=
v^TJ_f(x)u.
$$

右側是 $1\times1$ 純量，因此等於其轉置：

$$
v^TJ_f(x)u
=
\left(v^TJ_f(x)u\right)^T.
$$

利用 $(ABC)^T=C^TB^TA^T$，

$$
\left(v^TJ_f(x)u\right)^T
=
u^TJ_f(x)^Tv.
$$

再由內積的定義及對稱性，

$$
u^TJ_f(x)^Tv
=
\langle u,J_f(x)^Tv\rangle
=
\langle J_f(x)^Tv,u\rangle.
$$

故

$$
\langle v,J_f(x)u\rangle
=
\langle J_f(x)^Tv,u\rangle.
$$

證畢。

這個命題可用來核對反向傳播。左側的 JVP 可由前向方向有限差分近似，右側的 VJP 可由解析反傳取得，最後只需比較兩個純量，不必建立完整 Jacobian。

### 向量鏈式法則

若

$$
x\overset{f}{\longmapsto}y
\overset{g}{\longmapsto}z,
$$

其中 $y=f(x)$、$z=g(y)$，則

$$
J_{g\circ f}(x)=J_g(y)J_f(x).
$$

若最終損失 $L$ 是純量，反向形式為

$$
\nabla_xL
=
J_f(x)^T\nabla_yL.
$$

若還有更下游的 $z$，則

$$
\nabla_xL
=
J_f(x)^TJ_g(y)^T\nabla_zL.
$$

矩陣乘法由右向左作用，正好對應由損失端往輸入端傳播。

### 矩陣乘法的反向傳播

令

$$
Z=XW,
$$

其中

$$
X\in\mathbb{R}^{B\times D_{\mathrm{in}}},
\quad
W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}},
\quad
Z\in\mathbb{R}^{B\times D_{\mathrm{out}}}.
$$

微分為

$$
dZ=dX\,W+X\,dW.
$$

設上游梯度為

$$
G=\nabla_ZL
\in\mathbb{R}^{B\times D_{\mathrm{out}}}.
$$

由 Frobenius 梯度定義，

$$
dL=\operatorname{tr}(G^TdZ).
$$

代入 $dZ$：

$$
dL
=
\operatorname{tr}(G^TdXW)
+
\operatorname{tr}(G^TXdW).
$$

利用 trace 的循環性，第一項可改寫為

$$
\operatorname{tr}(G^TdXW)
=
\operatorname{tr}(WG^TdX)
=
\operatorname{tr}((GW^T)^TdX).
$$

第二項為

$$
\operatorname{tr}(G^TXdW)
=
\operatorname{tr}((X^TG)^TdW).
$$

與梯度定義比較，得到

$$
\nabla_XL=GW^T,
\qquad
\nabla_WL=X^TG.
$$

shape 核對如下：

- $GW^T:(B,D_{\mathrm{out}})(D_{\mathrm{out}},D_{\mathrm{in}})
  \rightarrow(B,D_{\mathrm{in}})$；
- $X^TG:(D_{\mathrm{in}},B)(B,D_{\mathrm{out}})
  \rightarrow(D_{\mathrm{in}},D_{\mathrm{out}})$。

### 廣播偏置的梯度

若

$$
Z=XW+b,
\qquad
b\in\mathbb{R}^{D_{\mathrm{out}}},
$$

則 $b$ 被廣播到 $B$ 筆樣本。分量式為

$$
Z_{ij}=(XW)_{ij}+b_j.
$$

因此

$$
\frac{\partial L}{\partial b_j}
=
\sum_{i=1}^{B}
\frac{\partial L}{\partial Z_{ij}},
$$

即

$$
\nabla_bL
=
\sum_{i=1}^{B}G_{i,:}.
$$

反向傳播必須把廣播軸 sum 回原 shape。若 $Z$ 的 shape 是 $(B,T,D)$，而 $b$ 是 $(D,)$，則 $\nabla_bL$ 必須沿 batch 軸 $0$ 及時間軸 $1$ 求和。

### 分支與梯度累加

若

$$
q=x^2,\qquad r=3x,\qquad L=q+r,
$$

則 $x$ 同時流向兩條路徑。總微分為

$$
dL=dq+dr=(2x+3)dx,
$$

所以

$$
\frac{dL}{dx}=2x+3.
$$

平方分支傳回 $2x$，線性分支傳回 $3$，兩者必須相加。若程式後寫入的梯度覆蓋前一分支，就只會留下其中一項。

更一般地，若節點 $x$ 被 $k$ 個下游節點使用，則

$$
\nabla_xL
=
\sum_{r=1}^{k}
\left(\frac{\partial y^{(r)}}{\partial x}\right)^T
\nabla_{y^{(r)}}L.
$$

共享參數、殘差連接及重複索引都遵循這條規則。

### 反向拓撲順序

無循環計算圖的正向運算要求先取得輸入，再計算節點。反向傳播採反向拓撲順序：

1. 將純量損失的梯度設為 $1$。
2. 從損失端開始處理節點。
3. 每個節點利用局部導數計算對輸入的 VJP。
4. 把結果累加到各輸入的梯度槽。
5. 某節點的所有下游貢獻累加完成後，才繼續向更上游傳播。

這不只是效能技巧。若分支尚未全部處理便提早向上游傳播，所得梯度會缺少路徑貢獻。

---

## 逐步手算例題

### 例一：矩陣仿射層與平方損失

給定

$$
X=
\begin{bmatrix}
1&2
\end{bmatrix},
\quad
W=
\begin{bmatrix}
1&-1\\
2&0
\end{bmatrix},
\quad
b=
\begin{bmatrix}
1&2
\end{bmatrix}.
$$

此處 $B=1$、$D_{\mathrm{in}}=2$、$D_{\mathrm{out}}=2$。即使 $B=1$，仍保留 batch 軸，因此 $X$ 的 shape 是 $(1,2)$，而不是 NumPy 一維 shape $(2,)$。

先算仿射輸出：

$$
XW
=
\begin{bmatrix}1&2\end{bmatrix}
\begin{bmatrix}
1&-1\\
2&0
\end{bmatrix}
=
\begin{bmatrix}
5&-1
\end{bmatrix}.
$$

加上偏置：

$$
Z=XW+b
=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

本例明確採「單筆樣本內，沿輸出特徵軸取平方和」：

$$
L=\frac12\sum_{j=1}^{2}Z_{1j}^2.
$$

因為 $B=1$，本例沒有額外的 batch 平均因子。代入數值：

$$
L
=
\frac12(6^2+1^2)
=
18.5.
$$

因此

$$
G=\nabla_ZL=Z
=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

對輸入的梯度為

$$
\nabla_XL=GW^T.
$$

由

$$
W^T=
\begin{bmatrix}
1&2\\
-1&0
\end{bmatrix},
$$

得到

$$
\nabla_XL
=
\begin{bmatrix}6&1\end{bmatrix}
\begin{bmatrix}
1&2\\
-1&0
\end{bmatrix}
=
\begin{bmatrix}
5&12
\end{bmatrix}.
$$

對權重的梯度為

$$
\nabla_WL=X^TG
=
\begin{bmatrix}
1\\
2
\end{bmatrix}
\begin{bmatrix}
6&1
\end{bmatrix}
=
\begin{bmatrix}
6&1\\
12&2
\end{bmatrix}.
$$

偏置沿 batch 軸求和。此處只有一筆：

$$
\nabla_bL
=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

其 shape 分別為：

- $\nabla_XL:(1,2)$；
- $\nabla_WL:(2,2)$；
- $\nabla_bL:(2,)$ 或在純矩陣手算中保留為 $(1,2)$ row；程式採 NumPy 偏置約定 $(2,)$。

### 例二：共享中間值的分支反傳

考慮

$$
a=2x,\qquad
b=a^2,\qquad
c=3a,\qquad
L=b+c.
$$

取 $x=1.5$。正向計算：

$$
a=2(1.5)=3,
$$

$$
b=3^2=9,
$$

$$
c=3(3)=9,
$$

$$
L=9+9=18.
$$

反向從

$$
\frac{\partial L}{\partial L}=1
$$

開始。加法節點把梯度傳給兩個輸入：

$$
\frac{\partial L}{\partial b}=1,
\qquad
\frac{\partial L}{\partial c}=1.
$$

平方分支對 $a$ 的貢獻為

$$
\left.\frac{\partial L}{\partial a}\right|_b
=
\frac{\partial L}{\partial b}
\frac{\partial b}{\partial a}
=
1\cdot2a
=
6.
$$

線性分支對 $a$ 的貢獻為

$$
\left.\frac{\partial L}{\partial a}\right|_c
=
\frac{\partial L}{\partial c}
\frac{\partial c}{\partial a}
=
1\cdot3
=
3.
$$

累加兩條路徑：

$$
\frac{\partial L}{\partial a}=6+3=9.
$$

最後經過 $a=2x$：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial x}
=
9\cdot2
=
18.
$$

直接展開可獨立驗算：

$$
L=(2x)^2+3(2x)=4x^2+6x,
$$

所以

$$
\frac{dL}{dx}=8x+6.
$$

代入 $x=1.5$ 得 $18$，與計算圖反傳一致。

### 例三：batch平均只除一次

兩筆樣本的未平均損失為

$$
\ell_1=(w-1)^2,
\qquad
\ell_2=(2w-1)^2.
$$

batch mean 定義為

$$
L=\frac{\ell_1+\ell_2}{2}.
$$

取 $w=0$，第一筆的梯度為

$$
\frac{d\ell_1}{dw}
=
2(w-1)
=
-2.
$$

第二筆要使用鏈式法則：

$$
\frac{d\ell_2}{dw}
=
2(2w-1)\cdot2
=
-4.
$$

因此

$$
\frac{dL}{dw}
=
\frac{-2-4}{2}
=
-3.
$$

若每筆梯度先除以 $2$，最後又對 batch 取 mean，會錯得 $-1.5$；若完全不除，則得到 sum loss 的梯度 $-6$。採 sum 或 mean 都可以，但前向損失定義與反向縮放必須一致。

---

## 實作與程式

以下自足程式只依賴 NumPy，使用 CPU 建立二分類計算圖：

$$
Z=XW+b,\qquad p=\sigma(Z).
$$

對單一有限 logit $z$ 與標籤 $y\in\{0,1\}$，穩定二元交叉熵為

$$
\ell(z,y)
=
\max(z,0)-zy+\log(1+e^{-|z|}).
$$

batch loss 沿 batch 軸取 mean：

$$
L=\frac1B\sum_{i=1}^{B}\ell(z_i,y_i).
$$

因此

$$
\frac{\partial L}{\partial z_i}
=
\frac{\sigma(z_i)-y_i}{B}.
$$

程式執行解析反傳、逐參數中央有限差分、方向導數核對、故障測試及合成資料訓練。梯度檢查若失敗會拋出 `AssertionError`，使後續訓練不會執行。以下程式未在本章寫作流程中執行。

```python
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def bce_logits_mean(logits, y):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if logits.shape != y.shape:
        raise ValueError("logits與y必須具有相同shape")
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("本例要求logits與y的shape為(B, 1)")
    if logits.shape[0] == 0:
        raise ValueError("不可對空batch取mean")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits必須全部有限")
    if not np.all(np.isfinite(y)):
        raise ValueError("標籤必須全部有限")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("硬標籤必須為0或1")

    per_item = (
        np.maximum(logits, 0.0)
        - logits * y
        + np.log1p(np.exp(-np.abs(logits)))
    )
    loss = np.mean(per_item, axis=0).item()
    dlogits = (sigmoid(logits) - y) / logits.shape[0]
    return loss, dlogits


def forward_backward(X, y, W, b):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X必須具有shape (B, Din)")
    if X.shape[0] == 0:
        raise ValueError("X不可為空batch")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必須全部有限")
    if W.ndim != 2 or W.shape != (X.shape[1], 1):
        raise ValueError("W必須具有shape (Din, 1)")
    if b.shape != (1,):
        raise ValueError("b必須具有shape (1,)")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise ValueError("參數必須全部有限")

    logits = X @ W + b
    loss, dlogits = bce_logits_mean(logits, y)

    dX = dlogits @ W.T
    dW = X.T @ dlogits
    db = np.sum(dlogits, axis=0)

    if dX.shape != X.shape:
        raise AssertionError("dX shape錯誤")
    if dW.shape != W.shape:
        raise AssertionError("dW shape錯誤")
    if db.shape != b.shape:
        raise AssertionError("db shape錯誤")

    return loss, logits, dX, dW, db


def parameter_loss(theta, X, y):
    theta = np.asarray(theta, dtype=np.float64)
    din = X.shape[1]
    if theta.shape != (din + 1,):
        raise ValueError("theta必須包含Din個權重及1個偏置")
    W = theta[:din].reshape(din, 1)
    b = theta[din:].reshape(1)
    return forward_backward(X, y, W, b)[0]


def finite_difference_gradient(theta, X, y, eps=1e-6):
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps必須是有限正數")

    grad = np.zeros_like(theta, dtype=np.float64)
    for i in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (
            parameter_loss(plus, X, y)
            - parameter_loss(minus, X, y)
        ) / (2.0 * eps)
    return grad


def gradient_checks(atol=1e-8, rtol=1e-6):
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("容差不可為負數")
    if not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("容差必須有限")

    X = np.array([[1.0, -2.0],
                  [0.5,  3.0]], dtype=np.float64)
    y = np.array([[1.0],
                  [0.0]], dtype=np.float64)
    W = np.array([[0.2],
                  [-0.4]], dtype=np.float64)
    b = np.array([0.1], dtype=np.float64)

    loss, _, dX, dW, db = forward_backward(X, y, W, b)
    theta = np.concatenate([W.ravel(), b])
    analytic = np.concatenate([dW.ravel(), db])
    numeric = finite_difference_gradient(theta, X, y)

    parameter_error = np.max(np.abs(analytic - numeric))
    parameter_ok = np.allclose(
        analytic, numeric, atol=atol, rtol=rtol
    )

    direction = np.array([0.3, -0.8, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    fd_direction = (
        parameter_loss(theta + eps * direction, X, y)
        - parameter_loss(theta - eps * direction, X, y)
    ) / (2.0 * eps)
    vjp_direction = float(analytic @ direction)
    direction_error = abs(fd_direction - vjp_direction)
    direction_ok = np.isclose(
        fd_direction, vjp_direction, atol=atol, rtol=rtol
    )

    print("gradient-check loss:", loss)
    print("parameter max abs error:", parameter_error)
    print("directional abs error:", direction_error)

    if not parameter_ok or not direction_ok:
        raise AssertionError(
            "梯度檢查失敗；"
            f"parameter_error={parameter_error}, "
            f"direction_error={direction_error}, "
            f"X={X.shape}, dX={dX.shape}, "
            f"W={W.shape}, dW={dW.shape}, "
            f"b={b.shape}, db={db.shape}"
        )

    print("gradient checks: accepted under configured tolerances")


def make_split(seed=7, n_train=80, n_val=20, n_test=20):
    sizes = (n_train, n_val, n_test)
    if not all(isinstance(n, (int, np.integer)) for n in sizes):
        raise ValueError("切分大小必須是整數")
    if min(sizes) <= 0:
        raise ValueError("每個切分都必須至少有一筆")

    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test

    X = rng.normal(size=(n_total, 2))
    true_logits = X[:, [0]] - 0.7 * X[:, [1]] + 0.2
    probability = sigmoid(true_logits)
    y = (
        rng.random((n_total, 1)) < probability
    ).astype(np.float64)

    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = (X[train_idx], y[train_idx])
    val = (X[val_idx], y[val_idx])
    test = (X[test_idx], y[test_idx])
    return train, val, test


def accuracy(X, y, W, b):
    prediction = ((X @ W + b) >= 0.0).astype(np.float64)
    return np.mean(prediction == y).item()


def majority_baseline(train_y, eval_y):
    majority_label = float(np.mean(train_y) >= 0.5)
    return np.mean(eval_y == majority_label).item()


def expect_value_error(name, function):
    try:
        function()
    except ValueError as error:
        print(f"{name}: rejected as expected ({error})")
        return
    raise AssertionError(
        f"{name}: expected ValueError, but input was accepted"
    )


def failure_tests():
    expect_value_error(
        "invalid label",
        lambda: bce_logits_mean(
            np.array([[0.0]]),
            np.array([[0.3]])
        )
    )
    expect_value_error(
        "empty batch",
        lambda: bce_logits_mean(
            np.empty((0, 1)),
            np.empty((0, 1))
        )
    )
    expect_value_error(
        "non-finite logit",
        lambda: bce_logits_mean(
            np.array([[np.inf]]),
            np.array([[1.0]])
        )
    )
    expect_value_error(
        "wrong label shape",
        lambda: bce_logits_mean(
            np.zeros((2, 1)),
            np.zeros((2,))
        )
    )


def boundary_tests():
    X = np.array([[2.0, -1.0]], dtype=np.float64)
    y = np.array([[1.0]], dtype=np.float64)
    W = np.array([[0.1], [0.2]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    _, logits, dX, dW, db = forward_backward(X, y, W, b)

    if logits.shape != (1, 1):
        raise AssertionError("B=1時遺失batch軸")
    if dX.shape != X.shape or dW.shape != W.shape:
        raise AssertionError("B=1梯度shape錯誤")
    if db.shape != b.shape:
        raise AssertionError("B=1偏置梯度shape錯誤")

    extreme = np.array([[1000.0], [-1000.0]])
    labels = np.array([[1.0], [0.0]])
    extreme_loss, extreme_grad = bce_logits_mean(extreme, labels)
    if not np.isfinite(extreme_loss):
        raise AssertionError("有限極端logits產生非有限loss")
    if not np.all(np.isfinite(extreme_grad)):
        raise AssertionError("有限極端logits產生非有限梯度")

    print("boundary tests: accepted")


def train_demo():
    train, val, test = make_split()
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    W = np.zeros((2, 1), dtype=np.float64)
    b = np.zeros((1,), dtype=np.float64)
    learning_rate = 0.2

    for _ in range(200):
        _, _, _, dW, db = forward_backward(
            X_train, y_train, W, b
        )
        W -= learning_rate * dW
        b -= learning_rate * db

    train_loss = forward_backward(
        X_train, y_train, W, b
    )[0]
    val_loss = forward_backward(X_val, y_val, W, b)[0]
    test_loss = forward_backward(X_test, y_test, W, b)[0]

    print("train/val/test loss:",
          train_loss, val_loss, test_loss)
    print("train/val/test accuracy:",
          accuracy(X_train, y_train, W, b),
          accuracy(X_val, y_val, W, b),
          accuracy(X_test, y_test, W, b))
    print("majority baseline val/test:",
          majority_baseline(y_train, y_val),
          majority_baseline(y_train, y_test))


if __name__ == "__main__":
    gradient_checks()
    boundary_tests()
    failure_tests()
    train_demo()
```

主程式先執行梯度、邊界及故障測試。任一檢查不符都會拋出例外，使 `train_demo()` 不會執行。這符合「梯度驗收失敗便停止訓練」的契約。

有限差分只適合小問題。若參數量為 $P$，逐元素中央差分約需 $2P$ 次前向計算；反向傳播則通常只需一次正向及一次反向。大型模型宜採方向導數或抽查參數，不應建立完整大型 Jacobian。

---

## 測試與預期結果

由於本章沒有執行程式，以下均為根據公式與程式控制流程所作的**預期**，不是實測結果。

### 正常測試

`gradient_checks()` 使用有限且遠離非光滑點的輸入，預期：

- `analytic` 與 `numeric` 在指定 `atol=1e-8`、`rtol=1e-6` 下接近。
- 方向有限差分與 `analytic @ direction` 在相同容差下接近。
- `parameter_error` 與 `direction_error` 都是有限非負數。
- `dW.shape == (2, 1)`、`db.shape == (1,)`、`dX.shape == X.shape`。
- 若任一比較不符，應拋出包含誤差及 shape 的 `AssertionError`，而不是繼續訓練。

容差是本例 float64 小問題的建議驗收值，不是對所有函數、步長或平台的普遍保證。

### 邊界測試

1. **$B=1$**：輸入保持 `(1, Din)`，輸出保持 `(1, 1)`，不可壓成一維。
2. **極端但有限 logits**：`1000.0` 與 `-1000.0` 應由穩定 BCE 公式產生有限損失及有限梯度。
3. **單類標籤**：數學上可以計算，但多數類別基線可能很強，不能只報模型準確率。
4. **接近飽和的 sigmoid**：梯度可能非常接近零，這是函數性質，不必然是程式故障。

### 故障測試

`failure_tests()` 要求下列輸入確實被拒絕：

- 標籤為 `0.3`，不符合本例硬 Bernoulli 標籤域 $\{0,1\}$。
- 空 batch，因為 mean 沒有合法分母。
- `inf` logit，避免非有限值靜默進入損失及梯度。
- logits 為 `(B,1)`、標籤為 `(B,)`，避免錯誤廣播成 `(B,B)`。

若被測函式意外接受非法輸入，`expect_value_error` 會主動拋出 `AssertionError`。因此故障測試不會因「沒有發生任何事」而被誤認為成功。

### 有限差分失敗時的診斷順序

若梯度核對失敗，應依序檢查：

1. 前向損失究竟是 sum 還是 mean。
2. batch 平均是否只除一次。
3. 廣播軸是否在反向時求和。
4. 解析梯度是否與參數同 shape。
5. 是否把 `X.T @ dY` 錯寫成 `dY.T @ X`。
6. 有限差分步長是否過大或過小。
7. 是否在 ReLU 零點等不可微位置檢查。
8. 是否存在 `nan`、`inf` 或錯誤 dtype。

### 訓練結果的解讀限制

`train_demo()` 預期更新訓練參數，但特定 seed 下的驗證或測試準確率不保證單調改善。一次合成切分不能支持統計顯著性主張。程式列印測試指標只是完整展示資料角色；若根據測試結果反覆調整學習率或步數，該集合便不再是未見測試集。

---

## 反例與常見陷阱

### 陷阱一：NumPy一維陣列的轉置

```python
x = np.array([1.0, 2.0])
print(x.shape)    # (2,)
print(x.T.shape)  # 仍為(2,)
```

一維陣列沒有明確 row／column 軸，`.T` 不會改變 shape。若需要單筆 batch，應使用 `x.reshape(1, 2)`；若需要 column 向量，應使用 `x.reshape(2, 1)`。`reshape` 改變軸的組織，transpose 交換既有軸，兩者不是同一操作。

### 陷阱二：錯誤廣播可執行但語意錯誤

若 logits 是 $(B,1)$，標籤錯寫成 $(B,)$，NumPy 相減可能廣播成 $(B,B)$。程式未必報錯，但每筆樣本會錯誤地與所有標籤配對。因此本章要求 `logits.shape == y.shape`，不能只檢查元素數量。

### 陷阱三：分支梯度覆寫

錯誤寫法：

```python
da = db * (2.0 * a)
da = dc * 3.0
```

第二行覆蓋第一條路徑。正確寫法是：

```python
da = db * (2.0 * a)
da += dc * 3.0
```

共享權重、殘差連接及同一 embedding 索引重複出現時，都有相同的累加要求。

### 陷阱四：有限差分取代證明

有限差分只提供數值證據，不能證明公式對所有輸入成立。中央差分

$$
\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
$$

的截斷誤差通常隨 $\epsilon^2$ 下降，但 $\epsilon$ 太小時會受浮點消去誤差影響。合理作法是比較多個步長，例如 $10^{-4}$、$10^{-5}$、$10^{-6}$，觀察誤差是否先下降再受浮點限制，而不是只測一個步長。

### 陷阱五：在非光滑點硬套中央差分

對 ReLU，

$$
\operatorname{ReLU}(x)=\max(0,x),
$$

$x=0$ 處沒有唯一導數。實作可以約定零點梯度為 $0$，但中央差分在零點給出約 $1/2$。兩者不同不必然表示反向程式錯誤，而是局部導數約定與對稱差分所測量的量不同。

### 陷阱六：建立完整大型Jacobian

若一層輸入及輸出各有一百萬個元素，完整 Jacobian 有 $10^{12}$ 個元素，通常不可行。反向傳播的價值正是利用計算圖結構直接計算所需 VJP。只有極小的教學問題適合顯式建立 Jacobian。

### 陷阱七：梯度通過便宣稱模型正確

梯度檢查只能支持「解析反傳與該數值近似一致」。它不能證明：

- 標籤規則合理；
- 資料沒有洩漏；
- 測試集未被用於調參；
- 模型會在分布外泛化；
- 應用決策安全。

這些問題需要不同的證據與測試。

---

## AI、幾何與養殖案例

### AI：共享參數就是多路梯度累加

神經網路中的同一權重矩陣可能在多個 token、時間步或分支重複使用。若

$$
L=L_1(W)+L_2(W),
$$

則

$$
\nabla_WL
=
\nabla_WL_1+\nabla_WL_2.
$$

梯度不是從多條路徑中選出「最重要的一條」，而是加總所有可微依賴路徑。注意力投影、循環計算、殘差網路及權重共享都依賴此原則。

### 幾何：梯度與局部最陡方向

純量函數 $f(x)$ 的一階近似為

$$
f(x+\epsilon u)
\approx
f(x)+\epsilon\nabla f(x)^Tu.
$$

若限制 $\|u\|_2=1$，由 Cauchy–Schwarz 不等式，

$$
\nabla f(x)^Tu
\leq
\|\nabla f(x)\|_2\|u\|_2
=
\|\nabla f(x)\|_2.
$$

當

$$
u=\frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

時取等號。因此梯度是歐氏距離下的局部最陡上升方向，負梯度則是局部最陡下降方向。這只是局部一階敘述，不保證固定的大步長一定降低損失，也不保證到達全域最小值。

### 合成養殖案例：離線人工複核分類

假設完全合成的兩個特徵表示感測摘要：

- $x_1$：以訓練集統計標準化的合成溶氧指標；
- $x_2$：以訓練集統計標準化的合成水溫指標。

模型輸出二分類 logit：

$$
z=x_1w_1+x_2w_2+b.
$$

若某筆樣本的損失對 logit 梯度為 $\delta$，則

$$
\nabla_wL
=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\delta,
\qquad
\nabla_xL
=
\begin{bmatrix}
w_1&w_2
\end{bmatrix}
\delta.
$$

若同一權重用於多個時間點，總權重梯度是所有有效時間點貢獻之和，再依損失定義進行平均。

資料必須先按養殖槽、來源群組或時間區間切分，再只以訓練資料擬合標準化統計。不能讓同一槽的高度相似時間片分散到訓練與測試集合。此案例沒有真實操作閾值，模型輸出也不是設備控制授權，不得據此控制曝氣、投餌、泵浦、加藥或其他設備。即使合成測試集表現良好，也不能推論真實養殖安全性。

---

## 習題

### 一、手算題

給定

$$
X=
\begin{bmatrix}
1&0\\
2&-1
\end{bmatrix},
\quad
W=
\begin{bmatrix}
2\\
3
\end{bmatrix},
\quad
b=
\begin{bmatrix}
-1
\end{bmatrix}.
$$

令 $Z=XW+b$，損失為

$$
L=\frac{1}{2B}\sum_{i=1}^{B}Z_i^2.
$$

求 $Z$、$L$、$\nabla_ZL$、$\nabla_XL$、$\nabla_WL$ 與 $\nabla_bL$，並列出各量的 shape。

### 二、程式題

修改本章程式，加入輸入 $X$ 的方向導數檢查。給定與 $X$ 同 shape 的方向 $U$，比較

$$
\frac{L(X+\epsilon U)-L(X-\epsilon U)}{2\epsilon}
$$

與

$$
\langle\nabla_XL,U\rangle_F.
$$

若比較超出明列容差，必須拋出 `AssertionError`。不得建立對 $X$ 的完整 Jacobian。

### 三、反例題

某人聲稱：「若 $z=x+x$，反向傳播只需把下游梯度傳回 $x$ 一次，因為兩個 $x$ 是同一變數。」請給出數值反例，說明此作法錯在哪裡。

### 四、整合題

設合成序列特徵 $X$ 的 shape 為 $(B,T,D)$，共享權重 $W$ 的 shape 為 $(D,K)$，偏置 $b$ 的 shape 為 $(K,)$：

$$
Z_{b,t,:}=X_{b,t,:}W+b.
$$

損失定義為所有 $BT$ 個位置之輸出元素平方和的一半，再沿位置取 mean：

$$
L=\frac{1}{2BT}\sum_{b,t,k}Z_{btk}^2.
$$

推導 $\nabla_XL$、$\nabla_WL$、$\nabla_bL$，明確指出 reduction 軸與平均因子。若部分時間位置無效，以布林 mask $M\in\{0,1\}^{B\times T}$ 表示，應如何改成有效位置平均？空有效集合如何處理？

---

## 習題解答

### 一、手算題解答

此處 $B=2$。先計算

$$
XW
=
\begin{bmatrix}
1&0\\
2&-1
\end{bmatrix}
\begin{bmatrix}
2\\
3
\end{bmatrix}
=
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

偏置沿 batch 軸廣播：

$$
Z=
\begin{bmatrix}
2\\
1
\end{bmatrix}
+
\begin{bmatrix}
-1\\
-1
\end{bmatrix}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

$Z$ 的 shape 為 $(2,1)$。損失為

$$
L
=
\frac{1}{2\cdot2}(1^2+0^2)
=
\frac14.
$$

由

$$
L=\frac{1}{2B}\sum_iZ_i^2
$$

得到

$$
\nabla_ZL
=
\frac{Z}{B}
=
\begin{bmatrix}
1/2\\
0
\end{bmatrix},
$$

shape 為 $(2,1)$。

輸入梯度為

$$
\nabla_XL
=
\nabla_ZL\,W^T
=
\begin{bmatrix}
1/2\\
0
\end{bmatrix}
\begin{bmatrix}
2&3
\end{bmatrix}
=
\begin{bmatrix}
1&3/2\\
0&0
\end{bmatrix}.
$$

其 shape 為 $(2,2)$。

權重梯度為

$$
\nabla_WL
=
X^T\nabla_ZL.
$$

因為

$$
X^T=
\begin{bmatrix}
1&2\\
0&-1
\end{bmatrix},
$$

所以

$$
\nabla_WL
=
\begin{bmatrix}
1&2\\
0&-1
\end{bmatrix}
\begin{bmatrix}
1/2\\
0
\end{bmatrix}
=
\begin{bmatrix}
1/2\\
0
\end{bmatrix}.
$$

其 shape 為 $(2,1)$。

偏置梯度沿 batch 軸 $0$ 求和：

$$
\nabla_bL
=
\sum_{i=1}^{2}(\nabla_ZL)_i
=
\frac12.
$$

在 NumPy 約定下，其 shape 為 $(1,)$。

### 二、程式題解答

可加入以下函式：

```python
def input_direction_check(
    X, y, W, b, U, eps=1e-6, atol=1e-8, rtol=1e-6
):
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    if U.shape != X.shape:
        raise ValueError("U必須與X具有相同shape")
    if not np.all(np.isfinite(U)):
        raise ValueError("U必須全部有限")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps必須是有限正數")
    if min(atol, rtol) < 0.0:
        raise ValueError("容差不可為負數")

    _, _, dX, _, _ = forward_backward(X, y, W, b)

    loss_plus = forward_backward(
        X + eps * U, y, W, b
    )[0]
    loss_minus = forward_backward(
        X - eps * U, y, W, b
    )[0]

    finite_difference = (
        loss_plus - loss_minus
    ) / (2.0 * eps)
    vjp_inner_product = float(np.sum(dX * U))

    error = abs(finite_difference - vjp_inner_product)
    if not np.isclose(
        finite_difference,
        vjp_inner_product,
        atol=atol,
        rtol=rtol
    ):
        raise AssertionError(
            "輸入方向導數檢查失敗；"
            f"error={error}, X={X.shape}, U={U.shape}"
        )

    return finite_difference, vjp_inner_product, error
```

其中

```python
np.sum(dX * U)
```

是 Frobenius 內積

$$
\langle\nabla_XL,U\rangle_F.
$$

它把兩個 $(B,D_{\mathrm{in}})$ 矩陣逐元素相乘，再沿所有軸取 sum，得到純量。

### 三、反例題解答

令

$$
z=x+x,
\qquad
L=z^2,
$$

並取 $x=3$。正向計算：

$$
z=6,
\qquad
L=36.
$$

先有

$$
\frac{dL}{dz}=2z=12.
$$

加法節點的兩個輸入都依賴同一個 $x$。第一條路徑貢獻 $12$，第二條路徑也貢獻 $12$，因此

$$
\frac{dL}{dx}=12+12=24.
$$

直接展開驗算：

$$
L=(2x)^2=4x^2,
$$

所以

$$
\frac{dL}{dx}=8x.
$$

代入 $x=3$ 得 $24$。若只傳回一次，會錯得 $12$。同一變數出現兩次不會消除其中一條依賴路徑，反而要求把兩條梯度貢獻相加。

### 四、整合題解答

損失為

$$
L=\frac{1}{2BT}\sum_{b,t,k}Z_{btk}^2,
$$

因此先定義

$$
G=\nabla_ZL=\frac{Z}{BT}.
$$

$G$ 的 shape 為 $(B,T,K)$，平均因子只在此引入一次。

對輸入而言，矩陣乘法作用於最後一軸：

$$
\nabla_XL=GW^T.
$$

其 shape 為

$$
(B,T,K)(K,D)\rightarrow(B,T,D).
$$

權重在所有 batch 與時間位置共享，因此要沿 $B$、$T$ 軸累加：

$$
\nabla_WL
=
\sum_{b=1}^{B}\sum_{t=1}^{T}
X_{bt,:}^TG_{bt,:}.
$$

shape 為 $(D,K)$。NumPy 可寫成

```python
dW = np.einsum("btd,btk->dk", X, G)
```

偏置沿廣播軸 $0$ 與 $1$ 求和：

$$
\nabla_bL
=
\sum_{b=1}^{B}\sum_{t=1}^{T}G_{bt,:}.
$$

shape 為 $(K,)$，NumPy 可寫成

```python
db = np.sum(G, axis=(0, 1))
```

若有 mask $M\in\{0,1\}^{B\times T}$，有效位置數為

$$
N_{\mathrm{eff}}
=
\sum_{b,t}M_{bt}.
$$

有效位置平均損失應定義為

$$
L
=
\frac{1}{2N_{\mathrm{eff}}}
\sum_{b,t,k}M_{bt}Z_{btk}^2.
$$

因此

$$
G_{btk}
=
\frac{M_{bt}Z_{btk}}{N_{\mathrm{eff}}}.
$$

mask 可擴充為 `M[:, :, None]`，shape 從 $(B,T)$ 變成 $(B,T,1)$，再沿 $K$ 軸廣播。後續 $\nabla_XL$、$\nabla_WL$、$\nabla_bL$ 公式不變。

除數是有效位置數 $N_{\mathrm{eff}}$，不是 $BTK$：每個位置先沿 $K$ 軸取平方和，再對有效位置取 mean。若 $N_{\mathrm{eff}}=0$，必須明確拒絕，不能除以零，也不能靜默把損失設成零。

---

## 本章小結

反向傳播不是獨立於微積分的神祕演算法，而是鏈式法則在計算圖上的有效組織。對純量損失而言，我們通常需要的是 VJP，而不是完整 Jacobian。

本章的核心規則如下：

- 以 $df=\operatorname{tr}(G^TdX)$ 定義矩陣梯度。
- 梯度必須與被微分變數具有相同 shape。
- JVP 描述輸入方向如何推動輸出；VJP 把輸出端梯度拉回輸入端。
- 對 $Z=XW+b$，有
  $\nabla_XL=GW^T$、$\nabla_WL=X^TG$，而 $\nabla_bL$ 沿廣播軸求和。
- 計算圖依反向拓撲順序處理。
- 一個值若有多條下游路徑，梯度必須累加。
- mean loss 的平均因子只施加一次，且 reduction 軸必須明列。
- 有限差分與內積對偶是驗證工具，不是數學證明。
- 驗收程式必須真正判定成功或失敗，不能只列印數字。
- 梯度正確不等於資料無洩漏、模型能泛化或應用決策安全。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, <https://arxiv.org/abs/1706.03762>。本章只將其視為後續 Transformer 計算圖與共享參數的背景入口；依提供的來源註記，僅摘要頁已取得，不能宣稱已完整核對論文。
2. NumPy, *Broadcasting*, <https://numpy.org/doc/stable/user/basics.broadcasting.html>。依來源註記，此入口尚待逐條核對；本章的廣播梯度由 shape、分量式及微分直接推導。
3. Dive into Deep Learning, <https://d2l.ai/>。僅列為延伸閱讀入口；依來源註記，未逐章核對。
4. PyTorch, *Reproducibility*, <https://docs.pytorch.org/docs/stable/notes/randomness.html>。僅列為後續框架可重現性議題的延伸入口；本章 NumPy 程式未據此宣稱跨平台逐位重現。