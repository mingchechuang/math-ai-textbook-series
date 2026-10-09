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

其中 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$，而 $b\in\mathbb{R}^{D_{\mathrm{out}}}$ 沿 batch 軸廣播。這與微分教科書常把單一向量寫成 column 向量並不衝突；關鍵是每一式的 shape 必須一致。

---

## 問題與直覺

設一個模型依序進行

$$
X\longrightarrow Z=XW+b\longrightarrow A=\tanh Z\longrightarrow L.
$$

若參數 $W$ 有百萬個元素，輸出 $A$ 也有大量元素，直接建立「每個輸出對每個參數的偏導數」會產生巨大 Jacobian。訓練其實通常不需要完整 Jacobian，因為最終目標是純量損失 $L$。我們只需要知道：

> 損失的一個微小變化，如何沿計算圖反向分配到每個參數？

反向傳播正是鏈式法則的有效實作。每個節點接收來自下游的梯度，計算對其輸入的 VJP，再把結果傳向上游。若同一值流向多個分支，各分支都會對它的梯度作出貢獻，因此必須相加，而不是覆寫。

### 模型、目標與證據契約

為避免把「程式跑得動」誤當成「模型具備能力」，本章採以下實驗契約：

- **資料生成**：只使用程式產生的合成二分類資料，不下載語料或模型。
- **生成規則**：特徵 $x\in\mathbb{R}^2$ 由標準常態抽樣；條件標籤由
  $P(y=1\mid x)=\sigma(x_1-0.7x_2+0.2)$ 定義。
- **機率域**：$\sigma(s)=1/(1+e^{-s})\in(0,1)$；抽樣標籤屬於 $\{0,1\}$。樣本中的類別比例不等於真實條件分布。
- **固定種子範圍**：種子只固定該次 NumPy 隨機數流程，不代表跨 NumPy 版本、平台或浮點實作逐位相同。
- **資料切分**：先生成互不重疊的訓練、驗證、測試索引，再以訓練集決定任何模型選擇。測試集不參與學習率、步數或架構調整。
- **損失平均**：二元交叉熵對 batch 軸取 mean，反向時只除以 $B$ 一次。
- **基線**：以訓練集多數類別作固定預測，再分別評估驗證集與測試集。
- **失敗報告**：若梯度檢查不符，應先停止訓練並報告最大誤差、方向及 shape；不能只展示下降的訓練損失。
- **泛化限制**：低訓練損失只表示模型適合訓練資料，不保證驗證、測試或分布外資料表現。

---

## 定義、定理與推導

### 微分、梯度與 Frobenius 內積

對純量函數 $f(X)$，其中 $X\in\mathbb{R}^{m\times n}$，梯度定義為與 $X$ 同 shape 的矩陣 $\nabla_X f$，使一階微分可寫成

$$
df=\operatorname{tr}\left((\nabla_X f)^T\,dX\right).
$$

矩陣的 Frobenius 內積定義為

$$
\langle A,B\rangle_F
=\operatorname{tr}(A^TB)
=\sum_{i,j}A_{ij}B_{ij}.
$$

因此也可寫成

$$
df=\langle \nabla_X f,dX\rangle_F.
$$

「梯度與參數同 shape」是實作時的重要檢查條件。例如 $W$ 是 $(D_{\mathrm{in}},D_{\mathrm{out}})$，則 $dW$ 也必須是相同 shape。

### Jacobian、JVP與VJP

考慮向量函數

$$
y=f(x),\qquad
x\in\mathbb{R}^n,\quad y\in\mathbb{R}^m.
$$

其 Jacobian 為

$$
J_f(x)_{ij}=\frac{\partial y_i}{\partial x_j},
\qquad J_f(x)\in\mathbb{R}^{m\times n}.
$$

給定輸入方向 $u\in\mathbb{R}^n$，JVP 是

$$
J_f(x)u\in\mathbb{R}^m,
$$

描述輸入沿 $u$ 微小移動時，輸出的方向導數。

給定輸出端向量 $v\in\mathbb{R}^m$，VJP 是

$$
v^TJ_f(x)\in\mathbb{R}^{1\times n}.
$$

若將梯度統一表示為 column 向量，則反向傳播通常寫成

$$
\nabla_x L=J_f(x)^T\nabla_y L.
$$

這就是 VJP 的轉置表示。實作只需知道如何把上游梯度 $\nabla_yL$ 映回 $\nabla_xL$，無須實際配置完整的 $m\times n$ Jacobian。

### 小命題：VJP的內積對偶

**命題。** 若 $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ 在 $x$ 可微，則對任意 $u\in\mathbb{R}^n$ 與 $v\in\mathbb{R}^m$，

$$
\langle v,J_f(x)u\rangle
=
\langle J_f(x)^Tv,u\rangle.
$$

**證明。**

由歐氏內積定義，

$$
\langle v,J_f(x)u\rangle=v^TJ_f(x)u.
$$

矩陣乘積的結果是純量，因此等於其轉置：

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

另一方面，

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

這個命題提供很實用的梯度檢查：左側可用前向有限差分近似 JVP，右側用反向傳播得到 VJP，再比較兩個純量。它只需要方向向量，不需要建立 Jacobian。

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

設上游梯度為 $G=\nabla_ZL$。則

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

利用 trace 的循環性：

$$
\operatorname{tr}(G^TdXW)
=
\operatorname{tr}((GW^T)^TdX),
$$

以及

$$
\operatorname{tr}(G^TXdW)
=
\operatorname{tr}((X^TG)^TdW).
$$

因此

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
\qquad b\in\mathbb{R}^{D_{\mathrm{out}}},
$$

則 $b$ 被複製到 $B$ 筆樣本。微分分量式為

$$
dZ_{ij}=\cdots+db_j.
$$

所以

$$
\frac{\partial L}{\partial b_j}
=
\sum_{i=1}^{B}\frac{\partial L}{\partial Z_{ij}}.
$$

即

$$
\nabla_bL=\sum_{i=1}^{B}G_{i,:}.
$$

這是「廣播的反向必須 sum 回原 shape」的具體例子。若前導軸還包含時間 $T$，例如 $Z$ 的 shape 是 $(B,T,D)$ 而 $b$ 是 $(D,)$，則 $db$ 必須沿 $B$ 與 $T$ 軸求和。

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

反向傳播時，平方分支傳回 $2x$，線性分支傳回 $3$，兩者必須相加。若程式後寫入的梯度覆蓋前一分支，就會錯誤地只留下其中一項。

### 反向拓撲順序

一個無循環計算圖的正向順序，要求節點在其輸入可用後才計算。反向傳播則採反向拓撲順序：

1. 將最終純量損失的梯度設為 $1$。
2. 從損失端開始處理節點。
3. 每個節點用局部導數計算對輸入的 VJP。
4. 將結果累加到各輸入的梯度槽。
5. 只有在所有下游貢獻都到齊後，才繼續向更上游傳播。

這不只是效能技巧。若分支尚未全部處理就提早傳播，所得梯度會缺少路徑貢獻。

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

此處 $B=1$、$D_{\mathrm{in}}=2$、$D_{\mathrm{out}}=2$。即使 $B=1$，仍保留 batch 軸，所以 $X$ 是 $(1,2)$，不是 NumPy 的一維 $(2,)$。

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
Z=XW+b=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

令損失為元素平方和的一半：

$$
L=\frac12\sum_{j=1}^{2}Z_j^2
=\frac12(6^2+1^2)
=18.5.
$$

因此

$$
G=\nabla_ZL=Z=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

對輸入的梯度：

$$
\nabla_XL=GW^T.
$$

因為

$$
W^T=
\begin{bmatrix}
1&2\\
-1&0
\end{bmatrix},
$$

所以

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

對權重的梯度：

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
\nabla_bL=
\begin{bmatrix}
6&1
\end{bmatrix}.
$$

所有梯度都與對應變數同 shape。

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

反向從 $\partial L/\partial L=1$ 開始。

加法節點把梯度分給兩個輸入：

$$
\frac{\partial L}{\partial b}=1,
\qquad
\frac{\partial L}{\partial c}=1.
$$

平方分支對 $a$ 的貢獻：

$$
\left.\frac{\partial L}{\partial a}\right|_{b}
=
\frac{\partial L}{\partial b}\frac{\partial b}{\partial a}
=
1\cdot2a
=6.
$$

線性分支對 $a$ 的貢獻：

$$
\left.\frac{\partial L}{\partial a}\right|_{c}
=
\frac{\partial L}{\partial c}\frac{\partial c}{\partial a}
=
1\cdot3
=3.
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
=18.
$$

直接展開也可驗算：

$$
L=(2x)^2+3(2x)=4x^2+6x,
$$

$$
\frac{dL}{dx}=8x+6.
$$

代入 $x=1.5$ 得 $18$，與計算圖反傳一致。

### 例三：損失平均只除一次

兩筆樣本的未平均損失為

$$
\ell_1=(w-1)^2,\qquad
\ell_2=(2w-1)^2.
$$

batch mean 為

$$
L=\frac{\ell_1+\ell_2}{2}.
$$

取 $w=0$：

$$
\frac{d\ell_1}{dw}=2(w-1)=-2,
$$

$$
\frac{d\ell_2}{dw}=2(2w-1)\cdot2=-4.
$$

因此

$$
\frac{dL}{dw}=\frac{-2-4}{2}=-3.
$$

若每筆損失的梯度先除以 $2$，最後又對 batch 取 mean，便會錯得 $-1.5$。反之，完全不除則得到 sum loss 的梯度 $-6$。損失採 sum 或 mean 都可以，但前向定義與反向縮放必須一致。

---

## 實作與程式

以下程式只依賴 NumPy，在 CPU 上建立一個小型二分類計算圖：

$$
Z=XW+b,\qquad
p=\sigma(Z),
$$

並以直接從 logits 計算的穩定二元交叉熵為目標。對單一 logit $z$ 與標籤 $y\in\{0,1\}$，

$$
\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|}).
$$

batch loss 是沿 batch 軸的 mean。其梯度為

$$
\frac{\partial L}{\partial z_i}
=
\frac{\sigma(z_i)-y_i}{B}.
$$

程式同時執行解析反傳、參數中央有限差分、方向導數及訓練／驗證／測試切分。以下僅提供可執行稿，未在本章寫作流程中執行；輸出均屬預期。

```python
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def bce_logits_mean(logits, y):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if logits.shape != y.shape:
        raise ValueError("logits與y必須同shape")
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("本例要求shape為(B, 1)")
    if logits.shape[0] == 0:
        raise ValueError("不可對空batch取mean")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits必須全部有限")
    if not np.all(np.isfinite(y)):
        raise ValueError("標籤必須全部有限")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("標籤必須為0或1")

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
        raise ValueError("X必須為(B, Din)")
    if W.ndim != 2 or W.shape != (X.shape[1], 1):
        raise ValueError("W必須為(Din, 1)")
    if b.shape != (1,):
        raise ValueError("b必須為(1,)")

    logits = X @ W + b
    loss, dlogits = bce_logits_mean(logits, y)

    dX = dlogits @ W.T
    dW = X.T @ dlogits
    db = np.sum(dlogits, axis=0)
    return loss, logits, dX, dW, db


def parameter_loss(theta, X, y):
    din = X.shape[1]
    W = theta[:din].reshape(din, 1)
    b = theta[din:].reshape(1)
    return forward_backward(X, y, W, b)[0]


def finite_difference_gradient(theta, X, y, eps=1e-6):
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


def make_split(seed=7, n_train=80, n_val=20, n_test=20):
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("每個切分都必須至少有一筆")
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    X = rng.normal(size=(n, 2))
    true_logits = X[:, [0]] - 0.7 * X[:, [1]] + 0.2
    prob = sigmoid(true_logits)
    y = (rng.random((n, 1)) < prob).astype(np.float64)

    train = (X[:n_train], y[:n_train])
    val = (X[n_train:n_train+n_val],
           y[n_train:n_train+n_val])
    test = (X[n_train+n_val:], y[n_train+n_val:])
    return train, val, test


def accuracy(X, y, W, b):
    pred = ((X @ W + b) >= 0.0).astype(np.float64)
    return np.mean(pred == y).item()


def majority_baseline(train_y, eval_y):
    label = float(np.mean(train_y) >= 0.5)
    return np.mean(eval_y == label).item()


def gradient_checks():
    X = np.array([[1.0, -2.0],
                  [0.5,  3.0]], dtype=np.float64)
    y = np.array([[1.0],
                  [0.0]], dtype=np.float64)
    W = np.array([[0.2],
                  [-0.4]], dtype=np.float64)
    b = np.array([0.1], dtype=np.float64)

    loss, _, _, dW, db = forward_backward(X, y, W, b)
    theta = np.concatenate([W.ravel(), b])
    analytic = np.concatenate([dW.ravel(), db])
    numeric = finite_difference_gradient(theta, X, y)

    direction = np.array([0.3, -0.8, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    fd_direction = (
        parameter_loss(theta + eps * direction, X, y)
        - parameter_loss(theta - eps * direction, X, y)
    ) / (2.0 * eps)
    vjp_direction = analytic @ direction

    print("loss:", loss)
    print("max gradient error:",
          np.max(np.abs(analytic - numeric)))
    print("directional difference:",
          abs(fd_direction - vjp_direction))


def train_demo():
    train, val, test = make_split()
    Xtr, ytr = train
    Xva, yva = val
    Xte, yte = test

    W = np.zeros((2, 1), dtype=np.float64)
    b = np.zeros((1,), dtype=np.float64)
    learning_rate = 0.2

    for _ in range(200):
        _, _, _, dW, db = forward_backward(Xtr, ytr, W, b)
        W -= learning_rate * dW
        b -= learning_rate * db

    train_loss = forward_backward(Xtr, ytr, W, b)[0]
    val_loss = forward_backward(Xva, yva, W, b)[0]
    test_loss = forward_backward(Xte, yte, W, b)[0]

    print("losses:", train_loss, val_loss, test_loss)
    print("accuracies:",
          accuracy(Xtr, ytr, W, b),
          accuracy(Xva, yva, W, b),
          accuracy(Xte, yte, W, b))
    print("baseline val/test:",
          majority_baseline(ytr, yva),
          majority_baseline(ytr, yte))


def failure_tests():
    try:
        bce_logits_mean(
            np.array([[0.0]]),
            np.array([[0.3]])
        )
    except ValueError as exc:
        print("invalid label rejected:", exc)

    try:
        bce_logits_mean(
            np.empty((0, 1)),
            np.empty((0, 1))
        )
    except ValueError as exc:
        print("empty batch rejected:", exc)

    try:
        bce_logits_mean(
            np.array([[np.inf]]),
            np.array([[1.0]])
        )
    except ValueError as exc:
        print("non-finite logit rejected:", exc)


if __name__ == "__main__":
    gradient_checks()
    train_demo()
    failure_tests()
```

有限差分只適合驗證小問題。若參數量為 $P$，逐元素中央差分約需 $2P$ 次前向計算，而反向傳播通常只需一次正向及一次反向。大型模型應採隨機方向導數或抽查參數，而不是建立完整 Jacobian。

---

## 測試與預期結果

由於本章未執行程式，以下是根據公式推導的**預期**，不是實測聲明。

### 正常測試

`gradient_checks()` 使用有限、非極端的輸入：

- 解析梯度與中央有限差分應接近。
- `max gradient error` 在 float64 與 `eps=1e-6` 下通常預期約為 $10^{-7}$ 至 $10^{-9}$ 級，但不同環境不保證固定數值。
- 方向有限差分與 `analytic @ direction` 應接近，驗證 VJP/JVP 的內積對偶。
- 所有梯度 shape 應分別為 `dW.shape == (2, 1)`、`db.shape == (1,)`、`dX.shape == X.shape`。

### 邊界測試

1. **$B=1$**：輸入應保持 `(1, Din)`；不可壓成 `(Din,)`，否則矩陣乘法及 batch reduction 語意可能改變。
2. **極端但有限 logits**：例如 `1000.0` 或 `-1000.0`，穩定 BCE 公式應維持有限損失；穩定 sigmoid 避免直接計算 `exp(1000)`。
3. **空 batch**：mean 沒有明確訓練語意，程式明確拒絕。
4. **單類訓練標籤**：數學上仍可計算，但多數類別基線可能達到很高準確率，必須據實報告，不能只展示模型準確率。

### 故障測試

- 標籤為 `0.3`：不符合本例 Bernoulli 硬標籤契約，應拋出 `ValueError`。
- logit 為 `inf` 或 `nan`：程式明確拒絕，避免非有限值靜默污染梯度。
- `W` shape 寫成 `(1, Din)`：應在 shape 檢查被拒絕，而不是嘗試錯誤轉置來「修好」。
- 若把 `db = dlogits` 而未沿 batch 軸求和，當 $B>1$ 時 `db` shape 錯誤。
- 若把 `dW = dlogits.T @ X`，結果 shape 是 `(1, Din)`，那是本章約定下的轉置錯誤。
- 若 BCE 前向已用 mean，但 `dlogits` 沒除以 $B$，解析梯度將比有限差分大 $B$ 倍。

### 訓練結果的解讀限制

`train_demo()` 預期使訓練損失下降，但特定 seed 的驗證或測試準確率不保證單調改善。只有一次合成切分，不能支持統計顯著性主張。測試集雖由程式列印，仍不應被用來反覆挑選學習率或訓練步數；若已用測試結果調參，該集合便不再是未見測試集。

---

## 反例與常見陷阱

### 陷阱一：NumPy一維陣列的轉置

```python
x = np.array([1.0, 2.0])
print(x.shape)    # (2,)
print(x.T.shape)  # 仍是(2,)
```

一維陣列沒有明確 row／column 軸，`.T` 不會改 shape。若需要單筆 batch，應寫成 `x.reshape(1, 2)`；若需要 column 向量，應寫成 `x.reshape(2, 1)`。`reshape` 改變軸的組織，而 transpose 交換既有軸，兩者不可混稱。

### 陷阱二：錯誤廣播產生「可執行但錯誤」結果

若 logits 是 $(B,1)$，標籤錯寫成 $(B,)$，NumPy 相減可能廣播成 $(B,B)$。程式未必報錯，但每筆樣本會錯誤地與所有標籤配對。本章因此要求 `logits.shape == y.shape`，而不接受「元素總數相同」作為替代。

### 陷阱三：分支梯度覆寫

錯誤概念如下：

```python
da = db * (2 * a)
da = dc * 3
```

第二行覆蓋第一條路徑。正確形式是：

```python
da = db * (2 * a)
da += dc * 3
```

共享權重、殘差連接與同一 embedding 索引重複出現時，都有相同的累加要求。

### 陷阱四：用有限差分取代推導

有限差分只能提供數值證據，不能證明公式對所有輸入成立。步長太大會有截斷誤差，太小則有浮點消去誤差；不可因單一 `eps` 不符就立即認定解析梯度錯誤。合理做法是：

1. 先重查前向損失究竟是 sum 還是 mean。
2. 核對每個 shape 與 reduction 軸。
3. 使用 float64。
4. 比較多個步長，例如 $10^{-4}$、$10^{-5}$、$10^{-6}$。
5. 避開 ReLU 零點等不可微位置。
6. 同時檢查逐元素差分及方向導數。

### 陷阱五：非光滑點硬套中央差分

對 ReLU

$$
\operatorname{ReLU}(x)=\max(0,x),
$$

$x=0$ 處沒有唯一導數。框架可以約定回傳 $0$，但中央差分在零點得到約 $1/2$。兩者不同不必然表示反向程式錯誤，而是局部導數約定與對稱差分所測量的量不同。

### 陷阱六：建立完整大型Jacobian

假設一層輸入及輸出各有一百萬個元素，完整 Jacobian 有 $10^{12}$ 個元素，通常完全不實際。反向傳播的目的正是利用計算圖結構直接計算所需 VJP。只有在極小教學問題中，完整 Jacobian 才適合拿來展示定義。

---

## AI、幾何與養殖案例

### AI：共享參數是梯度累加問題

神經網路中的同一權重矩陣可能在多個 token、時間步或分支重複使用。若

$$
L=L_1(W)+L_2(W),
$$

則

$$
\nabla_WL=\nabla_WL_1+\nabla_WL_2.
$$

這是共享參數訓練的基本機制。注意力、循環計算與權重共享都依賴相同原理。梯度不是「選出最重要的一條路」，而是總和所有影響損失的可微路徑。

### 幾何：梯度是局部最陡上升方向

純量函數 $f(x)$ 的一階近似是

$$
f(x+\epsilon u)
\approx f(x)+\epsilon\nabla f(x)^Tu.
$$

若限制 $\|u\|_2=1$，由 Cauchy–Schwarz 不等式，

$$
\nabla f(x)^Tu
\leq \|\nabla f(x)\|_2.
$$

等號在 $u=\nabla f/\|\nabla f\|_2$ 時成立。因此負梯度是局部一階近似下的最陡下降方向。但這只是局部敘述，不保證固定大步長一定降低損失，也不保證到達全域最小值。

### 合成養殖案例：只作離線風險分類

假設以完全合成的兩個特徵表示感測摘要：

- $x_1$：經訓練集統計標準化的合成溶氧指標；
- $x_2$：經訓練集統計標準化的合成水溫指標。

模型輸出一個二分類 logit，預測合成日誌中的「需人工複核」標籤。此案例只用來說明離線計算圖：

$$
z=x_1w_1+x_2w_2+b.
$$

若某筆樣本的損失對 logit 梯度為 $\delta$，則

$$
\nabla_wL=
\begin{bmatrix}x_1\\x_2\end{bmatrix}\delta,
\qquad
\nabla_xL=
\begin{bmatrix}w_1&w_2\end{bmatrix}\delta.
$$

資料必須先按養殖槽、來源群組或時間區間切分，再以訓練資料擬合標準化統計；不得讓同一槽的高度相似時間片分散到訓練與測試集合。這裡沒有真實操作閾值，模型輸出也不是控制授權，不得據此控制曝氣、投餌、泵浦、加藥或其他設備。即使測試準確率高，也只能說明指定合成分布上的分類結果，不能推論真實養殖安全性。

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

求 $Z$、$L$、$\nabla_ZL$、$\nabla_XL$、$\nabla_WL$ 與 $\nabla_bL$，並列出每個量的 shape。

### 二、程式題

修改本章程式，加入輸入 $X$ 的方向導數檢查。給定與 $X$ 同 shape 的方向 $U$，比較

$$
\frac{L(X+\epsilon U)-L(X-\epsilon U)}{2\epsilon}
$$

與

$$
\langle \nabla_XL,U\rangle_F.
$$

不得建立對 $X$ 的完整 Jacobian。

### 三、反例題

某人聲稱：「若 $z=x+x$，反向傳播只需把下游梯度傳回 $x$ 一次，因為兩個 $x$ 是同一變數。」請給出一個數值反例，說明此作法錯在哪裡。

### 四、整合題

設合成序列特徵 $X$ 的 shape 為 $(B,T,D)$，共享權重 $W$ 的 shape 為 $(D,K)$，偏置 $b$ 的 shape 為 $(K,)$：

$$
Z_{b,t,:}=X_{b,t,:}W+b.
$$

損失是所有 $B T$ 個位置之元素平方和的一半再除以 $BT$：

$$
L=\frac{1}{2BT}\sum_{b,t,k}Z_{btk}^2.
$$

推導 $dX$、$dW$、$db$，明確指出 reduction 軸與平均因子。若部分時間位置無效，另以布林 mask $M\in\{0,1\}^{B\times T}$ 表示，應如何改成有效位置平均？空有效集合應如何處理？

---

## 習題解答

### 一、手算題解答

此處 $B=2$。先計算

$$
XW=
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

沿 batch 軸廣播 $b=-1$：

$$
Z=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad Z\text{ shape}=(2,1).
$$

損失為

$$
L=\frac{1}{2\cdot2}(1^2+0^2)=\frac14.
$$

由

$$
L=\frac{1}{2B}\sum_i Z_i^2
$$

可得

$$
\nabla_ZL=\frac{Z}{B}
=
\begin{bmatrix}
1/2\\
0
\end{bmatrix},
\qquad \text{shape}=(2,1).
$$

因此

$$
\nabla_XL=\nabla_ZL\,W^T
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
\end{bmatrix},
$$

shape 為 $(2,2)$。

再算

$$
\nabla_WL=X^T\nabla_ZL.
$$

因為

$$
X^T=
\begin{bmatrix}
1&2\\
0&-1
\end{bmatrix},
$$

故

$$
\nabla_WL=
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
\end{bmatrix},
$$

shape 為 $(2,1)$。

偏置梯度沿 batch 軸 $0$ 求和：

$$
\nabla_bL=\sum_{i=1}^{2}(\nabla_ZL)_i
=\frac12,
$$

shape 為 $(1,)$。

### 二、程式題解答

可加入：

```python
def input_direction_check(X, y, W, b, U, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if U.shape != X.shape:
        raise ValueError("U必須與X同shape")

    _, _, dX, _, _ = forward_backward(X, y, W, b)

    loss_plus = forward_backward(X + eps * U, y, W, b)[0]
    loss_minus = forward_backward(X - eps * U, y, W, b)[0]
    finite_difference = (loss_plus - loss_minus) / (2.0 * eps)

    vjp_inner_product = np.sum(dX * U)
    return finite_difference, vjp_inner_product
```

`np.sum(dX * U)` 是 Frobenius 內積

$$
\langle dX,U\rangle_F.
$$

兩個純量應在合理浮點誤差內接近。若差異大，應檢查平均因子、shape、有限差分步長與輸入是否有限。

### 三、反例題解答

令

$$
z=x+x,\qquad L=z^2,
$$

取 $x=3$。正向得

$$
z=6,\qquad L=36.
$$

先有

$$
\frac{dL}{dz}=2z=12.
$$

加法節點的兩個輸入都等於同一變數 $x$，每條路徑各傳回 $12$，所以

$$
\frac{dL}{dx}=12+12=24.
$$

直接展開：

$$
L=(2x)^2=4x^2,
\qquad
\frac{dL}{dx}=8x.
$$

代入 $x=3$ 也得到 $24$。若只傳一次，會錯得 $12$。同一變數出現兩次不會取消其中一條依賴路徑，反而要求累加兩次貢獻。

### 四、整合題解答

令

$$
G=\nabla_ZL=\frac{Z}{BT},
$$

shape 為 $(B,T,K)$。平均因子只在此引入一次。

把每個 $(b,t)$ 視為一筆樣本，可得

$$
dX=GW^T,
$$

shape 為 $(B,T,D)$；此處矩陣乘法作用於最後一軸。

權重在所有 batch 與時間位置共享，因此

$$
dW=\sum_{b=1}^{B}\sum_{t=1}^{T}
X_{bt,:}^TG_{bt,:},
$$

shape 為 $(D,K)$。NumPy 可寫成：

```python
dW = np.einsum("btd,btk->dk", X, G)
```

偏置沿 batch 軸 $0$ 及時間軸 $1$ 求和：

$$
db=\sum_{b=1}^{B}\sum_{t=1}^{T}G_{bt,:},
$$

shape 為 $(K,)$。NumPy 寫成：

```python
db = np.sum(G, axis=(0, 1))
```

若有 mask $M\in\{0,1\}^{B\times T}$，有效位置數為

$$
N_{\mathrm{eff}}=\sum_{b,t}M_{bt}.
$$

損失應定義為

$$
L=
\frac{1}{2N_{\mathrm{eff}}}
\sum_{b,t,k}M_{bt}Z_{btk}^2.
$$

因此

$$
G_{btk}
=
\frac{M_{bt}Z_{btk}}{N_{\mathrm{eff}}}.
$$

mask 可用 `M[:, :, None]` 擴充成 $(B,T,1)$，再沿 $K$ 軸廣播。後續 $dX$、$dW$、$db$ 公式不變。除數是有效位置數，不是 $BTK$；每個有效位置的損失在特徵 $K$ 上取 sum，再對有效位置取 mean。若 $N_{\mathrm{eff}}=0$，應明確拒絕，不能除以零，也不能靜默把 loss 設成零。

---

## 本章小結

反向傳播不是獨立於微積分的神祕演算法，而是鏈式法則在計算圖上的有效組織方式。對純量損失而言，我們通常需要的是 VJP，而不是完整 Jacobian。

本章的核心規則如下：

- 以 $df=\operatorname{tr}(G^TdX)$ 定義矩陣梯度。
- 梯度必須與被微分變數具有相同 shape。
- 對 $Z=XW+b$，有
  $dX=dZW^T$、$dW=X^TdZ$，而 $db$ 沿廣播軸求和。
- 計算圖依反向拓撲順序處理。
- 一個值若有多條下游路徑，梯度必須累加。
- mean loss 的平均因子只施加一次，且必須明列 reduction 軸。
- 有限差分與內積對偶是驗證工具，不是數學證明。
- 梯度正確只證明局部微分實作一致，不證明資料無洩漏、模型能泛化或實際應用安全。

下一步可把這些規則套用到具有任意前導 batch／time 軸的仿射層，進一步處理 broadcast、共享權重及重複使用參數的梯度。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, <https://arxiv.org/abs/1706.03762>。本章只將其視為後續共享參數與 Transformer 計算圖的背景入口；依提供的來源註記，僅摘要頁已取得，不能宣稱已完整核對論文。
2. NumPy, *Broadcasting*，<https://numpy.org/doc/stable/user/basics.broadcasting.html>。依來源註記，此入口尚待逐條核對；本章的廣播反向公式由 shape 與微分直接推導，不以該頁作已查證證據。
3. Dive into Deep Learning, <https://d2l.ai/>。僅列為延伸閱讀入口；依來源註記，未逐章核對。
4. PyTorch, *Reproducibility*, <https://docs.pytorch.org/docs/stable/notes/randomness.html>。僅列為後續框架可重現性議題的延伸入口；本章 NumPy 程式未依此宣稱跨平台逐位重現。