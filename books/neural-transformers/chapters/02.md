# 第02章 張量形狀、批次與廣播

## 學習目標與先備知識

神經網路中的許多錯誤，不是公式本身錯誤，而是公式與程式對「哪一軸代表什麼」理解不同。兩個陣列即使元素總數相同，也不表示它們代表相同資料；一個能成功執行的廣播運算，也不表示其語義正確。

完成本章後，讀者應能：

1. 說明標準軸記號：
   - $B$：batch，批次中的樣本數。
   - $T$：sequence length，序列長度。
   - $D$：model feature，模型特徵維度。
   - $H$：attention head，注意力頭數。
   - $d_h=D/H$：每個注意力頭的特徵維度。
2. 為每個張量明列 shape 與各軸語義，而不只列元素總數。
3. 區分 `reshape`、`transpose` 與矩陣乘法。
4. 使用 `matmul` 與 `einsum` 表達批次線性運算。
5. 明確指出 reduction 的軸，以及使用 `sum` 或 `mean`。
6. 根據廣播規則判斷運算是否合法，並辨識「合法但語義錯誤」的廣播。
7. 保留 $B=1$ 的批次軸，避免單樣本與多樣本使用不同介面。
8. 理解 transpose 後的陣列可能是非連續布局；非連續不表示數值錯誤，但會影響 reshape 是否需要複製資料。
9. 在 loss、機率與資料切分中保留軸與平均方式的完整契約。

本章假設讀者熟悉純量、向量、矩陣及矩陣乘法。本文以 NumPy 為實作工具，只要求 CPU，不下載模型、語料或其他資料。本章未執行程式，也未核對本機 NumPy 版本；所有測試輸出均為依程式推導的**預期結果**，不構成實測紀錄。

---

## 問題與直覺

考慮一批序列資料：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

若 $X$ 的 shape 是 `(2,3,4)`，這串數字本身並不完整。我們還必須說：

- 第 $0$ 軸是 batch，大小為 $B=2$；
- 第 $1$ 軸是 time，大小為 $T=3$；
- 第 $2$ 軸是 feature，大小為 $D=4$。

如果把它轉置成 `(3,2,4)`，元素沒有增加或減少，但語義已變成 `(T,B,D)`。若後續函式仍假定輸入是 `(B,T,D)`，程式有時會立即報錯，有時卻會悄悄算出錯誤結果。

形狀錯誤大致可分成三類：

1. **立即失敗**：矩陣乘法的內維度不相容，NumPy 丟出例外。
2. **shape 合法但語義錯誤**：例如錯把 time 軸當 batch 軸做平均。
3. **被廣播掩蓋的錯誤**：shape 可以對齊，程式正常執行，但參數沿錯誤軸重複。

因此，可靠的張量程式至少需要三層契約：

- **shape 契約**：每一軸的大小。
- **語義契約**：每一軸代表什麼。
- **reduction 契約**：在哪些軸做 `sum` 或 `mean`，以及是否保留被縮減的軸。

可以把 shape 視為型別的一部分。`(B,T,D)` 與 `(T,B,D)` 雖然都是三階張量，卻不應被視為相同型別。

---

## 定義、定理與推導

### 1. 張量、索引與軸

一個三階張量

$$
X\in\mathbb{R}^{B\times T\times D}
$$

的元素寫成 $X_{btd}$，其中

$$
0\le b<B,\qquad 0\le t<T,\qquad 0\le d<D.
$$

NumPy 採零起始索引，因此 `X[b, t, d]` 對應 $X_{btd}$。

本卷固定使用下列主要軸：

| 符號 | 意義 | 常見位置 |
|---|---|---|
| $B$ | batch size | 第 $0$ 軸 |
| $T$ | sequence length | 第 $1$ 軸 |
| $D$ | model feature | 最後一軸 |
| $H$ | attention heads | 拆頭後第 $1$ 軸 |
| $d_h$ | 每頭維度，$D/H$ | 拆頭後最後一軸 |

批次線性層使用

$$
X\in\mathbb{R}^{B\times D_{\text{in}}},
\quad
W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}},
\quad
b\in\mathbb{R}^{D_{\text{out}}},
$$

並定義

$$
Y=XW+b\in\mathbb{R}^{B\times D_{\text{out}}}.
$$

batch 中每筆樣本以一個橫列儲存。這是資料布局約定，與微分教材常用的 column 向量表示不矛盾。

對序列輸入，線性層逐 token 作用：

$$
X\in\mathbb{R}^{B\times T\times D_{\text{in}}},
$$

$$
Y_{bto}=\sum_{i=0}^{D_{\text{in}}-1}X_{bti}W_{io}+b_o.
$$

輸出 shape 為 $(B,T,D_{\text{out}})$。權重 $W$ 與偏置 $b$ 在所有 batch 與 time 位置共享。

### 2. reshape 不等於 transpose

`reshape` 改變索引分組方式，通常依目前元素遍歷順序重新解釋資料；`transpose` 則重新排列軸。

令

$$
X=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix},
\qquad
X\in\mathbb{R}^{2\times3}.
$$

則

$$
\operatorname{reshape}(X,(3,2))
=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix},
$$

而

$$
X^\mathsf{T}
=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}.
$$

兩者 shape 都是 $(3,2)$，數值位置卻不同。不能因為目標 shape 相同，就用 `reshape` 取代 `transpose`。

對多頭張量尤其如此。從 $(B,T,D)$ 拆成 $H$ 個頭，通常先做

$$
(B,T,D)\xrightarrow{\text{reshape}}(B,T,H,d_h),
$$

再做

$$
(B,T,H,d_h)\xrightarrow{\text{transpose}}(B,H,T,d_h).
$$

只執行 `reshape(B,H,T,dh)`，會把原本相鄰的元素錯分到 head 與 time 軸。

### 3. matmul 的形狀

普通矩陣乘法為

$$
A\in\mathbb{R}^{M\times K},
\quad
W\in\mathbb{R}^{K\times N},
\quad
C=AW\in\mathbb{R}^{M\times N},
$$

其中

$$
C_{mn}=\sum_{k=0}^{K-1}A_{mk}W_{kn}.
$$

`np.matmul` 或運算子 `@` 對高階張量使用最後兩軸做矩陣乘法，前導軸依廣播規則配對。例如：

$$
A\in\mathbb{R}^{B\times M\times K},
\qquad
W\in\mathbb{R}^{K\times N},
$$

則

$$
A@W\in\mathbb{R}^{B\times M\times N}.
$$

最後兩軸中的 $K$ 被縮減，前導 batch 軸 $B$ 被保留。這正好表示同一權重矩陣作用於 batch 中每筆資料。

再看一個真正的批次矩陣乘法：

$$
Q\in\mathbb{R}^{B\times H\times T_q\times d_h},
\qquad
K\in\mathbb{R}^{B\times H\times T_k\times d_h}.
$$

先交換 $K$ 的最後兩軸：

$$
K^\mathsf{T}_{\text{last}}
\in\mathbb{R}^{B\times H\times d_h\times T_k}.
$$

因此

$$
QK^\mathsf{T}_{\text{last}}
\in\mathbb{R}^{B\times H\times T_q\times T_k}.
$$

這裡的 transpose 只交換最後兩軸，不是把整個四階張量當成二維矩陣轉置。

### 4. einsum：把索引寫進程式

序列線性層可以寫成：

```python
Y = np.einsum("bti,io->bto", X, W) + b
```

字串 `"bti,io->bto"` 表示：

- `X` 軸為 batch、time、input feature；
- `W` 軸為 input feature、output feature；
- 重複但未出現在輸出的索引 `i` 被求和；
- 輸出保留 `b,t,o`。

相同運算也可寫成：

```python
Y = X @ W + b
```

`einsum` 的優點是縮減索引較明確，但字母只是局部標記。NumPy 不知道 `b` 是否真的代表 batch，因此 `einsum` 不能取代 shape 斷言與介面文件。

### 5. 廣播規則

比較兩個 shape 時，從最右邊的軸開始對齊。每一對軸若符合以下任一條件，即可廣播：

1. 大小相等；
2. 其中一邊大小為 $1$；
3. 較短 shape 在該位置沒有軸，可視為補上大小 $1$ 的軸。

例如：

$$
X:(B,T,D),\qquad b:(D).
$$

從右對齊後，相當於：

$$
X:(B,T,D),\qquad b:(1,1,D).
$$

所以 `X + b` 合法，且 $b_d$ 會沿 batch 與 time 重複使用。

但合法不表示符合目的。假設要為每個 time step 加入位置值

$$
p\in\mathbb{R}^{T}.
$$

直接計算 `X + p` 會把 `p` 對齊最後一軸。只有在 $T=D$ 時它才碰巧合法，而且實際上會沿 feature 軸相加。正確表示應是

$$
p\in\mathbb{R}^{1\times T\times1},
$$

即：

```python
X + p[None, :, None]
```

### 6. 小命題：廣播值的縮減與廣播梯度

**命題。** 設 $c\in\mathbb{R}^{D}$，把它廣播成 $\widetilde c\in\mathbb{R}^{B\times T\times D}$，定義

$$
\widetilde c_{btd}=c_d.
$$

則對每個固定的 $d$，

$$
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde c_{btd}=BTc_d,
$$

且

$$
\frac1{BT}\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde c_{btd}=c_d.
$$

**證明。**

固定任意特徵索引 $d$。依廣播定義，對所有 $b,t$ 都有 $\widetilde c_{btd}=c_d$。因此

$$
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde c_{btd}
=
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}c_d.
$$

內層共有 $T$ 個相同的 $c_d$，外層共有 $B$ 組，所以總共有 $BT$ 項：

$$
\sum_b\sum_t c_d=BTc_d.
$$

兩邊除以 $BT$，便得到平均式。證畢。

前述命題描述前向廣播值的重複次數。反向傳播還需要由微分另行推導。若純量損失 $L$ 對廣播輸出 $\widetilde c$ 的梯度為

$$
G_{btd}=\frac{\partial L}{\partial\widetilde c_{btd}},
$$

因為

$$
d\widetilde c_{btd}=dc_d,
$$

所以

$$
dL
=
\sum_{b,t,d}G_{btd}\,d\widetilde c_{btd}
=
\sum_{b,t,d}G_{btd}\,dc_d.
$$

把同一個 $dc_d$ 的係數收集起來：

$$
dL
=
\sum_d
\left(
\sum_{b,t}G_{btd}
\right)dc_d.
$$

因此

$$
\frac{\partial L}{\partial c_d}
=
\sum_{b,t}G_{btd}.
$$

這才是廣播反向需沿 batch 與 time 軸求和的理由。梯度最後必須還原成參數的原 shape $(D,)$，不能保留為 $(B,T,D)$。

### 7. reduction 必須寫清楚軸

對 $X\in\mathbb{R}^{B\times T\times D}$：

- `X.sum(axis=1)`：沿 time 軸求和，輸出 shape 為 $(B,D)$。
- `X.mean(axis=(0,1))`：沿 batch 與 time 軸平均，輸出 shape 為 $(D,)$。
- `X.mean(axis=-1, keepdims=True)`：沿 feature 軸平均，輸出 shape 為 $(B,T,1)$。
- `X.mean()`：沿所有軸平均，輸出純量。

`keepdims=True` 會保留被縮減的軸，並把其大小設為 $1$。例如逐 token 中心化：

$$
\mu_{bt}=\frac1D\sum_{d=0}^{D-1}X_{btd},
\qquad
\mu\in\mathbb{R}^{B\times T\times1}.
$$

程式應寫成：

```python
mu = X.mean(axis=-1, keepdims=True)
centered = X - mu
```

此處平均只沿 feature 軸，batch 與 time 軸均被保留。

### 8. loss 平均、機率域與資料切分

若每筆樣本損失為 $\ell_b$，batch 平均損失明定為

$$
L=\frac1B\sum_{b=0}^{B-1}\ell_b.
$$

若序列含有效位置遮罩 $m_{bt}\in\{0,1\}$，有效 token 平均為

$$
L=
\frac{\sum_{b,t}m_{bt}\ell_{bt}}
{\sum_{b,t}m_{bt}},
$$

前提是

$$
\sum_{b,t}m_{bt}>0.
$$

分母為零時應明確拒絕，而不是回傳 NaN 或任意零。若 $\ell_{bt}$ 已是逐 token 損失，就以有效 token 總數作唯一分母，平均只除一次。

不能先對每條序列平均，再把序列平均值等權平均，除非這正是明定的評估目標。兩種平均在序列有效長度不同時通常不相等。

若張量表示離散機率

$$
P\in\mathbb{R}^{B\times T\times C},
$$

類別軸為最後一軸，則必須滿足

$$
P_{btc}\ge0,
\qquad
\sum_{c=0}^{C-1}P_{btc}=1.
$$

歸一化是沿 `axis=-1`，不是沿 batch 或 time。零機率可以是合法分布的一部分；但若後續計算 $\log P_{btc}$，則 $\log0=-\infty$，必須按損失的數學定義處理，不能任意加入 epsilon 後宣稱仍是精確同一運算。NaN、正無限與負無限不屬於有限機率值，核心介面應拒絕。

資料 shape 也不能取代資料切分。若同一條原始序列產生多個重疊窗口，即使每個窗口在 batch 中佔不同橫列，也不表示它們彼此獨立。正確順序是：

1. 先按文件、個體、群組或時間切成訓練／驗證／測試集合；
2. 只用訓練集擬合詞表、標準化統計與其他轉換；
3. 分別在三個集合內建立窗口；
4. 驗證集用於模型選擇，測試集不參與調參。

---

## 逐步手算例題

### 例題一：批次線性層與偏置廣播

給定

$$
X=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix},
\quad
W=
\begin{bmatrix}
1&-1&2\\
0&3&1
\end{bmatrix},
\quad
b=[1,0,-2].
$$

shape 分別是

$$
X:(B,D_{\text{in}})=(2,2),
$$

$$
W:(D_{\text{in}},D_{\text{out}})=(2,3),
$$

$$
b:(D_{\text{out}})=(3).
$$

第一筆樣本：

$$
[1,2]W
=
[
1\cdot1+2\cdot0,\;
1\cdot(-1)+2\cdot3,\;
1\cdot2+2\cdot1
]
=[1,5,4].
$$

第二筆樣本：

$$
[3,4]W
=
[
3\cdot1+4\cdot0,\;
3\cdot(-1)+4\cdot3,\;
3\cdot2+4\cdot1
]
=[3,9,10].
$$

所以

$$
XW=
\begin{bmatrix}
1&5&4\\
3&9&10
\end{bmatrix}.
$$

偏置 $b:(3,)$ 從右對齊輸出 `(2,3)`，沿 batch 軸廣播：

$$
Y=XW+b
=
\begin{bmatrix}
2&5&2\\
4&9&8
\end{bmatrix}.
$$

若純量損失定義為全部輸出元素的平均：

$$
L=\frac1{BD_{\text{out}}}\sum_{b,o}Y_{bo},
$$

則

$$
L=\frac{2+5+2+4+9+8}{2\cdot3}=5.
$$

這裡 reduction 軸是 batch 與 output feature，分母為六個元素，且只除一次。

### 例題二：reshape、transpose 與拆頭

令 $B=1,T=2,H=2,d_h=2$，所以 $D=Hd_h=4$。輸入為

$$
X=
\left[
\begin{array}{cccc}
1&2&3&4\\
5&6&7&8
\end{array}
\right],
$$

shape 是 $(1,2,4)=(B,T,D)$。

第一步，把最後一軸拆成 $(H,d_h)$：

$$
X_r=\operatorname{reshape}(X,(1,2,2,2)).
$$

若依 `(B,T,H,dh)` 解讀：

- time 0：
  - head 0：$[1,2]$
  - head 1：$[3,4]$
- time 1：
  - head 0：$[5,6]$
  - head 1：$[7,8]$

第二步，把 head 軸移到 time 軸前：

$$
X_h=\operatorname{transpose}(X_r,(0,2,1,3)).
$$

結果 shape 仍為 $(1,2,2,2)$，但軸語義是 `(B,H,T,dh)`：

- head 0：
  - time 0：$[1,2]$
  - time 1：$[5,6]$
- head 1：
  - time 0：$[3,4]$
  - time 1：$[7,8]$

若直接執行 `X.reshape(1,2,2,2)` 並宣稱結果已是 `(B,H,T,dh)`，就會被錯誤解讀為：

- head 0：$[1,2]$、$[3,4]$
- head 1：$[5,6]$、$[7,8]$

兩者 shape 完全相同，但索引語義不同。

### 例題三：遮罩損失的正確平均

逐 token 損失及有效遮罩為

$$
\ell=
\begin{bmatrix}
2&4&100\\
3&5&7
\end{bmatrix},
\qquad
m=
\begin{bmatrix}
1&1&0\\
1&1&1
\end{bmatrix}.
$$

有效損失總和為

$$
2+4+3+5+7=21.
$$

有效 token 數為

$$
2+3=5.
$$

因此

$$
L=\frac{21}{5}=4.2.
$$

被遮罩位置的 $100$ 不進入分子或分母。

若先算每列平均，再把兩列等權平均：

$$
\frac12
\left(
\frac{2+4}{2}
+
\frac{3+5+7}{3}
\right)
=
\frac12(3+5)=4.
$$

結果 $4$ 與有效 token 平均 $4.2$ 不同。前者讓每條序列等權，後者讓每個有效 token 等權；不能混稱為同一種 loss。

---

## 實作與程式

以下是自足的 NumPy CPU 程式，示範線性層、`einsum`、拆頭、合頭、reduction、機率檢查、有效 token 平均，以及正常／邊界／故障測試。程式不使用網路、GPU、外部模型或外部資料。

```python
import numpy as np


def linear_sequence(x, w, bias):
    """x:(B,T,Din), w:(Din,Dout), bias:(Dout,) -> (B,T,Dout)"""
    if x.ndim != 3:
        raise ValueError("x 必須是 (B,T,Din) 三階張量")
    if w.ndim != 2:
        raise ValueError("w 必須是 (Din,Dout) 矩陣")
    if bias.ndim != 1:
        raise ValueError("bias 必須是 (Dout,) 一維陣列")

    B, T, Din = x.shape
    win, Dout = w.shape
    if Din != win:
        raise ValueError("x 的 Din 與 w 的第一軸不一致")
    if bias.shape != (Dout,):
        raise ValueError("bias shape 必須恰為 (Dout,)")

    y_matmul = x @ w + bias
    y_einsum = np.einsum("bti,io->bto", x, w) + bias
    np.testing.assert_allclose(y_matmul, y_einsum)
    assert y_matmul.shape == (B, T, Dout)
    return y_matmul


def split_heads(x, H):
    """x:(B,T,D) -> (B,H,T,dh)"""
    if x.ndim != 3:
        raise ValueError("x 必須是 (B,T,D)")
    if not isinstance(H, (int, np.integer)) or H <= 0:
        raise ValueError("H 必須是正整數")

    B, T, D = x.shape
    if D % H != 0:
        raise ValueError("D 必須可被 H 整除")

    dh = D // H
    return x.reshape(B, T, H, dh).transpose(0, 2, 1, 3)


def merge_heads(x):
    """x:(B,H,T,dh) -> (B,T,D)"""
    if x.ndim != 4:
        raise ValueError("x 必須是 (B,H,T,dh)")

    B, H, T, dh = x.shape
    # transpose 後可能非 C-contiguous；reshape 可在需要時建立副本。
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


def masked_token_mean(losses, valid):
    """losses:(B,T), valid:(B,T) bool -> finite scalar"""
    if losses.ndim != 2:
        raise ValueError("losses 必須是 (B,T)")
    if valid.shape != losses.shape:
        raise ValueError("valid 必須與 losses 同 shape")
    if valid.dtype != np.bool_:
        raise TypeError("valid 必須是布林陣列")

    count = int(valid.sum())
    if count == 0:
        raise ValueError("至少必須有一個有效 token")

    selected = losses[valid]
    if not np.isfinite(selected).all():
        raise ValueError("有效位置的 loss 必須全部有限")

    total = selected.sum(dtype=np.float64)
    return total / count


def require_probabilities(p):
    """檢查 p:(...,C) 是否為最後一軸上的有限離散機率分布。"""
    if p.ndim < 1 or p.shape[-1] == 0:
        raise ValueError("類別軸必須存在且非空")
    if not np.isfinite(p).all():
        raise ValueError("機率不得含 NaN 或無限值")
    if np.any(p < 0.0):
        raise ValueError("機率不得為負")

    sums = p.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("機率必須沿最後類別軸加總為 1")


def expect_error(fn, error_type):
    try:
        fn()
    except error_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"預期 {error_type.__name__}，實際得到 {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"預期丟出 {error_type.__name__}")


def main():
    # 正常案例：B=2 的序列線性層。
    x = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    w = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, -1.0]])
    bias = np.array([0.5, -0.5])

    y = linear_sequence(x, w, bias)
    expected_y = np.array([[[2.5, -1.5],
                            [8.5, -1.5]],
                           [[14.5, -1.5],
                            [20.5, -1.5]]])
    np.testing.assert_allclose(y, expected_y)
    assert y.shape == (2, 2, 2)

    # 邊界案例：B=1 仍保留 batch 軸。
    x_one = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]])
    y_one = linear_sequence(x_one, w, bias)
    assert x_one.shape == (1, 2, 3)
    assert y_one.shape == (1, 2, 2)

    # 拆頭與合頭互逆。
    z = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    zh = split_heads(z, H=2)
    assert zh.shape == (2, 2, 2, 2)
    z_back = merge_heads(zh)
    np.testing.assert_array_equal(z_back, z)

    # transpose 產生本例中的非 C-contiguous view。
    q = np.arange(24).reshape(2, 3, 4)
    qt = q.transpose(0, 2, 1)
    assert qt.shape == (2, 4, 3)
    assert not qt.flags["C_CONTIGUOUS"]

    qc = np.ascontiguousarray(qt)
    assert qc.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(qc, qt)

    # 沿 feature 軸平均，保留大小為 1 的軸。
    mu = q.mean(axis=-1, keepdims=True)
    assert mu.shape == (2, 3, 1)
    centered = q - mu
    np.testing.assert_allclose(centered.mean(axis=-1), 0.0)

    # 有效 token 平均。
    losses = np.array([[2.0, 4.0, 100.0],
                       [3.0, 5.0, 7.0]])
    valid = np.array([[True, True, False],
                      [True, True, True]])
    result = masked_token_mean(losses, valid)
    np.testing.assert_allclose(result, 4.2)

    # 無效位置可為 NaN，因為不進入結果；有效位置必須有限。
    losses_with_masked_nan = losses.copy()
    losses_with_masked_nan[0, 2] = np.nan
    np.testing.assert_allclose(
        masked_token_mean(losses_with_masked_nan, valid),
        4.2
    )

    # 機率沿最後類別軸歸一化。
    p = np.array([[[0.25, 0.75],
                   [1.00, 0.00]]])
    require_probabilities(p)

    # 故障：D 不能被 H 整除。
    expect_error(
        lambda: split_heads(np.zeros((1, 2, 5)), H=2),
        ValueError
    )

    # 故障：完全沒有有效 token。
    expect_error(
        lambda: masked_token_mean(
            np.ones((1, 2)),
            np.zeros((1, 2), dtype=bool)
        ),
        ValueError
    )

    # 故障：有效位置含 NaN。
    expect_error(
        lambda: masked_token_mean(
            np.array([[1.0, np.nan]]),
            np.array([[True, True]])
        ),
        ValueError
    )

    # 故障：負機率。
    expect_error(
        lambda: require_probabilities(np.array([[1.1, -0.1]])),
        ValueError
    )

    # 故障：錯誤偏置 shape，避免意外廣播。
    expect_error(
        lambda: linear_sequence(
            np.zeros((2, 3, 4)),
            np.zeros((4, 5)),
            np.zeros((3, 1))
        ),
        ValueError
    )

    print("所有內建檢查完成。")


if __name__ == "__main__":
    main()
```

`masked_token_mean` 的契約是：無效位置不參與運算，因此無效位置即使含非有限值，也不影響結果；所有有效位置則必須為有限值。這項策略由程式與測試共同明定。

程式也刻意不接受 shape 為 `(1,Dout)` 的偏置，即使它在部分情況下可以廣播。嚴格限制為 `(Dout,)` 能縮小介面，避免呼叫端把其他軸的參數誤當成 feature 偏置。

---

## 測試與預期結果

本章未指定或核對 NumPy 版本，也未執行上述程式。程式只使用常見的 `ndarray`、`reshape`、`transpose`、`einsum`、`matmul`、`isfinite` 與 `numpy.testing` 功能；以下是依程式逐步推導的預期結果，不是本機測試報告。

### 1. 正常測試

對

```python
x = np.arange(12).reshape(2, 2, 3)
```

可分成四個長度為三的 token：

$$
[0,1,2],\ [3,4,5],\ [6,7,8],\ [9,10,11].
$$

配合程式中的 $W$ 與偏置，輸出預期為

```text
[[[ 2.5, -1.5],
  [ 8.5, -1.5]],

 [[14.5, -1.5],
  [20.5, -1.5]]]
```

shape 為 `(2,2,2)`。`x @ w + bias` 與 `einsum("bti,io->bto", x, w) + bias` 預期逐元素相同。

### 2. 邊界測試：$B=1$

輸入明確建成 `(1,2,3)`，而不是 `(2,3)`；輸出預期保持 `(1,2,2)`。這使同一函式能一致處理 $B=1$ 與 $B>1$。

不應用無參數的 `squeeze()` 隨意處理 batch，因為它會移除所有大小為 $1$ 的軸。若 $B=1,T=1$，可能同時失去 batch 與 time 軸。只有介面明確要求時，才應使用 `squeeze(axis=指定軸)`。

### 3. 非連續布局測試

本例中的

```python
qt = q.transpose(0, 2, 1)
```

預期不是 C-contiguous。這不影響索引數值；`np.ascontiguousarray(qt)` 得到的 `qc` 預期與 `qt` 逐元素相同。

不能由此推論「所有 transpose 結果都一定非連續」，也不能假設所有 reshape 都不複製資料。是否共享記憶體屬於布局問題，應與張量數值及軸語義分開檢查。

### 4. reduction 與遮罩測試

`q.mean(axis=-1, keepdims=True)` 預期 shape 為 `(2,3,1)`。減去此平均後，每個 `(b,t)` 位置沿 feature 軸的平均預期接近零。

遮罩損失的有效值為 $2,4,3,5,7$，其平均預期為 $4.2$。被遮罩的 `100` 或 NaN 不進入結果；若 NaN 位於有效位置，函式預期丟出 `ValueError`。

### 5. 故障測試

程式預期明確拒絕：

- $D=5,H=2$，因為 $D/H$ 不是整數；
- 有效 token 數為零；
- 有效位置含 NaN 或無限值；
- 機率含負值；
- 偏置 shape 為 `(3,1)` 而非 `(Dout,)`。

故障測試的目的不是讓錯誤運算繼續，而是確認程式在契約被破壞時停止。由函式入口丟出語義清楚的例外，通常比依賴底層廣播錯誤更容易除錯。

---

## 反例與常見陷阱

### 1. 用 reshape 冒充 transpose

錯誤：

```python
heads = x.reshape(B, H, T, dh)
```

若原始資料是 `(B,T,D)` 且 $D=Hd_h$，這通常不等於正確的 `(B,H,T,dh)`。正確方式是：

```python
heads = x.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
```

判斷標準不是 shape 是否相同，而是每個索引是否仍代表同一個 token、head 與 feature。

### 2. NumPy 一維陣列的 `.T` 不改 shape

```python
v = np.array([1.0, 2.0, 3.0])
assert v.shape == (3,)
assert v.T.shape == (3,)
```

一維陣列沒有可交換的兩個軸。若需要 row matrix 或 column matrix，應明確建立：

```python
row = v[None, :]   # (1,3)
col = v[:, None]   # (3,1)
```

`row` 與 `col` 都是二階矩陣，但 shape 不同。

### 3. 合法但錯誤的廣播

若 $T=D=4$：

```python
x = np.zeros((2, 4, 4))
p = np.arange(4)
y = x + p
```

運算合法，但 `p` 被當成 feature 偏置。若 `p` 表示位置值，正確寫法是：

```python
y = x + p[None, :, None]
```

此時 `p` 的 shape 為 `(1,T,1)`，沿 batch 與 feature 軸廣播。

### 4. 在錯誤軸做機率歸一化

分類機率若 shape 為 `(B,T,C)`，類別軸是 $C$，所以歸一化應沿最後一軸。沿 `axis=0` 歸一化會讓不同樣本互相競爭；沿 `axis=1` 則讓不同時間位置互相競爭。即使所得數值都在 $[0,1]$，也不是所需的條件分布。

### 5. 不明確的 `mean()`

```python
loss = token_losses.mean()
```

只有在每個位置都有效，且確實要讓全部位置等權時才正確。有 padding 時，必須使用遮罩並明示分母。不同長度序列若先各自平均，再做 batch 平均，會使短序列與長序列等權；這與有效 token 平均不同。

### 6. 遺失 $B=1$ 軸

不一致的做法：

```python
x = sample              # (T,D)
```

較穩定的做法：

```python
x = sample[None, :, :]  # (1,T,D)
```

批次軸保留後，同一模型函式不需要為單樣本另寫分支。

### 7. 以 contiguous 與否判定數值正確性

非連續布局不是數學錯誤。transpose 常回傳共享底層資料的 view，索引仍可正確。真正的風險包括：

- 將 reshape 後是否共享記憶體當成固定保證；
- 把底層記憶體順序誤認為軸語義；
- 把外部函式要求 contiguous 的介面限制誤說成數學限制。

若某介面明確要求連續布局，可使用 `np.ascontiguousarray`，但不應為了「看起來安全」而到處無條件複製。

### 8. 切窗後再隨機分割

同一長序列的相鄰窗口高度重疊。若先切窗，再把窗口隨機分到 train 與 test，測試窗口可能與訓練窗口共享大部分 token，造成資料洩漏。shape 完全正確也無法發現這種問題。

正確順序是：

1. 按原始文件、群組、池槽或時間切分；
2. 只用 train 擬合詞表與標準化統計；
3. 分別在 train、validation、test 內建立窗口；
4. 不用 test 選模型 shape 或超參數。

---

## AI、幾何與養殖案例

考慮完全合成的養殖感測序列。每筆樣本代表一個模擬池槽的一段時間窗口：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

假設：

- $B=8$：八個合成窗口；
- $T=24$：每個窗口有二十四個時間點；
- $D=5$：每個時間點有五個特徵。

這五個特徵可以抽象表示合成水溫、合成溶氧、合成 pH、合成濁度及缺失旗標。它們只是教材資料，不是真實現場安全閾值。

逐特徵標準化所需統計量應為

$$
\mu,\sigma\in\mathbb{R}^{1\times1\times D}.
$$

其中 $\mu$ 與 $\sigma$ 只能由訓練集估計。標準化為

$$
Z_{btd}=\frac{X_{btd}-\mu_d}{\sigma_d}.
$$

若某特徵的訓練標準差為零，不能直接相除；應依預處理契約拒絕、移除該特徵，或明確使用指定穩定化規則。本章不以任意 epsilon 隱藏資料問題。

`(1,1,D)` 的統計量可沿 batch 與 time 軸廣播。若誤算成 `(T,1)`，可能把每個時間位置的統計量錯套到所有特徵，或在某些巧合 shape 下靜默產生錯誤。

若要加入每個時間位置的合成週期訊號，而且每個位置只有一個純量，應令

$$
P\in\mathbb{R}^{1\times T\times1}.
$$

則

$$
Y=Z+P
$$

的 shape 仍是 $(B,T,D)$。不能只把 $P$ 寫成 `(T,)`，因為廣播會優先把它對齊最後一軸。

資料切分可先依合成池槽 ID 分組，例如：

- 訓練池槽：ID 0–5；
- 驗證池槽：ID 6；
- 測試池槽：ID 7。

之後才在各池槽內建立長度 $T$ 的窗口。這能避免同一池槽的重疊窗口同時出現在訓練與測試集合。若資料具有時間方向，也可採較早時間訓練、較晚時間驗證與測試，但必須先明定目標，不能查看測試結果後再改切分方式。

從幾何角度看，每個 token 的特徵向量位於 $\mathbb{R}^D$。batch 與 time 軸是在收集多個向量，不應與向量內部的 feature 軸混合。線性層

$$
W\in\mathbb{R}^{D\times D'}
$$

對每個 token 使用相同幾何映射：

$$
Y_{bt:}=X_{bt:}W+b.
$$

共享權重由 `X @ W` 的前導軸語義表達。若把 time 軸誤併入 feature 軸，運算就不再是「逐 token 使用同一映射」，而成為另一個模型。

這個案例只展示 shape、廣播與資料切分，不代表模型能替代養殖專業判斷，更不授權控制投餌、曝氣、加藥、泵浦或其他外部設備。

---

## 習題

### 一、手算題

令

$$
X=
\begin{bmatrix}
1&0\\
2&-1\\
3&1
\end{bmatrix},
\quad
W=
\begin{bmatrix}
2&1\\
-1&3
\end{bmatrix},
\quad
b=[1,-2].
$$

1. 寫出所有張量的 shape。
2. 計算 $Y=XW+b$。
3. 計算 `Y.mean(axis=0)`。
4. 計算全部元素的平均，並寫明 reduction 軸。

### 二、程式題

撰寫函式 `time_bias(x, p)`：

- `x` shape 為 `(B,T,D)`；
- `p` shape 必須恰為 `(T,)`；
- 輸出滿足 $y_{btd}=x_{btd}+p_t$；
- 必須保留 $B=1$；
- shape 不符時丟出 `ValueError`。

另寫至少一個正常、一個邊界及一個故障測試。

### 三、反例題

設 $B=2,T=D=3$，`x.shape == (2,3,3)`，`p.shape == (3,)`。某程式直接計算 `x + p`，並宣稱加入了位置偏置。

1. 為何這段程式不一定報錯？
2. 它實際沿哪一軸加入 `p`？
3. 請給出正確寫法。
4. 為何只檢查輸出 shape 無法找出此錯誤？

### 四、整合題

一批逐 token 損失 `losses` 的 shape 為 `(B,T)`，`valid` 是同 shape 的布林遮罩。

1. 寫出有效 token 平均公式。
2. 說明遮罩全為 `False` 時的策略。
3. 說明有效位置與無效位置出現 NaN 時的契約。
4. 若每條原始序列會產生多個重疊窗口，說明 train/validation/test 的正確切分順序。
5. 寫出 NumPy 函式，拒絕 shape 不同、非布林遮罩、空有效集合及有效位置的非有限損失。

---

## 習題解答

### 一、手算題解答

shape 為

$$
X:(3,2),\quad W:(2,2),\quad b:(2),\quad Y:(3,2).
$$

逐列計算：

$$
[1,0]W=[2,1],
$$

$$
[2,-1]W
=
[2\cdot2+(-1)(-1),\;2\cdot1+(-1)\cdot3]
=[5,-1],
$$

$$
[3,1]W
=
[3\cdot2+1\cdot(-1),\;3\cdot1+1\cdot3]
=[5,6].
$$

加入偏置：

$$
Y=
\begin{bmatrix}
3&-1\\
6&-3\\
6&4
\end{bmatrix}.
$$

沿 batch 軸，即 `axis=0` 平均：

$$
Y.\operatorname{mean}(\text{axis}=0)
=
\left[
\frac{3+6+6}{3},
\frac{-1-3+4}{3}
\right]
=[5,0].
$$

全部元素平均是沿 `axis=(0,1)` 縮減：

$$
\frac{3-1+6-3+6+4}{6}
=
\frac{15}{6}
=2.5.
$$

### 二、程式題解答

```python
import numpy as np


def time_bias(x, p):
    if x.ndim != 3:
        raise ValueError("x 必須是 (B,T,D)")
    if p.ndim != 1:
        raise ValueError("p 必須是 (T,)")

    B, T, D = x.shape
    if p.shape != (T,):
        raise ValueError("p 的長度必須等於 T")

    y = x + p[None, :, None]
    assert y.shape == (B, T, D)
    return y


# 正常測試
x = np.zeros((2, 3, 4))
p = np.array([10.0, 20.0, 30.0])
y = time_bias(x, p)
assert y.shape == (2, 3, 4)
np.testing.assert_array_equal(y[0, :, 0], p)
np.testing.assert_array_equal(y[1, :, 3], p)

# 邊界測試：B=1
one = time_bias(np.zeros((1, 3, 2)), p)
assert one.shape == (1, 3, 2)

# 故障測試
caught = False
try:
    time_bias(np.zeros((2, 3, 4)), np.zeros(4))
except ValueError:
    caught = True

assert caught
```

`p[None,:,None]` 的 shape 為 `(1,T,1)`，因此沿 batch 與 feature 軸廣播。這段程式未在本章寫作流程中執行；以上為預期行為。

### 三、反例題解答

`p.shape == (3,)` 會從右側對齊 `x.shape == (2,3,3)`。最後一軸大小也是 $3$，所以廣播合法，不一定報錯。

它實際執行的是

$$
y_{btd}=x_{btd}+p_d,
$$

也就是沿 feature 軸加入 `p`。

若 `p_t` 是位置偏置，正確寫法是：

```python
y = x + p[None, :, None]
```

此時

$$
y_{btd}=x_{btd}+p_t.
$$

錯誤與正確結果的 shape 都是 `(2,3,3)`，所以只檢查輸出 shape 無法發現問題。測試應使用非對稱數值並檢查指定索引，例如比較同一時間、不同 feature 是否增加相同的 $p_t$。

### 四、整合題解答

有效 token 平均為

$$
L=
\frac{
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}m_{bt}\ell_{bt}
}{
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}m_{bt}
}.
$$

若遮罩全為 `False`，分母為零，函式應丟出明確例外，不應回傳 NaN、無限值或假造的零損失。

本解答採用以下非有限值契約：

- 有效位置必須是有限 loss，否則拒絕；
- 無效位置完全不參與分子與分母，因此允許保存 NaN 作為「未定義」記號；
- 若系統希望所有輸入都有限，也可採更嚴格契約，但程式與文件必須一致。

資料切分應先在原始序列、文件、池槽或群組層級完成，再於各集合內切出窗口。詞表與標準化統計只從訓練集擬合；驗證集用於模型選擇，測試集不參與調參。

```python
import numpy as np


def valid_token_loss(losses, valid):
    if losses.ndim != 2:
        raise ValueError("losses 必須是 (B,T)")
    if valid.shape != losses.shape:
        raise ValueError("valid 與 losses shape 必須相同")
    if valid.dtype != np.bool_:
        raise TypeError("valid 必須是 bool")

    n = int(valid.sum())
    if n == 0:
        raise ValueError("沒有有效 token")

    selected = losses[valid]
    if not np.isfinite(selected).all():
        raise ValueError("有效位置的 losses 必須全部有限")

    return selected.sum(dtype=np.float64) / n
```

這裡先選取有效位置，再求一次總和，最後只除以有效 token 總數一次。

---

## 本章小結

張量 shape 不只是尺寸列表，也是一份軸語義契約。`(B,T,D)` 中的 batch、time 與 feature 不能任意交換，即使元素總數完全相同。

本章的核心原則如下：

1. 每個張量都應明列 shape 與軸語義。
2. `reshape` 重新分組元素，`transpose` 重新排列軸，兩者不可互換。
3. `matmul` 對最後兩軸做矩陣乘法，前導軸按廣播規則配對。
4. `einsum` 可直接表達索引與縮減軸，但仍需 shape 與語義檢查。
5. 廣播從右對齊；運算合法不代表語義正確。
6. 廣播反向梯度必須沿新增軸與原大小為 $1$ 的擴張軸求和，還原成參數原 shape。
7. reduction 必須明列軸、`sum` 或 `mean`，以及是否 `keepdims`。
8. $B=1$ 時仍保留 batch 軸，避免單樣本形成特殊介面。
9. transpose 後可能是非連續布局；非連續不等於數值錯誤。
10. 機率張量須在指定類別軸上非負、有限且加總為一。
11. 有遮罩的 loss 應按有效 token 總數平均，且空有效集合必須拒絕。
12. 資料切分先於切窗及統計量擬合；shape 正確不能防止資料洩漏。

後續的批次線性層、正規化、注意力與 Transformer 都建立在這些契約上。越早把軸與 reduction 寫清楚，越少需要從一個「能執行但答案錯誤」的結果中逆向除錯。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, 2017。  
   https://arxiv.org/abs/1706.03762  
   本章只將其作為後續注意力張量記號的背景入口。依既有來源註記，只取得摘要頁，未據此宣稱已完整核對論文內容。

2. NumPy, *Broadcasting* 使用指南。  
   https://numpy.org/doc/stable/user/basics.broadcasting.html  
   此來源仍標記為待逐條核對的延伸入口；本章不宣稱已在寫作流程中連網查證其全部內容。

3. Dive into Deep Learning。  
   https://d2l.ai/  
   可作為張量與深度學習實作的延伸入口；本章未依賴其中未核對章節建立證明。

4. PyTorch, *Reproducibility*。  
   https://docs.pytorch.org/docs/stable/notes/randomness.html  
   此來源僅列為後續可重現性議題的延伸入口。本章未核對本機框架版本，也不宣稱跨版本、跨平台逐位重現。