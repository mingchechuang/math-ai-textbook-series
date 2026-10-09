# 第02章 張量形狀、批次與廣播

## 學習目標與先備知識

神經網路中的多數錯誤，不是公式本身錯誤，而是公式與程式對「哪一軸代表什麼」理解不同。兩個陣列即使元素總數相同，也不表示它們代表相同資料；一個能夠成功執行的廣播運算，也不表示它在語義上正確。

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
7. 保留 $B=1$ 的批次軸，避免單樣本與多樣本程式產生不同介面。
8. 理解 transpose 後的陣列可能是非連續布局；非連續不表示數值錯誤，但會影響 reshape 是否複製資料及某些底層操作。
9. 在 loss、機率與資料切分中保留軸與平均方式的完整契約。

本章假設讀者熟悉純量、向量、矩陣及矩陣乘法。本文以 NumPy 為實作工具，只要求 CPU，不下載模型、語料或其他資料。以下程式未在本寫作流程中執行；所有輸出均標為**預期結果**，不構成實測紀錄。

---

## 問題與直覺

考慮一批序列資料：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

若 $X$ 的 shape 是 `(2, 3, 4)`，這串數字本身不夠完整。我們還必須說：

- 第 $0$ 軸是 batch，大小為 $B=2$；
- 第 $1$ 軸是 time，大小為 $T=3$；
- 第 $2$ 軸是 feature，大小為 $D=4$。

如果把它轉置成 `(3, 2, 4)`，元素沒有增加或減少，但語義已變成 `(T,B,D)`。若後續函式仍假定輸入是 `(B,T,D)`，程式有時會立即報錯，有時卻會悄悄算出錯誤結果。

形狀錯誤可分成三類：

1. **立即失敗**：內積維度不相容，NumPy 丟出例外。
2. **形狀正確但語義錯誤**：例如錯把 time 軸當 batch 軸做平均。
3. **被廣播掩蓋的錯誤**：shape 可以對齊，程式正常執行，但參數沿錯誤軸重複。

因此，可靠的張量程式至少需要三層契約：

- **shape 契約**：每一軸的大小。
- **語義契約**：每一軸代表什麼。
- **reduction 契約**：在哪些軸做 `sum` 或 `mean`，是否保留維度。

可以把 shape 視為型別的一部分。`(B,T,D)` 與 `(T,B,D)` 雖然都含三軸，卻不應被視為同一型別。

---

## 定義、定理與推導

### 1. 張量、索引與 shape

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
| $B$ | batch size | 第 0 軸 |
| $T$ | sequence length | 第 1 軸 |
| $D$ | model feature | 最後一軸 |
| $H$ | attention heads | 拆頭後第 1 軸 |
| $d_h$ | 每頭維度，$D/H$ | 拆頭後最後一軸 |

批次線性層使用

$$
X\in\mathbb{R}^{B\times D_{\text{in}}},
\quad
W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}},
\quad
b\in\mathbb{R}^{D_{\text{out}}},
$$

以及

$$
Y=XW+b\in\mathbb{R}^{B\times D_{\text{out}}}.
$$

batch 中每筆樣本以一個橫列儲存。這只是資料布局約定，與微分教材常用的 column 向量表示不矛盾。

對序列輸入，線性層逐 token 作用：

$$
X\in\mathbb{R}^{B\times T\times D_{\text{in}}},
\quad
Y_{bto}=\sum_{i=0}^{D_{\text{in}}-1}X_{bti}W_{io}+b_o.
$$

輸出 shape 為 $(B,T,D_{\text{out}})$。

### 2. reshape 不等於 transpose

`reshape` 改變索引分組方式，但依照目前元素遍歷順序重組資料；`transpose` 則重新排列軸。

令

$$
X=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix},
\qquad X\in\mathbb{R}^{2\times3}.
$$

則：

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

對多頭張量尤其如此。從 $(B,T,D)$ 拆成 $H$ 個頭，通常先：

$$
(B,T,D)\xrightarrow{\text{reshape}}(B,T,H,d_h),
$$

再：

$$
(B,T,H,d_h)\xrightarrow{\text{transpose}}(B,H,T,d_h).
$$

只做 reshape 成 `(B,H,T,dh)` 會把原本相鄰的元素錯分到 head 與 time 軸。

### 3. 矩陣乘法與批次矩陣乘法

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
\quad
W\in\mathbb{R}^{K\times N}
$$

得到

$$
A@W\in\mathbb{R}^{B\times M\times N}.
$$

這正好可表示同一個權重矩陣作用於 batch 中每筆資料。

### 4. einsum：把索引寫在程式中

序列線性層可以寫成：

```python
Y = np.einsum("bti,io->bto", X, W) + b
```

字串 `"bti,io->bto"` 表示：

- `X` 軸為 batch、time、input feature；
- `W` 軸為 input feature、output feature；
- 重複但未出現在輸出的索引 `i` 被求和；
- 輸出保留 `b,t,o`。

`einsum` 的字母只是局部標記，不會自動驗證 `b` 真的是 batch；它提升可讀性，卻不能取代 shape 斷言與語義文件。

### 5. 廣播規則

比較兩個 shape 時，從最右邊的軸開始對齊。每一對軸若符合以下任一條件，即可廣播：

1. 大小相等；
2. 其中一邊大小為 $1$；
3. 較短 shape 在該位置沒有軸，可視為補上大小 $1$ 的軸。

例如：

$$
X:(B,T,D),\qquad b:(D)
$$

從右對齊後相當於：

$$
X:(B,T,D),\qquad b:(1,1,D),
$$

所以 `X + b` 合法，且 $b_d$ 會沿 batch 與 time 重複使用。

但合法不表示符合目的。假設想為每個 time step 加偏置：

$$
p\in\mathbb{R}^{T}.
$$

直接計算 `X + p` 會把 `p` 對齊最後一軸。只有在 $T=D$ 時它才碰巧合法，且實際上是沿 feature 軸相加。正確 shape 應寫成：

$$
p\in\mathbb{R}^{1\times T\times1},
$$

即 `p[None, :, None]`。

### 6. 小命題：廣播後求和可還原重複次數

**命題。** 設 $b\in\mathbb{R}^{D}$，把它廣播成 $\widetilde b\in\mathbb{R}^{B\times T\times D}$，定義

$$
\widetilde b_{btd}=b_d.
$$

若沿 batch 與 time 軸求和，則

$$
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde b_{btd}=BT\,b_d.
$$

若改取平均，則

$$
\frac{1}{BT}\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde b_{btd}=b_d.
$$

**證明。**

固定任意特徵索引 $d$。依廣播定義，對每個 $b,t$ 都有 $\widetilde b_{btd}=b_d$。因此

$$
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}\widetilde b_{btd}
=
\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}b_d.
$$

內層共有 $T$ 個相同的 $b_d$，外層共有 $B$ 組，所以總共有 $BT$ 項：

$$
\sum_b\sum_t b_d=BT\,b_d.
$$

兩邊除以 $BT$ 即得平均式。證畢。

這個命題也說明廣播的反向傳播規則：若參數原 shape 為 $(D,)$，前向時被重複到 $(B,T,D)$，則其梯度必須沿新增或大小為 $1$ 的軸求和，還原成 $(D,)$。不能直接保留 $(B,T,D)$。

### 7. reduction 必須寫清楚軸

對 $X\in\mathbb{R}^{B\times T\times D}$：

- `X.sum(axis=1)`：沿 time 求和，shape 為 $(B,D)$。
- `X.mean(axis=(0,1))`：沿 batch 與 time 平均，shape 為 $(D,)$。
- `X.mean(axis=-1, keepdims=True)`：沿 feature 平均，shape 為 $(B,T,1)$。
- `X.mean()`：沿全部軸平均，得到純量。

`keepdims=True` 會保留被 reduction 的軸，大小設為 $1$。這對後續廣播特別重要。例如逐 token 中心化：

$$
\mu_{bt}=\frac1D\sum_dX_{btd},
\qquad
\mu\in\mathbb{R}^{B\times T\times1}.
$$

程式應寫：

```python
mu = X.mean(axis=-1, keepdims=True)
centered = X - mu
```

### 8. loss 平均、機率域與資料切分契約

若每筆樣本損失為 $\ell_b$，batch 平均損失明定為

$$
L=\frac1B\sum_{b=0}^{B-1}\ell_b.
$$

若序列含有效位置遮罩 $m_{bt}\in\{0,1\}$，則有效 token 平均為

$$
L=
\frac{\sum_{b,t}m_{bt}\ell_{bt}}
{\sum_{b,t}m_{bt}},
$$

前提是 $\sum_{b,t}m_{bt}>0$；分母為零時應拒絕，而不是回傳 NaN 或任意零。不能先對每筆序列平均，再對 batch 等權平均，除非每筆序列的有效 token 數相同，或這正是明定的評估目標。平均只除一次；若 $\ell_{bt}$ 已是逐 token 損失，就以有效 token 總數作唯一分母。

若張量表示離散機率 $P\in\mathbb{R}^{B\times T\times C}$，類別軸為最後一軸，必須滿足

$$
P_{btc}\ge0,\qquad
\sum_{c=0}^{C-1}P_{btc}=1.
$$

歸一化是沿 `axis=-1`，不是沿 batch 或 time。零機率可以是合法分布的一部分，但若後續計算 $\log P_{btc}$，則 $\log0=-\infty$，必須依損失定義處理，不能任意加 epsilon 宣稱等價。

資料 shape 也不能取代資料切分。假設同一條原始序列產生多個重疊窗口，即使每個窗口都有獨立 batch 列，也不代表它們獨立。應先按文件、個體、群組或時間把原始資料切成訓練／驗證／測試集合，再於各集合內建立窗口。標準化統計、詞表及其他可擬合轉換只使用訓練集；驗證集用於選擇設定，測試集保留到最終評估，不能參與調參。

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
b=
\begin{bmatrix}
1&0&-2
\end{bmatrix}.
$$

shape 分別是：

$$
X:(B,D_{\text{in}})=(2,2),
$$

$$
W:(D_{\text{in}},D_{\text{out}})=(2,3),
$$

$$
b:(D_{\text{out}})=(3).
$$

第一步，計算第一筆樣本：

$$
[1,2]W
=
[
1\cdot1+2\cdot0,\;
1\cdot(-1)+2\cdot3,\;
1\cdot2+2\cdot1
]
=
[1,5,4].
$$

第二步，計算第二筆樣本：

$$
[3,4]W
=
[
3\cdot1+4\cdot0,\;
3\cdot(-1)+4\cdot3,\;
3\cdot2+4\cdot1
]
=
[3,9,10].
$$

所以

$$
XW=
\begin{bmatrix}
1&5&4\\
3&9&10
\end{bmatrix}.
$$

第三步，$b:(3,)$ 從右對齊輸出 `(2,3)`，沿 batch 軸廣播：

$$
Y=XW+b
=
\begin{bmatrix}
1+1&5+0&4-2\\
3+1&9+0&10-2
\end{bmatrix}
=
\begin{bmatrix}
2&5&2\\
4&9&8
\end{bmatrix}.
$$

若定義純量損失為全部輸出元素的平均：

$$
L=\frac1{BD_{\text{out}}}\sum_{b,o}Y_{bo},
$$

則

$$
L=\frac{2+5+2+4+9+8}{2\cdot3}=5.
$$

這裡分母是 $6$，不是先除以 batch 再錯誤地重複除以輸出維度。

### 例題二：reshape、transpose 與拆頭

令 $B=1,T=2,H=2,d_h=2$，因此 $D=Hd_h=4$。輸入為

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

以 `(B,T,H,dh)` 表示：

- time 0：
  - head 0：$[1,2]$
  - head 1：$[3,4]$
- time 1：
  - head 0：$[5,6]$
  - head 1：$[7,8]$

第二步，把 head 軸移到 time 軸之前：

$$
X_h=\operatorname{transpose}(X_r,(0,2,1,3)).
$$

結果 shape 為 $(1,2,2,2)=(B,H,T,d_h)$，但軸語義已改變：

- head 0：
  - time 0：$[1,2]$
  - time 1：$[5,6]$
- head 1：
  - time 0：$[3,4]$
  - time 1：$[7,8]$

如果直接執行 `X.reshape(1, 2, 2, 2)` 並宣稱結果已是 `(B,H,T,dh)`，則會被解讀為：

- head 0：$[1,2]$、$[3,4]$
- head 1：$[5,6]$、$[7,8]$

這把同一 time 的不同 head 誤當成同一 head 的不同 time。shape 完全相同，語義卻錯誤。

### 例題三：遮罩損失的正確平均

兩筆序列的逐 token 損失及有效遮罩為

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

所以

$$
L=\frac{21}{5}=4.2.
$$

被遮罩位置的 $100$ 不應進入分子或分母。若先算每列平均再對兩列平均：

$$
\frac12\left(\frac{2+4}{2}+\frac{3+5+7}{3}\right)
=\frac12(3+5)=4,
$$

所得 $4$ 與 token 加權平均 $4.2$ 不同。兩者回答的是不同問題，不能混稱為同一種 batch loss。

---

## 實作與程式

以下是自足的 NumPy CPU 程式，示範線性層、`einsum`、拆頭、合頭、reduction、有效 token 平均，以及預期失敗測試。程式不使用網路、GPU、外部模型或資料。

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
    if not isinstance(H, int) or H <= 0:
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
    """losses:(B,T), valid:(B,T) bool -> scalar"""
    if losses.ndim != 2:
        raise ValueError("losses 必須是 (B,T)")
    if valid.shape != losses.shape:
        raise ValueError("valid 必須與 losses 同 shape")
    if valid.dtype != np.bool_:
        raise TypeError("valid 必須是布林陣列")

    count = int(valid.sum())
    if count == 0:
        raise ValueError("至少必須有一個有效 token")

    total = losses[valid].sum(dtype=np.float64)
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

    # transpose 通常產生非 C-contiguous view。
    q = np.arange(24).reshape(2, 3, 4)
    qt = q.transpose(0, 2, 1)
    assert qt.shape == (2, 4, 3)
    assert not qt.flags["C_CONTIGUOUS"]
    qc = np.ascontiguousarray(qt)
    assert qc.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(qc, qt)

    # reduction：沿 feature 平均並保留軸。
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

    # 機率沿最後類別軸正規化。
    p = np.array([[[0.25, 0.75],
                   [1.00, 0.00]]])
    require_probabilities(p)

    # 故障案例：D 不能被 H 整除。
    expect_error(
        lambda: split_heads(np.zeros((1, 2, 5)), H=2),
        ValueError
    )

    # 故障案例：完全沒有有效 token。
    expect_error(
        lambda: masked_token_mean(
            np.ones((1, 2)),
            np.zeros((1, 2), dtype=bool)
        ),
        ValueError
    )

    # 故障案例：負機率。
    expect_error(
        lambda: require_probabilities(np.array([[1.1, -0.1]])),
        ValueError
    )

    # 故障案例：錯誤偏置 shape，避免意外廣播。
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

這份程式刻意不接受 `(1,Dout)` 偏置，即使它在某些情況下可能可以廣播。嚴格限制為 `(Dout,)` 能縮小介面，避免呼叫端誤把其他軸的參數當成 feature 偏置。

---

## 測試與預期結果

上述程式未由本章寫作流程執行。若使用相容的 NumPy 環境在 CPU 執行，預期如下。

### 1. 正常測試

`linear_sequence` 同時計算：

```python
x @ w + bias
```

與：

```python
np.einsum("bti,io->bto", x, w) + bias
```

兩者預期逐元素相同，輸出 shape 為 `(2,2,2)`。

### 2. 邊界測試：$B=1$

輸入明確建成 `(1,2,3)`，而不是 `(2,3)`。輸出預期為 `(1,2,2)`。這保證同一函式對 $B=1$ 與 $B>1$ 有一致介面。

不要用無參數的 `squeeze()` 處理 batch，因為它會移除所有大小為 $1$ 的軸。若 $B=1$、$T=1$，可能同時失去 batch 與 time 軸。只有在介面明確要求時，才可使用如 `squeeze(axis=指定軸)`。

### 3. 非連續布局測試

`q.transpose(0,2,1)` 預期不是 C-contiguous。這不影響索引數值：

```python
np.testing.assert_array_equal(qc, qt)
```

仍應成立。`np.ascontiguousarray` 會在需要時建立連續副本；不能假設所有 transpose 都複製資料，也不能假設所有 reshape 都不複製資料。是否共享記憶體屬於布局問題，應與張量數值及軸語義分開檢查。

### 4. reduction 測試

`q.mean(axis=-1, keepdims=True)` 的 shape 預期為 `(2,3,1)`。減去此平均後，每個 `(b,t)` 位置沿 feature 軸的平均預期接近零。

### 5. 故障測試

程式預期明確拒絕：

- $D=5,H=2$，因為 $D/H$ 不是整數；
- 有效 token 數為零；
- 含負值的機率；
- shape 為 `(3,1)` 的錯誤偏置。

故障測試的目的不是讓錯誤運算繼續，而是確認程式在契約被破壞時停止。若 NumPy 自己先丟出難懂的廣播錯誤，雖然也停止了，但不如函式入口的語義檢查清楚。

---

## 反例與常見陷阱

### 1. `reshape` 冒充 `transpose`

錯誤：

```python
heads = x.reshape(B, H, T, dh)
```

若原始資料是 `(B,T,D)` 且 $D=H d_h$，這通常不等於正確的 `(B,H,T,dh)`。正確方式是：

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

### 3. 合法但錯誤的廣播

若 $T=D=4$：

```python
x = np.zeros((2, 4, 4))
p = np.arange(4)
y = x + p
```

運算合法，但 `p` 被當成 feature 偏置。若 `p` 是位置值，正確寫法是：

```python
y = x + p[None, :, None]
```

此時 `p` 的 shape 是 `(1,T,1)`，沿 batch 與 feature 軸廣播。

### 4. 對錯誤軸做 softmax 或機率正規化

分類機率若 shape 為 `(B,T,C)`，類別軸是 `C`，所以歸一化應沿最後一軸。沿 `axis=0` 正規化會讓不同樣本互相競爭；沿 `axis=1` 則讓不同時間位置互相競爭。即使輸出數值都落在 $[0,1]$，也不是所需的條件分布。

### 5. 不明確的 `mean()`

```python
loss = token_losses.mean()
```

只有在每個位置都有效且確實要對全部位置等權平均時才正確。有 padding 時，必須使用遮罩並明示分母。不同長度序列若先各自平均再做 batch 平均，會使短序列與長序列權重相同；這與有效 token 平均不同。

### 6. 遺失 $B=1$ 軸

錯誤做法：

```python
x = sample          # (T,D)
```

較一致的做法：

```python
x = sample[None, :, :]   # (1,T,D)
```

批次軸保留後，同一個模型函式不需要針對單樣本另寫分支。

### 7. 以 contiguous 與否判定數值正確性

非連續布局不是錯誤。transpose 常回傳共享底層資料的 view，索引依然正確。真正的風險是：

- 將 reshape 後是否共享記憶體當作固定保證；
- 把底層記憶體順序誤認為軸語義；
- 把外部函式要求 contiguous 的介面限制，誤說成數學限制。

### 8. 切窗後再隨機分割

同一長序列的相鄰窗口高度重疊。若先切窗再隨機分到 train 與 test，測試窗口可能與訓練窗口共享大部分 token，造成洩漏。正確順序是：

1. 按原始文件、群組或時間切分；
2. 只用 train 擬合詞表與標準化統計；
3. 分別在 train、validation、test 內建立窗口；
4. 不用 test 選 shape、模型或超參數。

---

## AI、幾何與養殖案例

考慮完全合成的養殖感測序列。每筆樣本代表一個模擬池槽的一段時間窗口：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

假設：

- $B=8$：八個訓練窗口；
- $T=24$：二十四個合成時間點；
- $D=5$：五個合成特徵。

這些特徵可抽象表示水溫、溶氧、pH、濁度及一個缺失旗標，但所有數值都只是教材生成資料，不是現場安全閾值。

逐特徵標準化所需統計量應為：

$$
\mu,\sigma\in\mathbb{R}^{1\times1\times D}.
$$

其中 $\mu$ 與 $\sigma$ 只能由訓練集估計。標準化為

$$
Z_{btd}=\frac{X_{btd}-\mu_{11d}}{\sigma_{11d}},
$$

實際程式中索引以零起始，但 shape 邏輯相同。`(1,1,D)` 可沿 batch 與 time 廣播。若誤算成 `(T,1)`，可能把每個時間位置的統計量套到所有特徵，或在某些巧合 shape 下靜默出錯。

若要加入每個時間位置的合成週期特徵：

$$
P\in\mathbb{R}^{1\times T\times D},
$$

則 `Z + P` 沿 batch 軸廣播。若位置訊號只有一個純量通道，應寫成 `(1,T,1)`，再依設計廣播到 feature 軸；不能只寫 `(T,)`，否則它會優先對齊最後一軸。

資料切分應先以模擬池槽 ID 分組，例如：

- 訓練池槽：ID 0–5；
- 驗證池槽：ID 6；
- 測試池槽：ID 7。

之後才在各池槽內建立長度 $T$ 的窗口。這能避免同一池槽的重疊窗口同時出現在訓練與測試集合。這個例子只展示 shape 與切分契約，不代表模型能替代養殖專業判斷，更不授權控制投餌、曝氣、加藥或其他設備。

從幾何角度看，每個 token 的特徵向量位於 $\mathbb{R}^D$。batch 與 time 軸是在收集多個向量，不應與向量內部的 feature 軸混合。線性層 $W\in\mathbb{R}^{D\times D'}$ 對每個 token 使用相同幾何映射；共享權重由 `X @ W` 的前導軸語義表達。若把 time 軸誤併入 feature 軸，模型就不再是「逐 token 使用同一映射」，而是另一個完全不同的運算。

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

1. 寫出所有 shape。
2. 計算 $Y=XW+b$。
3. 計算 `Y.mean(axis=0)` 與全部元素平均。

### 二、程式題

撰寫函式 `time_bias(x, p)`：

- `x` shape 為 `(B,T,D)`；
- `p` shape 必須恰為 `(T,)`；
- 輸出應滿足 $y_{btd}=x_{btd}+p_t$；
- 必須保留 $B=1$；
- shape 不符時丟出 `ValueError`。

另寫至少一個正常、一个邊界及一個故障測試。

### 三、反例題

設 $B=2,T=D=3$，`x.shape == (2,3,3)`，`p.shape == (3,)`。某程式直接計算 `x + p`，宣稱加入了位置偏置。

1. 為何這段程式不一定報錯？
2. 它實際沿哪一軸加入 `p`？
3. 請給出正確寫法。
4. 為何只檢查輸出 shape 無法找出此錯誤？

### 四、整合題

一批逐 token 損失 `losses` 的 shape 為 `(B,T)`，`valid` 是同 shape 的布林遮罩。

1. 寫出有效 token 平均公式。
2. 說明全為 `False` 時的策略。
3. 若每條原始序列會產生多個重疊窗口，說明 train/validation/test 的正確切分順序。
4. 寫出 NumPy 函式，拒絕 shape 不同、非布林遮罩及空有效集合。

---

## 習題解答

### 一、手算題解答

shape 為：

$$
X:(3,2),\quad W:(2,2),\quad b:(2),\quad Y:(3,2).
$$

逐列計算：

$$
[1,0]W=[2,1],
$$

$$
[2,-1]W=[2\cdot2+(-1)(-1),\;2\cdot1+(-1)3]=[5,-1],
$$

$$
[3,1]W=[3\cdot2+1(-1),\;3\cdot1+1\cdot3]=[5,6].
$$

加入偏置 $b=[1,-2]$：

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

全部元素平均為：

$$
\frac{3-1+6-3+6+4}{6}
=\frac{15}{6}=2.5.
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
try:
    time_bias(np.zeros((2, 3, 4)), np.zeros(4))
except ValueError:
    caught = True
else:
    caught = False

assert caught
```

預期 `p[None,:,None]` 的 shape 為 `(1,T,1)`，因此沿 batch 與 feature 軸廣播。以上程式未在寫作流程中執行。

### 三、反例題解答

1. 因為 `p.shape == (3,)` 會從右側對齊 `x.shape == (2,3,3)`。最後一軸大小也是 $3$，所以廣播合法。
2. 它實際沿最後的 feature 軸加入，即：

   $$
   y_{btd}=x_{btd}+p_d.
   $$

3. 若 `p_t` 是位置偏置，正確寫法是：

   ```python
   y = x + p[None, :, None]
   ```

   此時：

   $$
   y_{btd}=x_{btd}+p_t.
   $$

4. 錯誤與正確結果的 shape 都是 `(2,3,3)`。shape 只說明軸大小，不能單獨證明軸語義正確。應以非對稱測試值檢查指定索引，並在介面文件中明列每一軸。

### 四、整合題解答

有效 token 平均為

$$
L=
\frac{\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}
m_{bt}\ell_{bt}}
{\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}m_{bt}}.
$$

若全部遮罩為 `False`，分母為零。函式應丟出明確例外；不應回傳 NaN、無限值或假造的零損失。

資料切分應先在原始序列、文件、池槽或群組層級完成，再於各集合內切出窗口。詞表與標準化統計只從訓練集擬合；驗證集用於選擇設定，測試集不參與調參。

```python
import numpy as np


def valid_token_loss(losses, valid):
    if losses.ndim != 2:
        raise ValueError("losses 必須是 (B,T)")
    if valid.shape != losses.shape:
        raise ValueError("valid 與 losses shape 必須相同")
    if valid.dtype != np.bool_:
        raise TypeError("valid 必須是 bool")
    if not np.isfinite(losses).all():
        raise ValueError("losses 必須全部有限")

    n = int(valid.sum())
    if n == 0:
        raise ValueError("沒有有效 token")

    return losses[valid].sum(dtype=np.float64) / n
```

這裡先選出有效位置，再求一次總和，最後只除以有效 token 總數一次。

---

## 本章小結

張量 shape 不只是尺寸列表，也是一份軸語義契約。`(B,T,D)` 中的 batch、time 與 feature 不能任意交換，即使元素總數完全相同。

本章的核心原則如下：

1. 每個張量都應明列 shape 與軸語義。
2. `reshape` 重新分組元素，`transpose` 重新排列軸，兩者不可互換。
3. `matmul` 對最後兩軸做矩陣乘法，前導軸依廣播規則處理。
4. `einsum` 可直接表達索引與求和軸，但仍需 shape 檢查。
5. 廣播從右對齊；運算合法不代表語義正確。
6. reduction 必須明列軸、`sum` 或 `mean`，以及是否 `keepdims`。
7. $B=1$ 時仍保留 batch 軸，避免單樣本形成特殊介面。
8. transpose 後的非連續布局通常數值正確，但不能對資料連續性或是否複製作無條件假設。
9. 機率張量需在指定類別軸上非負且加總為一。
10. 有遮罩的 loss 應按有效 token 總數平均，且空有效集合必須拒絕。
11. 資料切分先於切窗及統計量擬合；shape 正確不能防止資料洩漏。

後續的注意力、批次線性層、正規化與 Transformer 都建立在這些契約上。越早把軸寫清楚，越少需要從一個「能執行但答案錯誤」的廣播結果中逆向除錯。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, 2017。  
   https://arxiv.org/abs/1706.03762  
   本章只將其作為後續注意力張量記號的背景入口；依題附來源註記，僅確認摘要入口，未據此宣稱完整核對論文細節。

2. NumPy, *Broadcasting* 使用指南。  
   https://numpy.org/doc/stable/user/basics.broadcasting.html  
   本章的廣播敘述與常見 NumPy 規則一致，但依題附來源註記，此入口仍標為待逐條核對，不宣稱已在本寫作流程連網查證。

3. Dive into Deep Learning。  
   https://d2l.ai/  
   可作為張量與深度學習實作的延伸入口；本章未依賴其中未核對章節建立證明。

4. PyTorch, *Reproducibility*。  
   https://docs.pytorch.org/docs/stable/notes/randomness.html  
   本章 NumPy 程式未宣稱跨版本、跨平台逐位重現；此連結僅列作後續可重現性議題的延伸入口。