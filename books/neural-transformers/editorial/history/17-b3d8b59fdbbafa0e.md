# 第17章 位置編碼、RoPE與相對位置

## 學習目標與先備知識

完成本章後，讀者應能：

1. 說明沒有位置資訊的注意力為何不能單靠內容分辨詞元順序。
2. 寫出 sinusoidal 位置編碼，並核對位置、批次與特徵軸的廣播。
3. 使用固定的相鄰偶奇維度配對實作 Rotary Position Embedding（RoPE）。
4. 證明二維旋轉保持範數，以及 RoPE 內積透過相對位置差依賴位置。
5. 區分「將位置向量加到內容表示」與「依位置旋轉 query、key」。
6. 正確處理偶數 head 維度、非零起始位置及 KV cache 的位置偏移。
7. 為位置方法建立可比較的合成實驗，而不把低訓練誤差當作泛化證據。
8. 明確區分位置向量、注意力 logits、注意力概率與訓練損失的數學域。

本章使用以下張量約定：

- $B$：batch 大小；
- $T$：序列長度；
- $D$：模型特徵維度；
- $H$：注意力頭數；
- $d_h=D/H$：每頭特徵維度。

多頭注意力中的 query 與 key 形狀為

$$
Q\in\mathbb{R}^{B\times H\times T_q\times d_h},
\qquad
K\in\mathbb{R}^{B\times H\times T_k\times d_h}.
$$

縮放內積分數為

$$
S=\frac{QK^\mathsf{T}}{\sqrt{d_h}}
\in\mathbb{R}^{B\times H\times T_q\times T_k},
$$

其中 $K^\mathsf{T}$ 只交換最後兩軸。softmax 沿最後的 key 軸 $T_k$ 計算，因此每個固定的 batch、head、query 對應一個 key 上的條件分布。

---

## 問題與直覺

### 注意力本身不認識順序

考慮兩個序列：

- 「魚 吃 蝦」
- 「蝦 吃 魚」

若每個詞元只有內容 embedding，模型沒有位置資訊，這兩個序列只是同一組向量的不同排列。自注意力會隨輸入排列一起排列，卻無法從運算本身判斷哪個詞元原本位於第一、第二或第三個位置。

設無位置資訊的注意力層為 $F$，排列矩陣為 $\Pi$。在沒有因果遮罩或其他位置結構時，通常有

$$
F(\Pi X)=\Pi F(X).
$$

這稱為排列等變性。它不是實作錯誤，而是表示模型需要額外訊號才能描述順序。

位置方法可粗分為三類：

1. **絕對位置表示**：為位置 $p$ 建立向量 $P_p$，加入詞元表示。
2. **相對位置偏置**：依 query 與 key 的距離修改注意力 logits。
3. **旋轉式位置表示**：依位置旋轉 query 與 key，使內積包含相對位移。

本章集中於 sinusoidal 位置編碼與 RoPE。兩者都使用不同頻率的正弦與餘弦，但介入模型的位置不同。

### 加法與旋轉不是同一操作

sinusoidal 位置編碼通常做

$$
X'_p=X_p+P_p.
$$

若 $X$ 的形狀為 $(B,T,D)$，位置表 $P$ 可取形狀 $(T,D)$，相加時視為 $(1,T,D)$，沿 batch 軸廣播。位置資訊先進入表示，再由線性投影產生 $Q,K,V$。

RoPE 通常在投影之後，對每個注意力頭的 $Q,K$ 旋轉：

$$
\widetilde Q_p=R_pQ_p,\qquad
\widetilde K_r=R_rK_r.
$$

value 通常不旋轉。RoPE 不是再加一個位置向量，而是改變不同位置向量之間的夾角，使 query-key 內積能以相對位移表示。

### 位置索引不等於實際時間

序列位置 $p=7$ 只表示某詞元在模型序列中的索引。它不自動表示第七分鐘、第七天，或與前一筆資料相隔固定時間。若感測資料有不規則時間間隔，還需要獨立的時間戳、時間差及缺測標記。不能把 RoPE 的索引距離直接解釋成真實物理時間。

---

## 定義、定理與推導

### Sinusoidal 位置編碼

令模型維度 $D$ 為正偶數。對位置 $p\ge 0$ 與配對索引 $i=0,\ldots,D/2-1$，定義頻率

$$
\omega_i=b^{-2i/D},
$$

其中 $b>0$ 是基底，常見設定為 $10000$。本章固定使用相鄰偶奇維度：

$$
P_{p,2i}=\sin(p\omega_i),
\qquad
P_{p,2i+1}=\cos(p\omega_i).
$$

長度為 $T$ 時，

$$
P\in\mathbb{R}^{T\times D}.
$$

加入批次輸入時，

$$
X'=X+P[None,:,:],
$$

其中

$$
X,X'\in\mathbb{R}^{B\times T\times D}.
$$

即使 $B=1$，也應保留 batch 軸。`reshape` 只能改變資料形狀的解讀方式，不能代替交換軸的 `transpose`。同樣地，依賴偶奇配對的特徵不能任意重排。

不同實作可能把 cosine 放在偶數維、sine 放在奇數維，也可能採用「前半特徵與後半特徵配對」。這些約定不一定誰絕對正確，但訓練、推論、checkpoint 與 KV cache 必須一致。本章一律採用相鄰配對 $(0,1),(2,3),\ldots$。

### 二維旋轉

對角度 $\theta$，定義

$$
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

對相鄰特徵 $(x_{2i},x_{2i+1})$，旋轉結果為

$$
\begin{aligned}
x'_{2i}
&=x_{2i}\cos\theta-x_{2i+1}\sin\theta,\\
x'_{2i+1}
&=x_{2i}\sin\theta+x_{2i+1}\cos\theta.
\end{aligned}
$$

RoPE 在位置 $p$ 的第 $i$ 組特徵使用

$$
\theta_{p,i}=p\omega_i.
$$

完整的 $R_p$ 由 $d_h/2$ 個二維旋轉區塊組成，因此每頭維度 $d_h$ 必須是正偶數。只檢查 $D$ 為偶數並不夠。例如 $D=12,H=4$ 時，

$$
d_h=D/H=3,
$$

仍不能依本章約定完成兩兩配對。實作應同時檢查 $D$ 可被 $H$ 整除，以及 $d_h$ 為偶數。

### 小命題：RoPE 保持範數並引入相對位置

**命題。** 設 $R_p$ 是各二維區塊角度為 $p\omega_i$ 的旋轉矩陣。對任意向量 $x$、query $q$、key $k$ 與位置 $p,r$：

$$
\|R_px\|_2=\|x\|_2,
$$

且

$$
(R_pq)^\mathsf{T}(R_rk)
=q^\mathsf{T}R_{r-p}k.
$$

**證明。**

先考慮一個二維區塊。直接計算：

$$
R(\theta)^\mathsf{T}R(\theta)
=
\begin{bmatrix}
\cos\theta&\sin\theta\\
-\sin\theta&\cos\theta
\end{bmatrix}
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
=I_2.
$$

完整的 $R_p$ 是這些正交矩陣組成的 block diagonal 矩陣，因此

$$
R_p^\mathsf{T}R_p=I.
$$

所以

$$
\begin{aligned}
\|R_px\|_2^2
&=(R_px)^\mathsf{T}(R_px)\\
&=x^\mathsf{T}R_p^\mathsf{T}R_px\\
&=x^\mathsf{T}x\\
&=\|x\|_2^2.
\end{aligned}
$$

兩邊皆非負，取平方根可得 $\|R_px\|_2=\|x\|_2$。

另一方面，二維旋轉滿足

$$
R(\alpha)^\mathsf{T}R(\beta)
=R(-\alpha)R(\beta)
=R(\beta-\alpha).
$$

逐區塊套用後有

$$
R_p^\mathsf{T}R_r=R_{r-p}.
$$

因此

$$
\begin{aligned}
(R_pq)^\mathsf{T}(R_rk)
&=q^\mathsf{T}R_p^\mathsf{T}R_rk\\
&=q^\mathsf{T}R_{r-p}k.
\end{aligned}
$$

命題得證。$\square$

這個結論表示位置部分可由差值 $r-p$ 表示，不表示分數只由距離決定。$q,k$ 的內容仍然參與內積。若另一份實作把相對位置定義為 query 位置減 key 位置，可能出現 $p-r$；必須連同旋轉方向與配對排列一起核對，不能只比較符號。

### RoPE 的張量形狀與廣播

令

$$
Q,K\in\mathbb{R}^{B\times H\times T\times d_h}.
$$

頻率向量形狀是 $(d_h/2)$，位置向量形狀是 $(T)$。外積形成角度表

$$
\Theta\in\mathbb{R}^{T\times d_h/2}.
$$

拆出偶數與奇數特徵後：

- `x[..., 0::2]`：$(B,H,T,d_h/2)$；
- `x[..., 1::2]`：$(B,H,T,d_h/2)$；
- `cos`、`sin`：$(1,1,T,d_h/2)$。

廣播只擴展 $B,H$ 軸。旋轉結果必須交錯寫回原位置。若直接使用

```python
np.concatenate([rot_even, rot_odd], axis=-1)
```

會得到「全部偶數結果在前、全部奇數結果在後」，不再符合相鄰配對約定。

### 與注意力概率的關係

RoPE 的輸出仍是實數向量，不是概率。旋轉後的注意力 logits 為

$$
S_{b,h,p,r}
=
\frac{\widetilde Q_{b,h,p,:}
\widetilde K_{b,h,r,:}^\mathsf{T}}
{\sqrt{d_h}}.
$$

若有遮罩，本卷約定 `True` 表示允許注意。呼叫端應先套用遮罩，再沿最後的 key 軸做穩定 softmax。對固定的 $(b,h,p)$，所得概率必須非負，且在允許的 key 支撐集上總和為一。

若某個 query 沒有任何允許的 key，實作 masked softmax 的呼叫端應明確拒絕，而不能讓 `NaN` 靜默傳播。需要特別說明：本章後面的自足程式只實作 RoPE 與**未遮罩的注意力 logits**，沒有實作 masked softmax，因此全遮罩拒絕不屬於該程式的驗收範圍。

### KV cache 的絕對位置

假設 cache 已保存位置 $0,\ldots,L-1$ 的 key。新 token 應使用位置 $L$：

$$
\widetilde q_L=R_Lq_L,\qquad
\widetilde k_L=R_Lk_L.
$$

若一次追加 $m$ 個 token，位置應為

$$
L,L+1,\ldots,L+m-1.
$$

本次位置索引的長度必須恰好等於輸入張量的 $T$ 軸長度。因果遮罩則應依絕對位置建立：

$$
\operatorname{allow}(p,r)\iff r\le p.
$$

cache 解碼時 $T_q$ 與 $T_k$ 通常不同，分數矩陣是矩形。因此不能無條件以 `tril(Tq, Tk)` 取代絕對位置比較。

---

## 逐步手算例題

### 例題一：sinusoidal 位置編碼

令 $D=4$、$b=100$。兩組頻率為

$$
\omega_0=100^0=1,
\qquad
\omega_1=100^{-2/4}=0.1.
$$

位置 $p=2$ 的編碼是

$$
P_2=[\sin2,\cos2,\sin0.2,\cos0.2].
$$

逐項取近似值：

$$
\sin2\approx0.9093,\qquad
\cos2\approx-0.4161,
$$

$$
\sin0.2\approx0.1987,\qquad
\cos0.2\approx0.9801.
$$

所以

$$
P_2\approx[0.9093,-0.4161,0.1987,0.9801].
$$

若內容向量為

$$
X_2=[1.0,0.5,-1.0,2.0],
$$

則

$$
\begin{aligned}
X'_2
&=X_2+P_2\\
&\approx[1.9093,0.0839,-0.8013,2.9801].
\end{aligned}
$$

若輸入形狀為 $(3,T,4)$，同一個 $P_2$ 沿 batch 軸廣播到三筆樣本。位置表不會因 batch 編號而改變。

### 例題二：RoPE 範數與相對位置

只考慮一組二維特徵，令頻率 $\omega=1$：

$$
q=
\begin{bmatrix}
1\\0
\end{bmatrix},
\qquad
k=
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

query 位於 $p=1$，key 位於 $r=3$。旋轉後：

$$
R_1q=
\begin{bmatrix}
\cos1\\
\sin1
\end{bmatrix},
$$

以及

$$
R_3k=
\begin{bmatrix}
\cos3-\sin3\\
\sin3+\cos3
\end{bmatrix}.
$$

其內積為

$$
\begin{aligned}
(R_1q)^\mathsf{T}(R_3k)
&=\cos1(\cos3-\sin3)
 +\sin1(\sin3+\cos3)\\
&=(\cos1\cos3+\sin1\sin3)\\
&\quad+(-\cos1\sin3+\sin1\cos3)\\
&=\cos(3-1)-\sin(3-1)\\
&=\cos2-\sin2.
\end{aligned}
$$

另一方面，

$$
R_{3-1}k
=
R_2
\begin{bmatrix}
1\\1
\end{bmatrix}
=
\begin{bmatrix}
\cos2-\sin2\\
\sin2+\cos2
\end{bmatrix},
$$

故

$$
q^\mathsf{T}R_2k=\cos2-\sin2.
$$

兩條計算路徑一致。範數方面，

$$
\|R_1q\|_2^2=\cos^21+\sin^21=1=\|q\|_2^2.
$$

### 例題三：cache 偏移錯位

prefill 已處理位置 $0,1,2$，故 cache 長度為 $L=3$。下一個 token 的未旋轉 query 為 $q$。

正確結果是

$$
\widetilde q=R_3q.
$$

若增量解碼程式只看到本次局部長度為一，便錯誤地從位置零開始，會得到

$$
\widetilde q_{\mathrm{wrong}}=R_0q=q.
$$

對 cache 中位置 $r=1$ 的 key，正確內積是

$$
(R_3q)^\mathsf{T}(R_1k)
=q^\mathsf{T}R_{1-3}k
=q^\mathsf{T}R_{-2}k.
$$

錯誤內積則是

$$
(R_0q)^\mathsf{T}(R_1k)
=q^\mathsf{T}R_1k.
$$

相對位移由 $-2$ 變成 $1$，因此不是無關緊要的共同平移。只有 query 與所有相關 key 的位置共同加上相同常數時，相對位置才保持不變。

---

## 實作與程式

以下程式只依賴 NumPy，預定在 CPU 執行；不下載權重、資料或 tokenizer。寫作流程沒有執行此程式，因此後文只列預期結果。

```python
import numpy as np


def _is_plain_integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(
        x, (bool, np.bool_)
    )


def sinusoidal_positions(length, dim, base=10000.0, start=0,
                         dtype=np.float64):
    if not _is_plain_integer(length) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if not _is_plain_integer(dim) or dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even integer")
    if not _is_plain_integer(start) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base must be positive and finite")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("dtype must be a floating dtype")

    positions = np.arange(start, start + length, dtype=dtype)
    pair = np.arange(dim // 2, dtype=dtype)
    inv_freq = np.asarray(base, dtype=dtype) ** (
        -2.0 * pair / dim
    )
    angles = positions[:, None] * inv_freq[None, :]

    out = np.empty((length, dim), dtype=dtype)
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)
    return out


def rope(x, positions, base=10000.0):
    x = np.asarray(x)
    positions = np.asarray(positions)

    if x.ndim != 4:
        raise ValueError("x must have shape (B, H, T, dh)")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("x must have a floating dtype")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base must be positive and finite")

    B, H, T, dh = x.shape
    if dh <= 0 or dh % 2 != 0:
        raise ValueError("head dimension dh must be positive and even")
    if positions.shape != (T,):
        raise ValueError("positions must have shape (T,)")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("positions must contain integer indices")
    if np.issubdtype(positions.dtype, np.bool_):
        raise TypeError("boolean positions are not valid indices")
    if np.any(positions < 0):
        raise ValueError("positions must be nonnegative")

    pair = np.arange(dh // 2, dtype=x.dtype)
    inv_freq = np.asarray(base, dtype=x.dtype) ** (
        -2.0 * pair / dh
    )
    angles = positions.astype(x.dtype)[:, None] * inv_freq[None, :]
    c = np.cos(angles)[None, None, :, :]
    s = np.sin(angles)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]

    out = np.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out


def rope_with_start(x, start, base=10000.0):
    if not _is_plain_integer(start) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError("x must have shape (B, H, T, dh)")
    T = x.shape[2]
    positions = np.arange(start, start + T, dtype=np.int64)
    return rope(x, positions, base=base)


def attention_scores(q, k):
    q = np.asarray(q)
    k = np.asarray(k)

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be rank-4")
    if q.shape[:2] != k.shape[:2]:
        raise ValueError("batch and head axes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must match")
    if q.shape[-1] <= 0:
        raise ValueError("head dimension must be positive")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(k)):
        raise ValueError("q and k must contain finite values")

    dh = q.shape[-1]
    return (
        np.matmul(q, np.swapaxes(k, -1, -2))
        / np.sqrt(dh)
    )


def run_tests():
    # 正常：位置表形狀與位置零。
    p = sinusoidal_positions(4, 6)
    assert p.shape == (4, 6)
    np.testing.assert_allclose(p[0, 0::2], 0.0, atol=1e-12)
    np.testing.assert_allclose(p[0, 1::2], 1.0, atol=1e-12)

    rng = np.random.default_rng(17)

    # 正常：RoPE 保持最後特徵軸的平方範數。
    x = rng.normal(size=(2, 3, 5, 4)).astype(np.float64)
    y = rope(x, np.arange(5, dtype=np.int64))
    np.testing.assert_allclose(
        np.sum(x * x, axis=-1),
        np.sum(y * y, axis=-1),
        rtol=1e-12,
        atol=1e-12
    )

    # 正常：共同平移全部位置，不改變兩兩分數。
    q = rng.normal(size=(1, 2, 3, 4))
    k = rng.normal(size=(1, 2, 3, 4))
    pos_a = np.array([0, 2, 5], dtype=np.int64)
    pos_b = pos_a + 11
    score_a = attention_scores(rope(q, pos_a), rope(k, pos_a))
    score_b = attention_scores(rope(q, pos_b), rope(k, pos_b))
    np.testing.assert_allclose(
        score_a, score_b, rtol=1e-11, atol=1e-11
    )

    # 正常：Tq != Tk 時得到矩形 logits。
    q_rect = rng.normal(size=(2, 3, 2, 4))
    k_rect = rng.normal(size=(2, 3, 5, 4))
    s_rect = attention_scores(q_rect, k_rect)
    assert s_rect.shape == (2, 3, 2, 5)

    # 邊界：T=0，仍保留 B、H、T、dh 四軸。
    empty = np.empty((1, 2, 0, 4), dtype=np.float64)
    empty_out = rope(empty, np.array([], dtype=np.int64))
    assert empty_out.shape == (1, 2, 0, 4)

    # 正常：整段旋轉等於使用正確 start 的分塊旋轉。
    full = rng.normal(size=(1, 2, 6, 4))
    full_rot = rope_with_start(full, 0)
    prefix = rope_with_start(full[:, :, :4, :], 0)
    suffix = rope_with_start(full[:, :, 4:, :], 4)
    joined = np.concatenate([prefix, suffix], axis=2)
    np.testing.assert_allclose(
        full_rot, joined, rtol=1e-12, atol=1e-12
    )

    # 故障注入：suffix 錯從位置 0 開始，應與正解不相近。
    wrong_suffix = rope_with_start(full[:, :, 4:, :], 0)
    wrong_joined = np.concatenate(
        [prefix, wrong_suffix], axis=2
    )
    assert not np.allclose(
        full_rot, wrong_joined, rtol=1e-12, atol=1e-12
    )

    # 故障：奇數 dh。
    caught = False
    try:
        rope(
            np.zeros((1, 1, 2, 3), dtype=np.float64),
            np.array([0, 1], dtype=np.int64)
        )
        raise AssertionError("odd dh was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：位置長度與 T 不一致。
    caught = False
    try:
        rope(
            np.zeros((1, 1, 2, 4), dtype=np.float64),
            np.array([0], dtype=np.int64)
        )
        raise AssertionError("position mismatch was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：sinusoidal 輸出 dtype 不可為整數。
    caught = False
    try:
        sinusoidal_positions(2, 4, dtype=np.int64)
        raise AssertionError("integer dtype was not rejected")
    except TypeError:
        caught = True
    assert caught

    # 故障：布林值不可冒充整數參數。
    caught = False
    try:
        sinusoidal_positions(True, 4)
        raise AssertionError("boolean length was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：非有限輸入。
    bad = np.zeros((1, 1, 1, 4), dtype=np.float64)
    bad[0, 0, 0, 0] = np.nan
    caught = False
    try:
        rope(bad, np.array([0], dtype=np.int64))
        raise AssertionError("NaN input was not rejected")
    except ValueError:
        caught = True
    assert caught

    print("all expected tests completed")


if __name__ == "__main__":
    run_tests()
```

`np.sum(..., axis=-1)` 只沿 $d_h$ 特徵軸求和，輸出形狀為 $(B,H,T)$。`attention_scores` 收縮 query、key 的最後特徵軸，輸出形狀為 $(B,H,T_q,T_k)$。矩形測試直接覆蓋 $T_q=2,T_k=5$ 的案例，而不只依賴讀者自行推導 `matmul` 規則。

`sinusoidal_positions` 明確拒絕整數 dtype，避免正弦與餘弦被截斷。它也拒絕布林 `length`、`dim` 與 `start`；在 Python 中布林值是整數的子類別，只檢查 `isinstance(True, int)` 會誤把 `True` 當成長度一。

cache 故障注入使用與正確比較相同的 `rtol=1e-12, atol=1e-12`，但斷言結果不相近。這裡比較的是同一組未旋轉向量在正確位置 $4,5$ 與錯誤位置 $0,1$ 下的結果。固定 seed 只讓測試資料可重建，不代表跨平台逐位相同。

---

## 測試與預期結果

本章沒有實際執行上述程式，以下均是預期行為。

### 正常測試

1. `sinusoidal_positions(4, 6)` 預期回傳形狀 `(4,6)`。
2. 位置零的 sine 維預期為零，cosine 維預期為一。
3. RoPE 前後沿 `axis=-1` 的平方範數預期在浮點容差內一致。
4. query 與 key 的全部位置共同加上常數時，注意力 logits 預期不變。
5. $T_q=2,T_k=5$ 時，`attention_scores` 預期回傳 `(B,H,2,5)`，不要求序列軸對稱。
6. 整段旋轉預期等於以正確 `start` 分塊旋轉後沿 `axis=2` 串接。
7. suffix 錯誤從零開始時，錯位結果預期不會在指定容差內等於正確結果。

### 邊界測試

- $T=0$ 時輸出應保持 `(B,H,0,dh)`。
- $B=1$、$H=1$ 時仍保留對應軸，不能使用 `squeeze` 意外刪除。
- $d_h=2$ 是相鄰配對下最小的正偶數 head 維度。
- 很大的位置索引在數學上仍可代入三角函數，但有限精度可能出現角度化簡誤差。可計算不等於可靠外推。
- `float32` 的容差通常需比 `float64` 寬，不能要求不合理的逐位相等。

### 故障測試

以下輸入應明確拒絕：

- 奇數 $d_h$；
- 位置向量長度不等於 $T$；
- 負位置、非整數位置或布林位置；
- 整數 sinusoidal 輸出 dtype；
- 非正或非有限基底；
- 含 `NaN` 或無限值的輸入；
- $Q,K$ 的 batch、head 或特徵軸不一致；
- cache suffix 使用錯誤起始位置。

本章程式沒有 masked softmax，因此不宣稱已測試全遮罩 query。若將本章函式接到完整注意力實作，該呼叫端還需另加正常遮罩、矩形因果遮罩與全遮罩拒絕測試。

---

## 反例與常見陷阱

### 陷阱一：把可計算外部位置當作外推保證

RoPE 與 sinusoidal 公式都能為訓練範圍外的位置產生數值，但模型可能只學到短序列上的規律。頻率配置、有限精度、資料分布、遮罩與模型參數共同決定實際表現。

因此：

> 能計算位置 $p$，不代表模型在位置 $p$ 上具有可靠能力。

研究長度外推時，應事先分開：

- 訓練：長度 $8$ 至 $32$；
- 驗證：相同長度範圍的新群組；
- IID 測試：相同範圍、未參與選模的群組；
- OOD 長度測試：長度 $33$ 至 $64$。

若根據 OOD 測試結果反覆修改模型，該集合便已參與調參，不能繼續稱為未碰觸測試集。

### 陷阱二：重疊窗口跨集合洩漏

若先從一條長日誌產生大量重疊窗口，再隨機把窗口分到訓練與測試，相鄰窗口可能只差一個 token。測試結果會受到近乎重複資料污染。

正確順序是：

1. 先按完整文件、池槽群組或時間區段切分；
2. 只用訓練集合建立詞表與其他統計量；
3. 再分別於各集合內產生窗口；
4. 用驗證集選模型及早停；
5. 測試集只做最後評估。

### 陷阱三：錯誤配對排列

本章配對為 `(0,1)、(2,3)、...`。將全部偶數結果與全部奇數結果直接串接，會改成另一種排列。若訓練與推論使用不同排列，即使 shape 完全相同，數值語義仍然錯誤。

### 陷阱四：cache 每步都從位置零開始

增量解碼張量的局部 $T$ 可能是 $1$，但它的絕對位置通常是目前 cache 長度。把每一步都當成位置零，會破壞新 query 與舊 key 的相對位置。

cache 也必須在不同樣本之間重設，否則上一筆樣本的 key、value 與位置狀態都會污染下一筆樣本。

### 陷阱五：範數不變等於分數不變

個別向量旋轉後範數不變，但兩個向量若旋轉不同角度，夾角與內積可以改變。只有所有相關位置共同平移相同常數時，相對位置不變，理想數學中的 RoPE 內積才保持不變。

### 陷阱六：注意力遮罩與 loss 遮罩混為一談

padding key 遮罩決定某 query 是否能注意該 key；loss 遮罩決定某 target 是否納入目標函數。兩者用途不同。即使 PAD key 不被注意，也不表示 PAD target 已自動從 loss 排除。

### 陷阱七：把注意力權重當成因果解釋

注意力權重會受位置方法影響，但高權重不等於某 token 是輸出的唯一因果來源。value、殘差、其他注意力頭、後續層與非線性都會影響結果。注意力圖可用於診斷，不應單獨充當因果證據。

---

## AI、幾何與養殖案例

考慮完全合成的養殖日誌，每筆記錄由以下離散欄位組成：

```text
[池槽代碼] [時間槽] [感測狀態] [例行事件]
```

任務是根據先前 token 預測下一 token。這只是位置方法的教材實驗，不含真實操作閾值，也不能控制投餌、泵浦、曝氣、加藥或其他設備。

### 資料契約

每個合成群組保存：

- `group_id`；
- 生成 seed；
- 合成時間範圍；
- 序列長度；
- 生成規則版本。

先按完整群組切分為訓練、驗證、IID 測試與長度偏移測試，再於各集合內建立窗口。同一原始序列的重疊窗口不得跨集合。詞表及任何標準化統計量只能由訓練集合擬合。

next-token 輸入與 target 錯開一格。例如原序列為

```text
BOS 池A 時段1 正常 記錄 EOS
```

則輸入取前五個 token，target 取後五個 token。PAD target 不納入 loss。

若 logits 形狀為 $(B,T,V)$，有效 token mask 為

$$
M\in\{0,1\}^{B\times T},
$$

而每個有效 token 的負對數似然為 $\ell_{b,t}$，則 loss 是

$$
L=
\frac{\sum_{b=1}^{B}\sum_{t=1}^{T}M_{b,t}\ell_{b,t}}
{\sum_{b=1}^{B}\sum_{t=1}^{T}M_{b,t}}.
$$

分子沿 batch 與 time 軸求和，分母是有效 token 總數，只除一次。若分母為零，應拒絕該 batch。評估集困惑度為

$$
\operatorname{PPL}
=
\exp\left(
\frac{\text{有效 token 總 NLL}}
{\text{有效 token 總數}}
\right).
$$

不能先求不同 token 數批次的困惑度，再取普通平均。

softmax 輸出是詞表上的條件概率，每項必須非負，且沿詞表軸總和為一。位置向量、旋轉後向量與注意力 logits 都不是概率。

### 可比較的三模型契約

比較三種設定：

1. 不加入任何位置資訊；
2. 在 $(B,T,D)$ embedding 上加入 sinusoidal 編碼；
3. 對 $(B,H,T,d_h)$ 的 $Q,K$ 使用 RoPE。

三種模型應使用：

- 相同訓練、驗證及測試資料；
- 相同 tokenizer 與詞表；
- 相同 embedding 維度、注意力頭數、FFN 維度及層數；
- 相同參數初始化 seed 規則；
- 相同 optimizer、學習率候選與最大更新步數；
- 相同有效 token 平均 loss；
- 相同驗證頻率與早停準則。

可預先規定最多訓練 $N$ 次更新，每隔固定更新數計算驗證 token NLL，以最低驗證 NLL 的 checkpoint 作為選模結果。超參數只能用驗證集選擇。模型選定後，才對 IID 測試與 OOD 長度測試評估一次，並報告：

- 有效 token 總數；
- token 加權 NLL；
- perplexity；
- 依序列長度分組的結果；
- 正常案例與失敗案例。

這是一份實驗設計，不是已完成的訓練報告。有限 seed 不代表統計顯著，也不能推廣到真實養殖資料。

---

## 習題

### 一、手算題

令 $d_h=4$、基底 $b=16$。求兩組頻率，並寫出位置 $p=2$ 對

$$
x=[1,0,0,1]
$$

施加 RoPE 後的結果。

### 二、程式題

使用本章的 `rope_with_start`，寫測試驗證：

1. 整段旋轉等於正確分塊旋轉；
2. suffix 錯誤從位置零開始時，結果與整段旋轉不相近；
3. 負數 `start` 被拒絕；
4. $T_q\ne T_k$ 時注意力 logits 形狀正確。

### 三、反例題

有人宣稱：「RoPE 保持每個 query 與 key 的範數，所以旋轉前後的注意力分數完全相同。」請給出二維反例。

### 四、整合題

設計一個比較無位置、sinusoidal 與 RoPE 的合成序列實驗。明列資料生成、切分順序、模型可比性、訓練與早停規則、loss 平均方式、測試指標，以及能支持與不能支持的結論。

---

## 習題解答

### 一、手算題解答

因 $d_h=4$，所以有兩組頻率：

$$
\omega_0=16^0=1,
\qquad
\omega_1=16^{-2/4}=16^{-1/2}=\frac14.
$$

位置 $p=2$ 的角度為

$$
\theta_0=2,\qquad
\theta_1=\frac12.
$$

第一組 $(1,0)$ 旋轉後為

$$
R(2)
\begin{bmatrix}
1\\0
\end{bmatrix}
=
\begin{bmatrix}
\cos2\\
\sin2
\end{bmatrix}.
$$

第二組 $(0,1)$ 旋轉後為

$$
R(1/2)
\begin{bmatrix}
0\\1
\end{bmatrix}
=
\begin{bmatrix}
-\sin(1/2)\\
\cos(1/2)
\end{bmatrix}.
$$

交錯寫回原位置：

$$
\operatorname{RoPE}(x,2)
=
[\cos2,\sin2,-\sin(1/2),\cos(1/2)].
$$

其平方範數為

$$
\cos^22+\sin^22+\sin^2(1/2)+\cos^2(1/2)=2,
$$

與原向量平方範數 $1^2+1^2=2$ 相同。

### 二、程式題解答

```python
rng = np.random.default_rng(9)
x = rng.normal(size=(1, 2, 7, 4))

whole = rope_with_start(x, 0)
left = rope_with_start(x[:, :, :3, :], 0)
right = rope_with_start(x[:, :, 3:, :], 3)
joined = np.concatenate([left, right], axis=2)

np.testing.assert_allclose(
    whole, joined, rtol=1e-12, atol=1e-12
)

wrong_right = rope_with_start(x[:, :, 3:, :], 0)
wrong_joined = np.concatenate([left, wrong_right], axis=2)
assert not np.allclose(
    whole, wrong_joined, rtol=1e-12, atol=1e-12
)

try:
    rope_with_start(x, -1)
    raise AssertionError("negative start was not rejected")
except ValueError:
    pass

q = rng.normal(size=(1, 2, 2, 4))
k = rng.normal(size=(1, 2, 5, 4))
scores = attention_scores(q, k)
assert scores.shape == (1, 2, 2, 5)
```

第一項核對正確位置偏移，第二項主動注入錯誤偏移，第三項核對輸入契約，第四項直接驗證矩形注意力 logits。這些是預期測試，本文沒有執行。

### 三、反例題解答

取

$$
q=k=
\begin{bmatrix}
1\\0
\end{bmatrix}.
$$

旋轉前內積為

$$
q^\mathsf{T}k=1.
$$

令 query 位於位置 $0$，key 位於整數位置 $1$，頻率為 $1$。則

$$
R_0q=
\begin{bmatrix}
1\\0
\end{bmatrix},
\qquad
R_1k=
\begin{bmatrix}
\cos1\\
\sin1
\end{bmatrix}.
$$

旋轉後內積為

$$
(R_0q)^\mathsf{T}(R_1k)=\cos1\ne1.
$$

兩個向量的範數仍然都是一，但內積已改變，因此原主張錯誤。

### 四、整合題解答

先生成多個獨立群組。每條序列含查詢符號及兩個候選符號，target 由候選符號的先後或固定相對距離決定，使無位置模型不能只靠 token 集合完成任務。保存每個群組的 seed、規則版本與長度。

先按群組切成訓練、驗證、IID 測試與 OOD 長度測試，之後才建立窗口。詞表只由訓練集合擬合。同一群組及其重疊窗口不得跨集合。

三種模型具有相同 $D,H,d_h$、層數、FFN、初始化規則、optimizer、最大更新數與學習率候選，只切換位置機制。每隔固定更新數計算驗證 NLL，以最低驗證 NLL 選 checkpoint；測試集不參與選模。

對有效 mask $M$，loss 為

$$
L=
\frac{\sum_{b,t}M_{b,t}\ell_{b,t}}
{\sum_{b,t}M_{b,t}},
$$

只按有效 token 總數除一次。若沒有有效 token，拒絕該 batch。

正常測試包括 shape、有限 logits、矩形注意力 shape、三模型接收相同資料，以及 RoPE 共同位置平移不改變分數。邊界測試包括 $B=1$、最短序列、最大預定長度與單一有效 token。故障測試包括奇數 $d_h$、位置長度錯誤、全 PAD loss、cache 錯位與跨樣本 cache 污染。若完整模型加入 masked softmax，還需測試全遮罩 query 被拒絕。

最終報告 IID 與 OOD 集合的有效 token 數、token 加權 NLL、perplexity 與按長度分組結果。若 RoPE 在保留資料上較佳，只能支持它在此生成規則、模型尺度與有限 seed 下較佳；不能證明它對任意長度、任意語料或真實養殖場景都能外推。

---

## 本章小結

sinusoidal 位置編碼把位置向量加到內容表示；RoPE 則依位置旋轉每個注意力頭的 query 與 key。本章固定使用相鄰偶奇特徵配對，因此 $d_h$ 必須是正偶數。

二維旋轉是正交變換，所以保持範數。query 位於 $p$、key 位於 $r$ 時，旋轉後內積可寫成 $q^\mathsf{T}R_{r-p}k$，使位置透過相對位移進入分數。但分數仍依賴內容，而且這項幾何性質不保證模型能對任意序列長度外推。

實作上應核對 $(B,H,T,d_h)$ 軸、浮點 dtype、位置長度、配對排列與 cache 起點。測試不只要驗證正例，也要注入錯誤位置偏移，並直接覆蓋 $T_q\ne T_k$ 的矩形情況。本章程式只處理 RoPE 與未遮罩 logits；全遮罩 query 的拒絕應由完整 masked softmax 實作另行驗收。

模型能力必須透過先切群組、再建窗口的訓練／驗證／測試契約評估。低訓練 loss、可計算長位置與注意力圖，都不能單獨充當泛化證據。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, 2017，https://arxiv.org/abs/1706.03762  
   提供 Transformer 與 sinusoidal 位置編碼的來源入口。依題目附註，只取得摘要頁，未宣稱逐段核對全文。

2. PyTorch 2.14 `scaled_dot_product_attention` API，https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html  
   與遮罩語義及注意力介面相關。已依題目附註核對 `True` 參與注意、評估時需明確令 dropout 機率為零及矩形因果遮罩語義；這不是本機版本或執行證據。

3. NumPy broadcasting 使用指南，https://numpy.org/doc/stable/user/basics.broadcasting.html  
   待逐條核對的延伸入口，本章不宣稱已查證其全部內容。

4. Dive into Deep Learning，https://d2l.ai/  
   延伸教材入口，未逐章核對。

5. PyTorch reproducibility notes，https://docs.pytorch.org/docs/stable/notes/randomness.html  
   可重現性延伸入口，未逐條核對。固定 seed 不代表跨版本與跨平台逐位相同。

題目提供的來源清單未包含 RoPE 原始論文，因此本章不虛構相關文獻查證狀態，而以明確定義、完整小命題證明、逐步手算及自足測試說明其數學性質。