# 第16章 多頭注意力與拆分合併

## 學習目標與先備知識

讀完本章，讀者應該能夠：

1. 寫出多頭注意力從 `(B,T,D)` 到 `(B,H,T,dh)` 的拆分與合併操作，說明為何這兩個運算互為逆運算，並推導合併與拆分的反向接口。
2. 在 NumPy 中從零實作一個自足的多頭注意力前向，包含穩定 softmax、遮罩與全遮罩行的拒絕策略。
3. 區分 `reshape` 與 `transpose` 的語意差異，指出在拆分與合併中各自扮演的角色。
4. 識別常見失敗：`D` 不被 `H` 整除、全遮罩列、先轉置再重塑造成的索引錯位、`batch=1` 時丟失軸。

**先備知識**：前一章的縮放點積注意力、固定張量軸約定 `B/T/D/H/dh`、穩定 softmax 的減最大值技巧、布林遮罩 True=允許的約定、矩陣微分的基本鏈式法則。

## 問題與直覺

單頭注意力在一個 `D` 維子空間內計算相似度。若輸入需要同時關注不同「種類」的關係（例如相鄰位置的語法關係、遠距離的共指關係、與特徵通道相關的鍵值關係），單一頭只能折衷，注意力會互相競爭。多頭注意力把 `D` 維特徵切成 `H` 個 `dh=D/H` 維子空間，每個頭在自己的子空間內**獨立**做縮放點積注意力，再把 `H` 個輸出合併回 `D` 維。

代價是形狀變換：要把 `(B,T,D)` 改成 `(B,H,T,dh)`，計算後再合併回去。關鍵契約是：**每個 token 的特徵向量沿最後一維被切成 H 段，第 h 段進入第 h 個頭；合併時必須按同一規則拼回**。若順序搞錯（例如先把 `(T,D)` 轉置再 reshape），token 與特徵的對應就會打亂，模型雖然形狀正確但語意錯誤。

拆分與合併在數學上是一對互逆的重排。它們不引入任何數值運算——只是軸的置換與形狀改變。所有數值計算（投影、縮放、softmax、加權和）都發生在拆分後或合併前。

## 定義、定理與推導

### 形狀契約

固定符號：

- `B`：batch 大小。
- `T`：序列長度（query 與 key 的長度都寫 `T`；若要用交叉注意力可在推廣中改為 `Tq`、`Tk`）。
- `D`：模型特徵維度，`H` 為頭數，要求 `D % H == 0`。
- `dh = D / H`：每個頭的特徵維度。
- `Q_h, K_h, V_h ∈ ℝ^{B×H×T×dh}`：拆分後的張量。
- `S ∈ ℝ^{B×H×T×T}`：分數；softmax 沿最後一維（key 軸）。
- `A ∈ ℝ^{B×H×T×T}`：注意力權重。
- `O_h ∈ ℝ^{B×H×T×dh}`：每頭輸出。
- `O ∈ ℝ^{B×T×D}`：合併後輸出。
- `Y ∈ ℝ^{B×T×D_out}`：輸出投影後。

### 定義 16.1（拆分 split 與合併 merge）

給定 `X ∈ ℝ^{B×T×D}` 與正整數 `H`，令 `dh = D/H`。

**拆分**：split(X) 為兩步複合：

1. `reshape`：把 `(B, T, D)` 改為 `(B, T, H, dh)`。沿最後一維按順序每 `dh` 個元素歸一組。
2. `transpose`：交換軸 1（`T`）與軸 2（`H`），得到 `(B, H, T, dh)`。

**合併**：merge(Y) 為反向複合，`Y ∈ ℝ^{B×H×T×dh}`：

1. `transpose`：交換軸 1 與軸 2，得到 `(B, T, H, dh)`。
2. `reshape`：把 `(B, T, H, dh)` 改回 `(B, T, D)`，`D = H * dh`。

### 命題 16.1（拆合互逆）

若 `D = H · dh`，則對任意 `X ∈ ℝ^{B×T×D}` 都有 `merge(split(X)) = X`。

**證明**。記 reshape 運算 `R`：`(B,T,D) → (B,T,H,dh)`；`R'`：`(B,T,H,dh) → (B,T,D)`；記交換軸 1 與 2 為 `P`。

- `split = P ∘ R`。
- `merge = R' ∘ P`。

先證 `P ∘ P = I`：轉置是軸置換，交換軸 1 和 2 兩次即回到原張量，恆等成立。

再證 `R' ∘ R = I`：reshape 在記憶體的 C 順序（row-major）下把扁平的 `D` 個元素按 `D = h·dh + i`（`h∈{0,…,H-1}`，`i∈{0,…,dh-1}`）分組；`R'` 把 `(h,i)` 還原為 `d = h·dh + i`。兩者對扁平索引的映射互為逆，且不改變元素在序列中的排列。因此 `R' ∘ R = I`。

最後，

$$
\operatorname{merge}(\operatorname{split}(X)) = R'(P(P(R(X)))) = R'(R(X)) = X. \qquad \blacksquare
$$

**推論**。拆分的反向即合併、合併的反向即拆分，梯度在兩者之間傳遞時僅做軸重排，不引入新的數值運算。

### 多頭注意力前向

給定 `X ∈ ℝ^{B×T×D}`，參數 `W_Q, W_K, W_V ∈ ℝ^{D×D}`、`W_O ∈ ℝ^{D×D_out}`，偏置 `b_Q, b_K, b_V ∈ ℝ^{D}`、`b_O ∈ ℝ^{D_out}`：

$$
\begin{aligned}
Q &= XW_Q + b_Q, & K &= XW_K + b_K, & V &= XW_V + b_V, \\
Q_h &= \operatorname{split}(Q), & K_h &= \operatorname{split}(K), & V_h &= \operatorname{split}(V), \\
S &= Q_h K_h^{\top} / \sqrt{dh}, & S &\in \mathbb{R}^{B\times H \times T \times T}, \\
A &= \operatorname{softmax}(S; \text{axis}=-1), \\
O_h &= A V_h, & O_h &\in \mathbb{R}^{B\times H\times T\times dh}, \\
O &= \operatorname{merge}(O_h), & O &\in \mathbb{R}^{B\times T\times D}, \\
Y &= OW_O + b_O, & Y &\in \mathbb{R}^{B\times T\times D_{out}}.
\end{aligned}
$$

**註**：

- `K_h^{\top}` 在此指對最後兩軸做轉置：`(B,H,T,dh) → (B,H,dh,T)`。寫成索引是 `K_h[b,h,k,:]`，點積沿 `dh` 收縮。
- 縮放因子用 `√dh` 而非 `√D`：每個頭的內積來自 `dh` 維，若各維獨立同分布、均值 0、變異數 1，則內積方差維 `dh`，除以 `√dh` 後方差歸一。
- 遮罩施加在 softmax 之前；本卷約定布林 True=允許。
- **參數數量**：多頭用 `4(D^2 + D)` 個參數（含偏置），與單頭在相同維度下相同。差異是 `W_Q` 等被隱式等分為 `H` 個 `D×dh` 塊，每塊供一個頭使用。

### 反向的拆分與合併

記上游梯度 `dO_h ∈ ℝ^{B×H×T×dh}`、`dQ ∈ ℝ^{B×T×D}`。由命題 16.1：

```
merge 的反向：dO (B,T,D)    -> reshape (B,T,H,dh) -> transpose(1,2) -> (B,H,T,dh)
split 的反向：dQh (B,H,T,dh) -> transpose(1,2) -> (B,T,H,dh) -> reshape (B,T,D)
```

這兩個反向操作的形狀和正向互補。實作中經常把它們寫成獨立函式，好處是把「重排」與「數值」分開除錯：如果有限差分在重排函式上不通過，問題必在軸操作而不在公式。

## 逐步手算例題

### 手算 1：拆分的往返

給定 `B=1, T=2, D=4, H=2, dh=2`。

$$
X[0,0,:] = [1, 2, 3, 4], \quad X[0,1,:] = [5, 6, 7, 8].
$$

**reshape 到 (1,2,2,2)**：

- token 0 → `[[1,2],[3,4]]`
- token 1 → `[[5,6],[7,8]]`

**transpose 軸 1、2 到 (1,2,2,2)**：

- 頭 0（每 token 的第 0 塊）：token 0 → `[1,2]`；token 1 → `[5,6]`
- 頭 1：token 0 → `[3,4]`；token 1 → `[7,8]`

即 `X_h[0,0,:,:] = [[1,2],[5,6]]`，`X_h[0,1,:,:] = [[3,4],[7,8]]`。

**合併**：先 transpose 恢復為 `(1,2,2,2)`：

- token 0 → `[[1,2],[3,4]]`
- token 1 → `[[5,6],[7,8]]`

再 reshape 回 `(1,2,4)`：token 0 → `[1,2,3,4]`，token 1 → `[5,6,7,8]`，與原 `X` 逐元素相同。這驗證了命題 16.1 在一個具體大小上的成立。

### 手算 2：雙頭注意力前向

沿用 `B=1, T=2, D=4, H=2, dh=2`。取 `W_Q = W_K = W_V = W_O = I_4`，所有偏置為零，則 `Q = K = V = X`。

$$
X[0,0,:] = [1, 2, 0, 0], \quad X[0,1,:] = [0, 0, 1, 1].
$$

**拆分**（沿用定義）：

- 頭 0：`Q_h^0 = K_h^0 = V_h^0 = [[1,2],[0,0]]`，形狀 `(1,2,2)`。
- 頭 1：`Q_h^1 = K_h^1 = V_h^1 = [[0,0],[1,1]]`。

**頭 0**：

$$
S_0 = \frac{Q_h^0 K_h^{0\top}}{\sqrt{2}}
= \frac{1}{\sqrt{2}}\begin{pmatrix} 1\cdot 1 + 2\cdot 2 & 1\cdot 0 + 2\cdot 0 \\ 0\cdot 1 + 0\cdot 2 & 0\cdot 0 + 0\cdot 0 \end{pmatrix}
= \begin{pmatrix} 3.5355 & 0 \\ 0 & 0 \end{pmatrix}.
$$

逐列 softmax：

- 第 0 列：`[e^{3.5355}, e^0] / (e^{3.5355}+e^0) ≈ [34.31, 1]/35.31 ≈ [0.9717, 0.0283]`。
- 第 1 列：`[1,1]/2 = [0.5, 0.5]`。

$$
A_0 \approx \begin{pmatrix} 0.9717 & 0.0283 \\ 0.5 & 0.5 \end{pmatrix},\quad
O_h^0 = A_0 V_h^0 = \begin{pmatrix} 0.9717 & 1.9434 \\ 0.5 & 1.0 \end{pmatrix}.
$$

**頭 1**：

$$
S_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 0\cdot 0 + 0\cdot 0 & 0\cdot 1 + 0\cdot 1 \\ 1\cdot 0 + 1\cdot 0 & 1\cdot 1 + 1\cdot 1 \end{pmatrix}
= \begin{pmatrix} 0 & 0 \\ 0 & 1.4142 \end{pmatrix}.
$$

逐列 softmax：

- 第 0 列：`[1,1]/2 = [0.5, 0.5]`。
- 第 1 列：`[1, e^{1.4142}]/(1+e^{1.4142}) ≈ [1, 4.1132]/5.1132 ≈ [0.1956, 0.8044]`。

$$
A_1 \approx \begin{pmatrix} 0.5 & 0.5 \\ 0.1956 & 0.8044 \end{pmatrix},\quad
O_h^1 = A_1 V_h^1 = \begin{pmatrix} 0.5 & 0.5 \\ 0.8044 & 0.8044 \end{pmatrix}.
$$

**合併**（拼接每頭輸出到特徵維）：

- token 0：`[0.9717, 1.9434, 0.5, 0.5]`
- token 1：`[0.5, 1.0, 0.8044, 0.8044]`

`W_O = I`，輸出 `Y = O`，形狀 `(1,2,4)`。

**檢查**：兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的「模式」。若把 `X` 改成兩頭得到相同投影的情形（例如 `X = [[1,0,1,0],[0,1,0,1]]`），兩頭的輸出一模一樣——這正是「頭塌縮」的小型演示。

## 實作與程式

以下程式為自足 NumPy 實作，**未在本寫作環境執行**；形狀與數值均為預期結果，讀者須自行運行驗證。

```python
import numpy as np

class MultiHeadAttention:
    """自足多頭注意力前向。

    張量約定：
      X:    (B, T, D)
      Q,K,V:(B, T, D) 投影後 -> 拆分 (B,H,T,dh)
      S,A:  (B, H, T, T)
      O_h:  (B, H, T, dh)
      O:    (B, T, D)
    """
    def __init__(self, D, H, D_out=None, seed=0):
        if D <= 0 or H <= 0:
            raise ValueError("D 與 H 必須為正")
        if D % H != 0:
            raise ValueError(f"D={D} 不被 H={H} 整除；請改用能整除的 (D,H)")
        self.D = D
        self.H = H
        self.dh = D // H
        self.D_out = D if D_out is None else D_out
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(D)
        self.W_Q = rng.normal(0, s, (D, D))
        self.W_K = rng.normal(0, s, (D, D))
        self.W_V = rng.normal(0, s, (D, D))
        self.W_O = rng.normal(0, s, (D, self.D_out))
        self.b_Q = np.zeros(D)
        self.b_K = np.zeros(D)
        self.b_V = np.zeros(D)
        self.b_O = np.zeros(self.D_out)

    def split_heads(self, X):
        # (B,T,D) -> (B,H,T,dh)
        if X.ndim != 3 or X.shape[-1] != self.D:
            raise ValueError(f"split 期望 (B,T,{self.D})，得到 {X.shape}")
        B, T, D = X.shape
        Xc = np.ascontiguousarray(X, dtype=X.dtype)
        return Xc.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)

    def merge_heads(self, Y):
        # (B,H,T,dh) -> (B,T,D)
        if Y.ndim != 4 or Y.shape[1] != self.H or Y.shape[-1] != self.dh:
            raise ValueError(f"merge 期望 (B,{self.H},T,{self.dh})，得到 {Y.shape}")
        B, H, T, dh = Y.shape
        return Y.transpose(0, 2, 1, 3).reshape(B, T, H * dh)

    def forward(self, X, mask=None):
        if X.ndim != 3 or X.shape[-1] != self.D:
            raise ValueError(f"forward 期望 (B,T,{self.D})，得到 {X.shape}")
        B, T, D = X.shape
        Q = X @ self.W_Q + self.b_Q
        K = X @ self.W_K + self.b_K
        V = X @ self.W_V + self.b_V
        Qh = self.split_heads(Q)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)
        S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
        if mask is not None:
            m = mask
            if m.dtype != np.bool_:
                raise ValueError("mask 必須為布林；True=允許")
            if m.ndim == 2:
                m = m[None, None]
            elif m.ndim == 3:
                m = m[:, None]
            elif m.ndim != 4:
                raise ValueError(f"mask 維度應為 2/3/4，得到 {m.ndim}")
            if m.shape[-2:] != (T, T):
                raise ValueError(f"mask 最後兩維應為 ({T},{T})，得到 {m.shape[-2:]}")
            if np.any(np.all(~m, axis=-1)):
                raise ValueError("存在全遮罩 query 列；拒絕輸出 NaN")
            S = np.where(m, S, -np.inf)
        mx = np.max(S, axis=-1, keepdims=True)
        ex = np.exp(S - mx)
        A = ex / np.sum(ex, axis=-1, keepdims=True)
        Oh = A @ Vh
        O = self.merge_heads(Oh)
        Y = O @ self.W_O + self.b_O
        return Y, A
```

### 反向接口

拆分與合併的反向只做軸重排：

```python
def split_backward(dQh, B, T, D, H, dh):
    # dQh: (B,H,T,dh) -> (B,T,D)
    return dQh.transpose(0, 2, 1, 3).reshape(B, T, D)

def merge_backward(dO, B, T, D, H, dh):
    # dO: (B,T,D) -> (B,H,T,dh)
    return np.ascontiguousarray(dO).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
```

這些函式不改變元素排列之外的值，因此若上游梯度正確，重排層的能量守恆（逐元素對應），有限差分在「全 1 上游梯度」下應為精確恒等。

## 測試與預期結果

**本節結果為預期；未在寫作環境執行。**

### 正常路徑

1. `D=8, H=2, B=2, T=3`：`Y.shape == (2,3,8)`，`A.shape == (2,2,3,3)`。
2. `A.sum(axis=-1)` 沿 key 軸應約為 1（誤差 < 1e-8，float64）。若偏差大於 1e-5，檢查是否沿錯誤軸做 softmax。
3. 拆分往返：對隨機 `X (2,3,8)`，`np.allclose(att.merge_heads(att.split_heads(X)), X)` 應為 True。

### 邊界條件

1. `H=1`：退化為單頭，`A.shape == (B,1,T,T)`，模型與單頭縮放點積注意力等價（權重不同）。
2. `H=D`（即 `dh=1`）：分數為純量內積的縮放。預期不會引發形狀錯誤；若實作把 `dh=1` 誤寫為 `reshape(B,T,1,D)` 則會失敗。
3. `B=1`：仍需保留 4 軸，`A.shape == (1,H,T,T)`；不要因為 batch 為 1 就擠壓軸，否則後續廣播會悄悄出錯。
4. `T=1`：分數為 `1×1`，softmax 退化為 1.0；`A[...,0,0] == 1.0`。
5. `D_out ≠ D`：若在 `__init__` 指定 `D_out`，`Y.shape == (B,T,D_out)`，`W_O.shape == (D, D_out)`。

### 故障測試

1. **非整除**：`D=6, H=4` → `__init__` 拋 `ValueError`。
2. **全遮罩列**：`mask` 含全 False 的列 → `forward` 拋 `ValueError`，訊息需含「全遮罩」。**不要**用 `np.where` 讓 `-inf` 進入 softmax 產生 NaN。
3. **遮罩形狀錯**：`mask.shape=(T,)` 或 `(B,T,T+1)` → 拋 `ValueError`。
4. **非布林遮罩**：傳入 `float32` mask → 拋 `ValueError`；這避免用 0/1 混過布林約定。
5. **錯誤順序**：故意先 `transpose` 後 `reshape`，形狀可能仍正確，但 `merge(split(X)) != X` 會被命題 16.1 的數值檢驗抓到。

## 反例與常見陷阱

1. **先轉置再重塑**。若有人把 `(B,T,D)` 先視為 `(B,T,H,dh)` 之前就 `transpose` 到 `(B,D,T)`，再 `reshape` 到 `(B,H,T,dh)`，形狀看起來對，但 `d` 索引對應已亂。診斷方法：用一個「只有一個位置非零」的 one-hot 張量輸入，看合併後非零位置是否回到原位。
2. **用 `√D` 而非 `√dh` 縮放**。當 `H>1` 且各維獨立、方差為 1 時，`QK^T` 的每個元素方差為 `dh` 而非 `D`。若仍除以 `√D`，分數尺度偏小，softmax 進入「幾乎均勻」區域，梯度訊號變弱。
3. **合併時忘了轉置回來**。直接 `reshape(B,H,T,dh)` 到 `(B,T,D)` 會得到 `H·dh = D` 但順序是「按頭優先」而非「按位置優先」，與拆分不對稱。
4. **全遮罩行**。`-inf` 進入 `np.max`，得到 `-inf`，再 `exp(-inf - (-inf)) = nan`。NaN 一旦產生，任一參數梯度都會變 NaN。必須在 softmax 前顯式拒絕或另立規則（如整體輸出為零），且該規則必須伴隨測試。
5. **`B=1` 就把軸壓掉**。壓軸後某些廣播會意外成功，卻在下一個 `matmul` 才失敗，錯誤訊息不指向根因。
6. **注意力權重當因果解釋**。權重大不代表「模型在看那裡」，也不代表該維度對下游有貢獻。任何頭級的可解釋性主張需另行驗證。
7. **資料切分洩漏**。注意力本身不會處理「同文件重疊窗口跨集合」，這需要在窗口建立之前先把文件／序列分到 train/val/test。多頭不會修正切分錯誤。

## AI、幾何與養殖案例

**幾何直覺**：如果把 `W_Q` 的行看成「查詢基底」，多頭相當於把 `D` 維查詢空間拆成 `H` 個 `dh` 維子空間的直和。每個頭學習一組方向，並在自己的子空間內做相似度。**但這只是分解的語言，不是約束**：訓練不會強迫頭與頭之間正交，也不會強迫頭對應到人類可命名的關係。要斷言「第 3 個頭在看時間鄰接」需要控制變數的實驗（例如替換、消融、注意力可視化），不能只看權重。

**AI 訓練觀點**：多頭把一個大 `D×D` 投影拆成 `H` 個小塊並行，好處是 (i) 每個頭獨立 softmax，避免單一大分數壟斷；(ii) 不同頭可以在不同子空間用不同縮放下的相似度；(iii) 在 GPU 上 `B×H` 可以合併維度以增加並行度。很多框架內部把 `(B,H,T,dh)` 寫成 `(B*H, T, dh)` 更快，但數學等價。實作對接時，必須在 `(B,H,T,dh)` 與 `(B*H,T,dh)` 之間做明確轉換。

**合成養殖日誌案例**：考慮一個完全合成的日誌表格，每列是一個時間戳的感測讀數（溫度、溶氧、pH），每列另外附一段簡短的文字操作記錄（「換水」「投餌」「巡檢」）。

設計一個合成任務：給定前 `T` 個時間戳的讀數與文字，預測下一個時間戳的溶氧等級（離散分箱）。用多頭注意力的動機是：

- 頭 A 可以關注「同一感測器在時間上的自相關」。
- 頭 B 可以關注「文字記錄（如投餌）與後續讀數變化」。
- 頭 C 可以關注「不同感測器在同一時刻的相關」。

這些只是設計意圖，不是實驗結果。真實是否發生需在合成資料上跑消融。**所有感測器讀數、操作記錄、詞表、切分、seed 都必須由程式碼顯式合成並記錄**，不得引用任何真實養殖場資料；任何「投餌閾值、曝氣時機」都不在本章討論範圍，也不得由此模型驅動。

## 習題

**手算題 16.1** 給定 `B=1, T=2, D=2, H=2, dh=1`。`Q = [[1,0],[0,1]]`、`K = [[1,1],[1,1]]`、`V = [[1,2],[3,4]]`（每列按 `(T, D)` 排）。無遮罩。分別計算兩頭的注意力輸出並合併。輸出每步的形狀與數值。

**程式題 16.2** 擴充 `MultiHeadAttention` 支援 `dv ≠ dh` 的情況：`W_V ∈ ℝ^{D×D_v}`、`W_O ∈ ℝ^{H·dv×D_out}`、合併後最後維為 `H·dv`。只在類別裡改動所需的最少行數，並保留「D 必須被 H 整除」的檢查（用於 `dh`）。報告：(a) 你改了哪幾行；(b) 拆分後 `V_h` 的形狀；(c) 合併函式為什麼要同時知道 `H` 和 `dv`。

**反例題 16.3** 給定 `X` 形狀 `(1,2,4)`、`X[0,0]=[1,2,3,4]`、`X[0,1]=[5,6,7,8]`。有人寫：

```python
Xh = X.transpose(0, 2, 1).reshape(1, 2, 2, 2)  # 先轉置後重塑
```

請列出 `Xh[0,h,t,i]` 的具體數值，並說明它與正確拆分 `X.split_heads` 的差異。用一句話描述何時「形狀對但語意錯」。

**綜合題 16.4** 用 PyTorch 的 `torch.nn.MultiheadAttention`（`batch_first=True`, `dropout=0.0`, `bias=True`）與本章 `MultiHeadAttention` 交叉驗證：把 PyTorch 的 `in_proj_weight`、`in_proj_bias`、`out_proj.weight`、`out_proj.bias` 複製到 NumPy 實作（先確定切分順序）。在相同 `X`（float64）與無遮罩下，比較 `Y` 的最大絕對誤差。`eval()` 模式、`dropout=0.0`、CPU。報告誤差量級與「若 > 1e-3 你會先檢查什麼」。**不執行**：本題須由讀者在自己機器上跑；寫作流程不執行任何外部程式。

## 習題解答

**16.1** `D=2, H=2, dh=1`。拆分（沿最後一維切成 2 段，每段 1 維）：

- `Q_h^0 = [[1],[0]]`, `Q_h^1 = [[0],[1]]`
- `K_h^0 = [[1],[1]]`, `K_h^1 = [[1],[1]]`
- `V_h^0 = [[1],[3]]`, `V_h^1 = [[2],[4]]`

**頭 0**：

$$
S_0 = \frac{Q_h^0 K_h^{0\top}}{\sqrt{1}} = \begin{pmatrix}1\\0\end{pmatrix}\begin{pmatrix}1 & 1\end{pmatrix} = \begin{pmatrix}1 & 1 \\ 0 & 0\end{pmatrix}.
$$

逐列 softmax 得 `A_0 = [[0.5,0.5],[0.5,0.5]]`。`O_h^0 = A_0 V_h^0 = [[0.5·1+0.5·3],[0.5·1+0.5·3]] = [[2],[2]]`。

**頭 1**：

$$
S_1 = \begin{pmatrix}0\\1\end{pmatrix}\begin{pmatrix}1 & 1\end{pmatrix} = \begin{pmatrix}0 & 0 \\ 1 & 1\end{pmatrix},\quad A_1 = [[0.5,0.5],[0.5,0.5]].
$$

`O_h^1 = A_1 V_h^1 = [[0.5·2+0.5·4],[0.5·2+0.5·4]] = [[3],[3]]`。

**合併**：`O = [[2,3],[2,3]]`，形狀 `(B,T,D) = (1,2,2)`。

**16.2** (a) 需要改：

- `self.W_V = rng.normal(0, s, (D, D_v))`，新增 `D_v` 參數。
- `self.b_V = np.zeros(D_v)`。
- `self.W_O = rng.normal(0, 1/np.sqrt(H*D_v), (H*D_v, self.D_out))`。
- `self.b_O = np.zeros(self.D_out)`（不變）。
- `merge_heads` 的檢查改為 `Y.shape[-1] == dv`，輸出維度為 `H*dv`。
- `forward` 中 `Kh` 的拆分仍按 `dh`（`D/H`），但 `Vh` 的拆分按 `dv`；**注意** `split_heads` 目前寫死 `self.dh`，要拆成兩個小函式或加參數。
- `O_h` 形狀 `(B,H,T,dv)`，合併輸出 `(B,T,H·dv)`。
- `Y = O @ W_O`，`W_O` 的第一維是 `H*dv`。

(b) `V_h.shape == (B,H,T,dv)`。

(c) 合併函式需要知道 `H` 決定何時停止拼接、`dv` 決定每段多長；只知道總維度 `H*dv` 無法還原「每段是 dv 維」的切割。

**16.3** 錯誤版本得到：

- `X.transpose(0,2,1)` 形狀 `(1,4,2)`，內容：`[[1,5],[2,6],[3,7],[4,8]]`（每列是一個 `(T=2)` 的向量）。
- `reshape(1,2,2,2)` 後 `Xh[0,0,:,:] = [[1,5],[2,6]]`，`Xh[0,1,:,:] = [[3,7],[4,8]]`。

正確版本（`split_heads`）得到：`Xh[0,0,:,:] = [[1,2],[5,6]]`，`Xh[0,1,:,:] = [[3,4],[7,8]]`。

**差異**：錯誤版本把「每個 token 的前兩維」與「時間軸」交換了。當下游只檢查形狀時，錯誤不會浮現；但 `merge` 回去得到的張量會是「按位置閱讀 `[T,H,dh]`」的錯誤順序，與拆分不互逆。

一句話：**「形狀對但語意錯」發生在對同一個張量先做軸置換再做 reshape（或反之），順序與定義相反的情況；此時元素總數正確，但每個元素在軸上的鄰居關係已被打亂**。

**16.4** 讀者應執行：

```python
import torch, numpy as np
torch.manual_seed(0)
B, T, D, H = 2, 3, 8, 2
m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0)
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
W = m.in_proj_weight.detach().numpy()  # (3D, D)，切分為 Q,K,V 三塊
```

`in_proj_weight` 的切分順序是 PyTorch 的實作約定，須在該版本對照；不要假設與本章一致。切分後複製到 `MultiHeadAttention` 的四個矩陣與 `out_proj`，在相同 `X`（float64）下比較。

**預期**：

- float64 下若切分正確，最大絕對誤差在 `1e-10` 量級或更小（取決於浮點累加順序）。
- 若看到 `> 1e-3`，先檢查切分順序與 `(B,H,T,dh)` vs `(B*H,T,dh)` 轉換。
- 若 `W_O` 的行與 `merge` 的輸出維度對不上（例如 PyTorch 內部按 `(B*H,T,dv)` 順序拼接），會產生「同一形狀但語意不同」的錯誤，現象與反例 16.3 相似。

## 本章小結

- 多頭注意力把 `D` 維特徵沿最後一維切成 `H` 段，每段 `dh=D/H` 維，分別做縮放點積注意力，再合併回 `D` 維。要求 `D % H == 0`。
- 定義 16.1 給出拆分與合併的兩步複合（reshape 與 transpose），命題 16.1 證明兩者互逆。梯度通過這兩層時僅做軸重排，不引入新的數值運算。
- softmax 沿最後一維（key 軸）；縮放因子為 `√dh`，不是 `√D`。遮罩在 softmax 前施加，本卷約定布林 True=允許；全遮罩列必須顯式拒絕，避免 NaN 擴散。
- 形狀契約：`Q_h,K_h,V_h ∈ (B,H,T,dh)`，`S,A ∈ (B,H,T,T)`，`O_h ∈ (B,H,T,dh)`，合併後 `(B,T,D)`。
- 多頭不是「可解釋性裝置」，頭與語意角色之間沒有必然對應；任何此類主張需要實驗支持。
- 本章所有數值結果均為預期與手算推導，未在本寫作環境執行；讀者須依賴自己的 NumPy 或 PyTorch 環境驗證。

## 參考來源

- [N1] Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762
- [N2] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, https://arxiv.org/abs/2106.09685
- [N3] PyTorch 2.14 `scaled_dot_product_attention` API, https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
- [N4] NumPy broadcasting 使用指南（待逐條核對）, https://numpy.org/doc/stable/user/basics.broadcasting.html
- [N5] *Dive into Deep Learning*（延伸入口，未逐章核對）, https://d2l.ai/
- [N6] PyTorch reproducibility（延伸入口，未逐條核對）, https://docs.pytorch.org/docs/stable/notes/randomness.html

**來源說明**：2026-10-06 取得 N1、N2 摘要頁，未完整閱讀論文；N3 已取得明確 2.14 API 全文並核對「mask True=參與、evaluation 須 dropout_p=0、矩形因果遮罩左上對齊」。N4–N6 目前僅為待核對延伸入口，**不可宣稱已查證其內容**。本章未執行任何外部程式、未下載模型或語料、未安裝任何套件；所有數值皆為手算或預期。