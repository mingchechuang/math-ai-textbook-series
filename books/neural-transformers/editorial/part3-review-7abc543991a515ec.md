## 審稿範圍與自行重算

本次只審第13至18章，重新核對跨章座標、符號、shape／broadcast／reduction、梯度、mask／cache／position、來源聲明與自足程式。未執行程式、未使用外部工具，也不以先前審稿結論作為證據。

自行重算後，以下部分成立：

- 第13章的 embedding 查表等價、重複索引 scatter-add、共享輸入／輸出矩陣的兩路梯度均正確。
- 第14章的方差命題在「所有相關分量相互獨立、零均值、單位方差」條件下成立；softmax VJP 與 $dQ,dK,dV$ 公式正確，兩個手算例主要數值亦正確。
- 第15章以 $k_j\le q_i$ 建立絕對位置 causal mask 的規則正確，$(T_q,T_k)=(2,6)$ 的 cache 手算正確。
- 第16章 split／merge 的索引映射與兩個前向手算正確。
- 第17章 RoPE 的旋轉方向與
  $$
  (R_pq)^{\mathsf T}(R_rk)=q^{\mathsf T}R_{r-p}k
  $$
  相符；共同平移及 cache 起始位置的測試方向正確。
- 第18章 LN、RMSNorm、殘差與 FFN 的主要前反傳公式正確。

但第16章仍有多個會阻止本部核心驗收的問題，另有跨章 mask 型別不一致及無證據的來源查閱聲明。以下按嚴重度列出。

---

# 必須修訂

## 1. 第16章跨章依賴座標錯誤

### 逐字原句

> 「**先備知識**：前一章的縮放點積注意力、固定張量軸約定 `B/T/D/H/dh`、穩定 softmax 的減最大值技巧、布林遮罩 True=允許的約定、矩陣微分的基本鏈式法則。」

### 原因

第16章的前一章是第15章「Padding、因果遮罩與未來資訊洩漏」，縮放內積注意力與完整 $Q,K,V$ 梯度則在第14章。第16章同時依賴：

- 第14章：SDPA 前向、softmax VJP、$dQ,dK,dV$；
- 第15章：mask 語義、全遮罩拒絕、padding／causal 區分。

因此「前一章的縮放點積注意力」是可定位的依賴錯誤。

### 最小修法

改成：

> 先備知識：第14章的縮放內積注意力與反向傳播、第15章的布林遮罩及全遮罩拒絕策略，以及固定張量軸約定……

---

## 2. 第16章未完成「多頭 attention 的梯度」驗收

### 逐字原句

> 「推導合併與拆分的反向接口。」

以及只提供：

```python
def split_backward(...)
def merge_backward(...)
```

又稱：

> 「有限差分在『全 1 上游梯度』下應為精確恒等。」

### 原因

本卷明定的階段驗收包含「多頭attention的形狀／mask／梯度」。第14章只驗證已給定 $Q,K,V$ 的單頭 SDPA 梯度；第16章新增的計算圖還包括：

$$
Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V,
$$

多頭 split／merge 與：

$$
Y=OW_O+b_O.
$$

完整反向至少應包含：

$$
dW_O=O_{\rm flat}^{\mathsf T}dY_{\rm flat},
\qquad
db_O=\sum_{b,t}dY_{b,t,:},
$$

$$
dO=dYW_O^{\mathsf T},
$$

以及注意力反向得到 $dQ,dK,dV$ 後：

$$
dW_Q=X_{\rm flat}^{\mathsf T}dQ_{\rm flat},
\quad
dW_K=X_{\rm flat}^{\mathsf T}dK_{\rm flat},
\quad
dW_V=X_{\rm flat}^{\mathsf T}dV_{\rm flat},
$$

偏置梯度沿 $(B,T)$ 求和，輸入梯度則必須累加三路：

$$
dX=dQW_Q^{\mathsf T}+dKW_K^{\mathsf T}+dVW_V^{\mathsf T}.
$$

目前沒有任何程式驗證三路累加、投影參數梯度或 mask 下梯度。split／merge 互逆測試不能取代完整 MHA 梯度驗收。

只使用「全1上游梯度」也不足，因對稱方向可能掩蓋 transpose 或路徑漏加錯誤。

### 最小修法

不必重寫整章，只需：

1. 前向 cache 保存 `X,Qh,Kh,Vh,A,Oh,O`；
2. 加入完整 `backward(dY, cache)`；
3. 使用 `B=1,T=2,D=4,H=2`、float64、非對稱固定 `dY`；
4. 對 $X,W_Q,W_K,W_V,W_O$ 與偏置做中央有限差分；
5. 再加一個部分硬遮罩案例，核對禁止位置的權重與 $dS$ 為零。

所有結果仍應寫為預期，不可聲稱已通過。

---

## 3. 第16章四維 mask 的 batch/head 軸沒有契約

### 逐字原句

```python
if m.shape[-2:] != (T, T):
    raise ValueError(...)
if np.any(np.all(~m, axis=-1)):
    raise ValueError(...)
S = np.where(m, S, -np.inf)
```

### 原因

只檢查最後兩軸，沒有核對前兩軸能否合法對應 $(B,H)$。例如 `S.shape=(2,2,T,T)`，而 mask 是 `(2,3,T,T)`，章內檢查會放行，最後由 `np.where` 拋出底層廣播錯誤。這不符合正文所稱的明確 shape 契約。

合理可接受的四維形狀應明列為：

- `(1,1,T,T)`；
- `(B,1,T,T)`；
- `(1,H,T,T)`；
- `(B,H,T,T)`。

全遮罩檢查也應在 mask 廣播到實際 `S.shape` 後，逐 $(b,h,q)$ 檢查。

### 最小修法

加入：

```python
if m.shape[0] not in (1, B) or m.shape[1] not in (1, self.H):
    raise ValueError("mask 的 batch/head 軸不可廣播至 (B,H)")
m = np.broadcast_to(m, S.shape)
if np.any(np.all(~m, axis=-1)):
    raise ValueError("存在全遮罩 query 列")
```

並補 `(B,1,T,T)`、`(1,H,T,T)` 正常測試及 `(B,H+1,T,T)` 故障測試。

---

## 4. 第16章 mask 非陣列時會拋出非契約的 `AttributeError`

### 逐字原句

```python
m = mask
if m.dtype != np.bool_:
    raise ValueError(...)
```

### 原因

若 `mask` 是 Python list，`m.dtype` 不存在。程式會拋 `AttributeError`，而不是明確的 mask 型別錯誤。第14章已採「必須是 NumPy array」的介面，第16章應一致。

### 最小修法

```python
if not isinstance(mask, np.ndarray):
    raise TypeError("mask 必須是 NumPy 陣列")
m = mask
if m.dtype != np.bool_:
    raise TypeError("mask 必須為布林；True=允許")
```

加入 list mask 故障測試。

---

## 5. 第15章把任意數值 mask 靜默轉成布林，與第14、16章衝突

### 逐字原句

```python
allowed = np.asarray(allowed, dtype=bool)
```

另有：

```python
query_is_valid = np.asarray(query_is_valid, dtype=bool)
valid_targets = np.asarray(valid_targets, dtype=bool)
```

### 原因

例如 `0.2`、`-1.0` 都會被轉成 `True`。若呼叫端誤把浮點加性偏置傳給 `allowed`，程式不會拒絕，而會改變 mask 語義。第14、16章明確拒絕非布林 mask，因此本卷介面目前不一致。

### 最小修法

先不指定 dtype：

```python
allowed = np.asarray(allowed)
if allowed.dtype != np.bool_:
    raise TypeError("allowed 必須是布林；True=允許")
```

`query_is_valid` 與 `valid_targets` 同理。增加浮點 0/1 mask 及一般浮點 mask 的故障測試。

---

## 6. 第16章的參數數量公式只在 $D_{\text{out}}=D$ 時成立

### 逐字原句

> 「**參數數量**：多頭用 `4(D^2 + D)` 個參數（含偏置），與單頭在相同維度下相同。」

同章又定義：

> `W_O ∈ ℝ^{D×D_out}`

並測試：

> 「`D_out ≠ D`」

### 原因

一般總參數數量是：

$$
3(D^2+D)+(DD_{\text{out}}+D_{\text{out}})
=3D^2+3D+DD_{\text{out}}+D_{\text{out}}.
$$

只有 $D_{\text{out}}=D$ 時才化為：

$$
4D^2+4D=4(D^2+D).
$$

### 最小修法

把一般式與特殊式分開寫，並說明在總投影寬度固定為 $D$ 時，改變頭數 $H$ 不改變參數總數。

---

## 7. 習題16.2的 $D_v$、$d_v$ 定義互相矛盾

### 逐字原句

題目寫：

> `W_V ∈ ℝ^{D×D_v}`、`W_O ∈ ℝ^{H·dv×D_out}`

解答又寫：

> `self.W_V = rng.normal(0, s, (D, D_v))`

> `self.b_V = np.zeros(D_v)`

> `V_h.shape == (B,H,T,dv)`

### 原因

這裡混用了總 value 投影寬度 $D_v$ 與每頭寬度 $d_v$，但沒有定義兩者關係。

若 `W_V` 的輸出總寬度是 $D_v$，split 後應要求：

$$
D_v=H d_v,
$$

並檢查 $D_v$ 可被 $H$ 整除。此時：

$$
W_V\in\mathbb R^{D\times D_v},
\quad
V_h\in\mathbb R^{B\times H\times T\times d_v},
\quad
d_v=D_v/H,
$$

合併後寬度是 $D_v$，所以：

$$
W_O\in\mathbb R^{D_v\times D_{\text{out}}},
$$

不是另寫成 $H d_v$ 而不說兩者相等。

另一種定義是直接把 `dv` 當每頭寬度，此時必須寫：

$$
W_V\in\mathbb R^{D\times(Hd_v)}.
$$

現稿把兩種約定混在一起，讀者無法按「最少行數」得到唯一實作。

### 最小修法

二選一並全章一致。建議採本卷固定約定：

- `dv` 表示每頭 value 維度；
- `W_V.shape == (D, H*dv)`；
- `b_V.shape == (H*dv,)`；
- `V_h.shape == (B,H,T,dv)`；
- merge 後 `(B,T,H*dv)`；
- `W_O.shape == (H*dv,D_out)`。

若保留 `D_v`，則明定 `D_v=H*dv` 並檢查整除。

---

## 8. 第16章 PyTorch 交叉驗證有 dtype 錯誤，且缺少權重轉置說明

### 逐字原句

```python
m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0)
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
```

並稱：

> 「float64 下若切分正確，最大絕對誤差在 `1e-10` 量級或更小」

### 原因

模組參數預設通常是 float32，輸入卻是 float64。直接執行比較會因 dtype 不一致失敗。若要討論 float64 誤差，模組也必須 `.double()`。

此外，本卷採：

$$
Y=XW,\qquad W\in\mathbb R^{D_{\text{in}}\times D_{\text{out}}},
$$

而 PyTorch `Linear` 類權重通常以 `(out_features,in_features)` 儲存。`in_proj_weight` 切成三個 `(D,D)` 區塊後，複製到本卷 NumPy `W_Q,W_K,W_V` 時必須轉置；`out_proj.weight` 亦須轉置。原解答只說「複製」，沒有完成跨框架座標轉接。

### 最小修法

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
).double()
m.eval()
```

並明示：

```python
Wq_t, Wk_t, Wv_t = np.split(
    m.in_proj_weight.detach().cpu().numpy(), 3, axis=0
)
att.W_Q = Wq_t.T.copy()
att.W_K = Wk_t.T.copy()
att.W_V = Wv_t.T.copy()
att.W_O = m.out_proj.weight.detach().cpu().numpy().T.copy()
```

偏置按 Q/K/V 三段切分，不需轉置。仍只能寫預期，不能聲稱誤差已量得。

---

## 9. 第16章把未訓練例子稱為「學到了」及「頭塌縮」

### 逐字原句

> 「兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的『模式』。」

> 「這正是『頭塌縮』的小型演示。」

### 原因

手算例使用固定身份投影，沒有 loss、optimizer 或訓練 loop。它只能證明固定特徵分塊產生不同或相同數值，不能證明「學到了」。人工令兩頭輸入相同，也不能直接等同訓練後頭塌縮。

### 最小修法

改為：

> 兩個固定特徵子空間在此輸入上產生不同注意力結果；本例沒有訓練，不能據此宣稱各頭學到不同語義模式。

第二句改成：

> 這是兩頭輸出相同的退化構造，不是已觀察到的訓練後頭塌縮。

---

## 10. 第16、17章有無證據支持的來源取得與核對聲明

### 第16章逐字原句

> 「2026-10-06 取得 N1、N2 摘要頁……N3 已取得明確 2.14 API 全文並核對……」

### 第17章逐字原句

> 「依題目附註，只取得摘要頁……」

> 「已依題目附註核對 `True` 參與注意……」

### 原因

提供資料只有來源條目與 URL，沒有特定日期的存取紀錄，也沒有「已核對全文」的證據。章稿不得自行虛構查閱行為。API 敘述即使正確，也不能據此反推作者確實在某日取得或逐條核對文件。

### 最小修法

刪除日期及「已取得／已核對」措辭，改成：

> N3 為題目提供的 PyTorch 2.14 API 來源入口；使用時仍須依實際版本核對。此連結不是本機安裝、來源查閱或程式執行證據。

---

# 應一併修正

## 11. 第16章沒有拒絕非有限輸入與參數

### 逐字原句

```python
def forward(self, X, mask=None):
    if X.ndim != 3 or X.shape[-1] != self.D:
        ...
```

### 原因

含 `NaN` 或 `inf` 的 $X$、權重或偏置會流入 scores 及 softmax。第14、15、17、18章都明確拒絕非有限數，第16章不應成為例外。

### 最小修法

檢查 `X` 與所有參數有限，並在 softmax 前檢查允許位置 scores 有限。增加 NaN 輸入與 inf 權重故障測試。

---

## 12. 第16章的正整數型別契約不足

### 逐字原句

```python
if D <= 0 or H <= 0:
...
self.D_out = D if D_out is None else D_out
```

### 原因

Python bool 是 int 子類別，`D=True`、`H=True` 可能通過。`D_out=0` 會建立空輸出軸，浮點或負數則在較晚的 NumPy shape 操作才報錯。第17章已正確拒絕 bool 冒充整數，第16章應一致。

### 最小修法

增加「非 bool 的正整數」檢查，套用於 `D,H,D_out`。

---

## 13. 第16章的 $H=1$ 說明自相矛盾

### 逐字原句

> 「模型與單頭縮放點積注意力等價（權重不同）。」

### 原因

若參數、輸入、mask 與縮放相同，$H=1$ 的注意力權重也相同；若權重參數不同，則不是同一模型的數學退化比較。

### 最小修法

改為：

> 當參數與遮罩相同時，$H=1$ 退化為帶 QKV 及輸出投影的單頭縮放內積注意力。

---

# 通過部分與總評

第13章的共享梯度、第14章的 SDPA 梯度、第15章的絕對位置 causal mask、第17章的 RoPE 相對位置推導、第18章的 norm 與殘差梯度均無需因上述問題重寫。資料切分、有效 token 平均、PPL、padding key mask 與 loss mask 的概念也大致跨章一致。

然而，第16章是本部多頭整合的核心章，目前仍只有前向與重排反向，未完成多頭參數及輸入梯度驗收；其 mask 前導軸、PyTorch dtype／轉置及 $D_v/d_v$ 介面也尚未閉合。再加上第15章 mask 型別靜默轉換，以及第16、17章無紀錄支持的來源查閱聲明，現稿不能批准。

VERDICT: REVISE