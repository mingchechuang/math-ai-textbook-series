## 審稿結論

本次重新由公式與索引自行核算，未執行程式、未使用外部工具，也不以舊審稿作證據。第13、14、15、17、18章的核心數學大致成立；第16章仍未完成本卷要求的多頭注意力梯度驗收，且有 shape、mask、$D_v/d_v$、PyTorch 轉接與來源聲明等實質錯誤。現稿與上一輪所見問題相比沒有可辨識的修復，因此仍需修訂。

重算確認：

- 第13章查表梯度與共享矩陣兩路梯度正確。
- 第14章 softmax VJP、$dQ,dK,dV$ 與手算數值正確。
- 第15章以絕對位置建立 causal mask 的規則正確。
- 第16章 split／merge 互逆及前向手算正確。
- 第17章 RoPE 相對位置公式與旋轉方向正確。
- 第18章 LN、RMSNorm 與殘差梯度正確。

以下只列可定位的實質問題。

---

# 必須修訂

## 1. 第16章跨章依賴座標錯誤

### 原句

> 「**先備知識**：前一章的縮放點積注意力……」

### 原因

第16章前一章是第15章，縮放內積注意力與完整反向傳播則在第14章。第16章實際同時依賴第14章的 SDPA 梯度和第15章的 mask 契約。

### 最小修法

改為：

> 先備知識：第14章的縮放內積注意力與反向傳播、第15章的遮罩及全遮罩拒絕策略……

---

## 2. 第16章沒有完成多頭注意力的梯度驗收

### 原句

> 「推導合併與拆分的反向接口。」

程式只提供：

```python
def split_backward(...)
def merge_backward(...)
```

### 原因

第14章只驗證已給定 $Q,K,V$ 的單頭注意力梯度。多頭模組還有：

$$
Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V,
$$

以及：

$$
Y=OW_O+b_O.
$$

完整反向必須包含：

$$
dW_O=O_{\rm flat}^{\mathsf T}dY_{\rm flat},
\qquad
db_O=\sum_{b,t}dY_{b,t,:},
$$

$$
dO=dYW_O^{\mathsf T},
$$

以及注意力返回的 $dQ,dK,dV$ 經逆重排後：

$$
dW_Q=X_{\rm flat}^{\mathsf T}dQ_{\rm flat},
\quad
dW_K=X_{\rm flat}^{\mathsf T}dK_{\rm flat},
\quad
dW_V=X_{\rm flat}^{\mathsf T}dV_{\rm flat},
$$

最後必須累加三條輸入路徑：

$$
dX=dQW_Q^{\mathsf T}+dKW_K^{\mathsf T}+dVW_V^{\mathsf T}.
$$

目前沒有這些梯度，也沒有對 $X,W_Q,W_K,W_V,W_O$ 的有限差分，因此尚未滿足本卷「多頭attention的形狀／mask／梯度」驗收。

### 最小修法

保留現有前向，新增完整 backward 與 cache。以 `B=1,T=2,D=4,H=2`、float64、非對稱固定上游梯度，對輸入、四個權重及偏置做中央差分；另加入部分硬遮罩梯度案例。不能只用全一上游，因對稱方向可能掩蓋 transpose 或漏加路徑。

---

## 3. 第16章四維 mask 未檢查 batch/head 軸

### 原句

```python
if m.shape[-2:] != (T, T):
    raise ValueError(...)
if np.any(np.all(~m, axis=-1)):
    raise ValueError(...)
S = np.where(m, S, -np.inf)
```

### 原因

若 `S.shape == (B,H,T,T)`，合理的 mask 前導軸可能是：

- `(1,1,T,T)`；
- `(B,1,T,T)`；
- `(1,H,T,T)`；
- `(B,H,T,T)`。

現稿只檢查最後兩軸。像 `(B,H+1,T,T)` 會通過章內檢查，最後由 `np.where` 丟出底層廣播錯誤。全遮罩檢查也應作用於廣播後的每個 $(b,h,q)$ 列。

### 最小修法

```python
if m.shape[0] not in (1, B) or m.shape[1] not in (1, self.H):
    raise ValueError("mask 的 batch/head 軸不可廣播至 (B,H)")
m = np.broadcast_to(m, S.shape)
if np.any(np.all(~m, axis=-1)):
    raise ValueError("存在全遮罩 query 列")
```

補前述正常廣播與錯誤 head 軸測試。

---

## 4. 第16章 mask 非 NumPy 陣列時會出現 `AttributeError`

### 原句

```python
m = mask
if m.dtype != np.bool_:
```

### 原因

Python list 沒有 `.dtype`，因此會拋出非契約的 `AttributeError`。第14章已明確要求 mask 是 NumPy 陣列，第16章應一致。

### 最小修法

```python
if not isinstance(mask, np.ndarray):
    raise TypeError("mask 必須是 NumPy 陣列")
if mask.dtype != np.bool_:
    raise TypeError("mask 必須為布林；True=允許")
```

加入 list mask 故障測試。

---

## 5. 第15章靜默將任意數值 mask 轉成 bool

### 原句

```python
allowed = np.asarray(allowed, dtype=bool)
```

另有：

```python
query_is_valid = np.asarray(query_is_valid, dtype=bool)
valid_targets = np.asarray(valid_targets, dtype=bool)
```

### 原因

任意非零浮點值都會變成 `True`。若把加性 attention bias 誤傳為 `allowed`，程式不會拒絕。第14與16章卻明確拒絕非布林 mask，形成跨章介面衝突。

### 最小修法

先保留原 dtype：

```python
allowed = np.asarray(allowed)
if allowed.dtype != np.bool_:
    raise TypeError("allowed 必須是布林；True=允許")
```

其他兩個 mask 同理，並增加浮點 mask 故障測試。

---

## 6. 第16章參數數量公式缺少 $D_{\text{out}}$

### 原句

> 「多頭用 `4(D^2 + D)` 個參數（含偏置）」

同章又定義：

> `W_O ∈ ℝ^{D×D_out}`

### 原因

一般總數是：

$$
3(D^2+D)+DD_{\text{out}}+D_{\text{out}}.
$$

只有 $D_{\text{out}}=D$ 時才等於：

$$
4(D^2+D).
$$

### 最小修法

先列一般式，再說 $D_{\text{out}}=D$ 時化為 $4(D^2+D)$。

---

## 7. 習題16.2混淆總 value 寬度 $D_v$ 與每頭寬度 $d_v$

### 原句

題目：

> `W_V ∈ ℝ^{D×D_v}`、`W_O ∈ ℝ^{H·dv×D_out}`

解答：

> `self.W_V = rng.normal(0, s, (D, D_v))`

> `self.W_O = rng.normal(0, 1/np.sqrt(H*D_v), (H*D_v, self.D_out))`

> `V_h.shape == (B,H,T,dv)`

### 原因

若 $D_v$ 是總投影寬度，則必須：

$$
D_v=Hd_v,
$$

且：

$$
W_V\in\mathbb R^{D\times D_v},
\qquad
W_O\in\mathbb R^{D_v\times D_{\text{out}}}.
$$

現稿的 `W_O.shape == (H*D_v,D_out)` 會把頭數再乘一次。初始化分母 `sqrt(H*D_v)` 也因此基於錯誤 fan-in。

若 `dv` 是每頭寬度，則應直接寫：

$$
W_V\in\mathbb R^{D\times(Hd_v)},
\quad
b_V\in\mathbb R^{Hd_v},
\quad
W_O\in\mathbb R^{Hd_v\times D_{\text{out}}}.
$$

### 最小修法

建議刪去 `D_v`，全題只用每頭 `dv`：

```python
W_V.shape == (D, H * dv)
b_V.shape == (H * dv,)
Vh.shape == (B, H, T, dv)
W_O.shape == (H * dv, D_out)
```

若保留 `D_v`，就明定 `D_v=H*dv` 並檢查整除，不可同時把 $D_v$ 當每頭和總寬度。

---

## 8. 第16章 PyTorch 交叉驗證片段 dtype 不一致

### 原句

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
)
X = torch.randn(B, T, D, dtype=torch.float64)
```

### 原因

模組參數預設通常為 float32，而輸入是 float64。直接呼叫會因 dtype 不一致而失敗，無法得到所稱 float64 比較。

### 最小修法

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
).double()
m.eval()
```

並在 `torch.no_grad()` 下執行 forward。仍只能寫預期結果。

---

## 9. PyTorch 與本卷權重座標轉接未說明轉置

### 原句

> 「把 PyTorch 的 `in_proj_weight`、`in_proj_bias`、`out_proj.weight`、`out_proj.bias` 複製到 NumPy 實作」

### 原因

本卷採：

$$
Y=XW,\qquad W\in\mathbb R^{D_{\text{in}}\times D_{\text{out}}},
$$

PyTorch 線性權重通常以 `(out_features,in_features)` 儲存。將 `in_proj_weight` 分成 Q、K、V 三塊後，必須轉置才可賦給本卷的 `W_Q,W_K,W_V`；`out_proj.weight` 亦須轉置。只說「複製」會得到 shape 相同但座標語義錯誤的矩陣。

### 最小修法

解答明列：

```python
Wq_t, Wk_t, Wv_t = np.split(
    m.in_proj_weight.detach().cpu().numpy(), 3, axis=0
)
att.W_Q = Wq_t.T.copy()
att.W_K = Wk_t.T.copy()
att.W_V = Wv_t.T.copy()
att.W_O = m.out_proj.weight.detach().cpu().numpy().T.copy()
```

偏置按三段切分，不轉置。

---

## 10. 第16章未訓練的手算例被稱為「學到了」與「頭塌縮」

### 原句

> 「兩頭學到了不同的『模式』。」

> 「這正是『頭塌縮』的小型演示。」

### 原因

例中權重固定為身份矩陣，沒有 loss、optimizer 或訓練 loop。只能說固定特徵分塊產生不同結果，不能說「學到了」。人工構造兩頭輸出相同也不能直接稱為訓練後頭塌縮。

### 最小修法

改成：

> 兩個固定特徵子空間在此輸入上產生不同注意力結果；本例沒有訓練，不能推論各頭學得語義模式。

第二句改為「兩頭輸出相同的退化構造」。

---

## 11. 第16章的幾何敘述與本卷 row-vector 權重約定不精確

### 原句

> 「如果把 `W_Q` 的行看成『查詢基底』……」

### 原因

本卷採 $Q=XW_Q$，其中每個輸出 query 座標是 $X$ 與 $W_Q$ 的一個 column 做內積。若要把參數方向解釋為輸出座標的投影方向，較直接對應的是 $W_Q$ 的 columns，而不是 rows。rows 描述各輸入特徵對全部輸出座標的係數。把 rows 稱作查詢基底會與本卷固定矩陣方向衝突。

此外，多頭並不是把原始 $D$ 維輸入空間預先分成固定直和；是完整 $W_Q$ 先投影到 $D$ 維輸出座標，再把輸出座標分塊。各頭對輸入空間的投影方向不必構成正交直和。

### 最小修法

改成：

> 在本卷 $Q=XW_Q$ 的 row-vector 約定下，$W_Q$ 的各 columns 決定 query 輸出座標的投影方向；投影輸出的最後一軸再分成 $H$ 個座標區塊。這不要求各頭在輸入空間正交。

---

## 12. 第16、17章有無紀錄支持的來源查閱聲明

### 原句

第16章：

> 「2026-10-06 取得 N1、N2 摘要頁……N3 已取得明確 2.14 API 全文並核對……」

第17章：

> 「依題目附註，只取得摘要頁……」

> 「已依題目附註核對……」

### 原因

提供內容只有來源清單與 URL，沒有特定日期存取紀錄或逐條核對紀錄。不能自行聲稱在某日取得、閱讀或核對文件。

### 最小修法

刪除日期與「已取得／已核對」措辭，改為：

> N3 為題目提供的 PyTorch 2.14 API 來源入口；實際使用仍須按安裝版本核對。此連結不是本機安裝、執行或來源查閱證據。

---

# 其他應修問題

## 13. 第16章未拒絕非有限輸入與參數

### 原句

```python
def forward(self, X, mask=None):
    if X.ndim != 3 or X.shape[-1] != self.D:
```

### 原因

`NaN` 或 `inf` 可直接進入 scores 和 softmax。第14、15、17、18章都有非有限值策略，第16章不應例外。

### 最小修法

檢查 `X` 及全部參數有限；softmax 前再檢查允許 scores 有限。增加 NaN 輸入及 inf 權重故障測試。

---

## 14. 第16章 `D,H,D_out` 的型別契約不足

### 原句

```python
if D <= 0 or H <= 0:
...
self.D_out = D if D_out is None else D_out
```

### 原因

Python bool 是 int 子類別；`True` 可能通過。`D_out=0` 會得到空輸出軸，浮點值則到建立陣列時才失敗。

### 最小修法

明確要求非 bool 的正整數，套用於 `D,H,D_out`。

---

## 15. 第16章 $H=1$ 的說明自相矛盾

### 原句

> 「模型與單頭縮放點積注意力等價（權重不同）。」

### 原因

若參數、輸入與 mask 相同，注意力權重也相同；若參數不同，就不是同一數學退化。

### 最小修法

改為：

> 當參數與遮罩相同時，$H=1$ 退化為帶 QKV 及輸出投影的單頭縮放內積注意力。

---

# 通過部分

第13章的重複索引累加、共享 embedding 梯度與資料切分契約正確。第14章的單頭注意力梯度、非方形 shape 與固定遮罩有限差分設計合理。第15章的絕對位置 causal 規則與第17章的 cache 位置偏移一致。第17章的 RoPE 證明、相鄰偶奇配對及共同平移測試正確。第18章的 reduction 軸、$\epsilon$ 位置、共享 norm 參數梯度及殘差兩路累加正確。

但第16章作為多頭注意力整合章，目前尚缺完整反向與梯度測試；其 value 維度符號、mask 前導軸與跨框架轉接也未閉合。這些是本部核心驗收問題，不能以第14章已有單頭梯度取代。

VERDICT: REVISE