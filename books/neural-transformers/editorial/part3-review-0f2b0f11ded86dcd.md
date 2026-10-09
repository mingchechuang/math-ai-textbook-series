## 審稿結論概述

本次重新由公式、索引與程式介面自行核算，未執行程式、未使用外部工具，也不沿用先前審稿結論作為證據。第13、14、15、17、18章的核心數學大致正確；主要阻礙仍集中在第16章的依賴描述、多頭梯度驗收、mask 前導軸、PyTorch dtype 與來源查證聲明。此外，第15章的 mask 型別處理與第14、16章不一致。

重算確認：

1. 第13章重複索引梯度及共享 embedding 的兩路梯度正確。
2. 第14章 SDPA 的 softmax VJP、$dQ,dK,dV$ 公式與手算值正確。
3. 第15章矩形 causal mask 的絕對位置比較正確。
4. 第16章 split／merge 互逆與手算前向正確，但完整多頭梯度尚未實作及驗收。
5. 第17章 RoPE 的旋轉符號、相對位置內積與 cache 起點原理正確。
6. 第18章 LN、RMSNorm、殘差及 FFN 的主要前反傳公式正確。

以下只列可定位的實質問題及最小修法。

---

# 一、必須修訂

## 1. 第16章跨章依賴指向錯誤

### 原句

> 「**先備知識**：前一章的縮放點積注意力、固定張量軸約定……」

### 原因

第16章的前一章是第15章「Padding、因果遮罩與未來資訊洩漏」；縮放內積注意力與完整 $Q,K,V$ 反傳在第14章。這不只是章號風格：第16章省略了 SDPA backward，實際依賴第14章的公式，同時又依賴第15章的 mask 與 cache 語義。寫成「前一章」會造成錯誤依賴座標。

### 最小修法

改成：

> 先備知識：第14章的縮放內積注意力與反向傳播、第15章的布林遮罩與全遮罩拒絕策略，以及固定張量軸約定……

不需要虛構其他卷章號或可匯入模組。

---

## 2. 第16章沒有完成多頭注意力的梯度驗收

### 原句

> 「推導合併與拆分的反向接口。」

程式只提供：

```python
def split_backward(...)
def merge_backward(...)
```

並稱：

> 「有限差分在『全 1 上游梯度』下應為精確恒等。」

### 原因

本卷明列的階段驗收是「多頭attention的形狀／mask／梯度」。第14章驗證的是已給定 $Q,K,V$ 的單頭 SDPA 梯度；第16章新加入的計算圖還包括：

$$
Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V,
$$

split、各頭注意力、merge，以及

$$
Y=OW_O+b_O.
$$

完整反向至少應包含：

$$
dW_O=O_{\rm flat}^{\mathsf T}dY_{\rm flat},\qquad
db_O=\sum_{b,t}dY_{b,t,:},
$$

以及從注意力返回的 $dQ,dK,dV$ 經 split 的逆重排後：

$$
dW_Q=X_{\rm flat}^{\mathsf T}dQ_{\rm flat},
$$

Q、K、V 三路皆同理，最後輸入梯度必須累加：

$$
dX=dQW_Q^{\mathsf T}+dKW_K^{\mathsf T}+dVW_V^{\mathsf T}.
$$

目前程式沒有這些梯度，也沒有對 $X,W_Q,W_K,W_V,W_O$ 的有限差分。只測 split／merge 不能抓到漏加 Q/K/V 路徑、輸出投影轉置錯誤或遮罩下分數梯度錯誤。

「全1上游梯度」也不宜作唯一方向；對稱目標可能令某些錯誤互相抵消。

### 最小修法

在第16章現有類別中加入完整 `backward`，前向 cache 至少保存 `X,Qh,Kh,Vh,A,Oh,O`。測試使用小型 `B=1,T=2,D=4,H=2`、float64 和非對稱固定上游梯度，對：

- $X$；
- $W_Q,W_K,W_V,W_O$；
- 四組偏置；

做中央差分抽樣或逐元素核對。另加部分硬遮罩案例，確認禁止位置權重及相應 $dS$ 為零。測試只能寫預期，不可聲稱已通過。

---

## 3. 第16章四維 mask 的 batch/head 軸未驗證

### 原句

```python
if m.shape[-2:] != (T, T):
    raise ValueError(...)
if np.any(np.all(~m, axis=-1)):
    raise ValueError(...)
S = np.where(m, S, -np.inf)
```

### 原因

只檢查最後兩軸，沒有定義四維 mask 的前兩軸。假設 `S.shape == (2,2,T,T)`：

- `(1,1,T,T)` 可合理廣播；
- `(2,1,T,T)` 可合理廣播；
- `(1,2,T,T)` 可合理廣播；
- `(2,2,T,T)` 可逐 batch/head 指定；
- `(2,3,T,T)` 不合法。

目前最後一種會通過章內 shape 檢查，再由 `np.where` 丟出底層廣播錯誤。這與正文宣稱的明確 shape 契約不符。全遮罩檢查也應針對廣播後每個 $(b,h,q)$ 列，而不是只檢查未展開的表示。

### 最小修法

在 mask 升為四維後加入：

```python
if m.shape[0] not in (1, B) or m.shape[1] not in (1, self.H):
    raise ValueError("mask 的 batch/head 軸不可廣播至 (B,H)")
m = np.broadcast_to(m, S.shape)
if np.any(np.all(~m, axis=-1)):
    raise ValueError("存在全遮罩 query 列")
```

補測 `(B,1,T,T)`、`(1,H,T,T)` 正常案例，以及 `(B,H+1,T,T)` 故障案例。

---

## 4. 第16章 mask 非 NumPy 輸入會產生非契約錯誤

### 原句

```python
m = mask
if m.dtype != np.bool_:
    raise ValueError(...)
```

### 原因

若傳入 Python list，`m.dtype` 不存在，會拋 `AttributeError`，而不是本章定義的 mask 型別或形狀錯誤。第14章明確要求 mask 是 NumPy 陣列；第16章應採相同介面，或者先以 `np.asarray` 轉換後再嚴格檢查原 dtype。不可讓偶然的屬性錯誤替代故障契約。

### 最小修法

與第14章一致：

```python
if not isinstance(mask, np.ndarray):
    raise TypeError("mask 必須是 NumPy 陣列")
m = mask
if m.dtype != np.bool_:
    raise TypeError("mask 必須是布林；True=允許")
```

並加入 list mask 故障測試。

---

## 5. 第15章會靜默把非布林 mask 轉成 bool，與第14、16章介面衝突

### 原句

```python
allowed = np.asarray(allowed, dtype=bool)
```

### 原因

這會把任意非零浮點數轉為 `True`。例如包含 `0.2`、`-3.0` 的加性偏置若誤傳入 `allowed`，會被靜默解釋為允許位置。第14章與第16章都拒絕非布林 mask，因此同一本卷的 `True=允許` 介面目前不一致。

同樣問題出現在：

```python
query_is_valid = np.asarray(query_is_valid, dtype=bool)
valid_targets = np.asarray(valid_targets, dtype=bool)
```

若本卷的 mask 契約是布林，就不宜接受任意可轉型數值。

### 最小修法

先轉成不指定 dtype 的陣列，再檢查：

```python
allowed = np.asarray(allowed)
if allowed.dtype != np.bool_:
    raise TypeError("allowed 必須是布林；True=允許")
```

`query_is_valid`、`valid_targets` 同理。補上浮點 mask 故障測試。

---

## 6. 第16章參數數量公式忽略了 $D_{\text{out}}$

### 原句

> 「參數數量：多頭用 `4(D^2 + D)` 個參數（含偏置）……」

同章卻定義：

> `W_O ∈ ℝ^{D×D_out}`

並允許：

> 「`D_out ≠ D`」

### 原因

一般情形的參數數量為：

- Q、K、V 三個投影：$3(D^2+D)$；
- 輸出投影：$DD_{\text{out}}+D_{\text{out}}$。

總計：

$$
3D^2+3D+DD_{\text{out}}+D_{\text{out}}.
$$

只有 $D_{\text{out}}=D$ 時才等於：

$$
4D^2+4D=4(D^2+D).
$$

### 最小修法

把原句限定為 $D_{\text{out}}=D$，並先列一般公式。另可補充：在總投影寬度固定時，改變 $H$ 本身不改變參數數量。

---

## 7. 第16章 PyTorch 交叉驗證片段存在 dtype 不一致

### 原句

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
)
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
```

並預期：

> 「float64 下若切分正確，最大絕對誤差在 `1e-10` 量級或更小」

### 原因

PyTorch 模組預設參數通常為 float32，而 $X$ 是 float64。若依習題目的實際呼叫 `m(X,X,X)`，dtype 不一致會阻止比較。要討論 float64 誤差，模組參數也必須轉為 float64。

此外，片段未展示 PyTorch forward 與完整權重轉接；作為「解答」至少應包含能完成比較的必要步驟，不能只建立物件後停止。

### 最小修法

改成：

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
).double()
m.eval()
```

並加入：

```python
with torch.no_grad():
    y_torch, _ = m(X, X, X, need_weights=False)
```

轉接時還要明示 PyTorch 權重採 $(D_{\rm out},D_{\rm in})$ 儲存，而本卷採 $(D_{\rm in},D_{\rm out})$，所以切出的每塊要轉置後才可賦給 NumPy 的 `W_Q,W_K,W_V`；`out_proj.weight` 亦須轉置。這是跨框架座標轉接的核心，不應只說「複製」。

---

## 8. 第16章把未訓練手算例稱為「學到了」

### 原句

> 「兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的『模式』。」

### 原因

該例使用固定身份矩陣：

> `W_Q = W_K = W_V = W_O = I_4`

沒有 optimizer、loss 或訓練 loop。因此只能說不同固定特徵分塊產生了不同輸出，不能說「學到了」。後一句稱為「頭塌縮」也容易把人工構造的相同輸出誤稱為訓練後診斷。

### 最小修法

改成：

> 兩個固定特徵子空間在此輸入上產生不同注意力結果；本例沒有訓練，不能據此宣稱各頭學到不同語義模式。

把「頭塌縮」改為「兩頭數值相同的退化構造」。

---

## 9. 第16、17章含沒有證據支持的來源取得／核對聲明

### 第16章原句

> 「2026-10-06 取得 N1、N2 摘要頁……N3 已取得明確 2.14 API 全文並核對……」

### 第17章原句

> 「依題目附註，只取得摘要頁……」

以及：

> 「已依題目附註核對 `True` 參與注意……」

### 原因

提供的來源清單只有書目與 URL，沒有作者在特定日期實際取得頁面、閱讀摘要或核對全文的執行紀錄。章稿不能自行虛構查閱日期或聲稱已核對。即使 API 敘述本身正確，來源存取行為仍不能無證據宣稱。

第16章所列日期還是精確日期，更需要可定位紀錄；目前沒有。

### 最小修法

刪除日期及「已取得／已核對」措辭，統一改為：

> N3 為題目提供的 PyTorch 2.14 API 來源入口；使用時仍須依實際版本核對。此連結不是本機安裝、執行或來源查閱紀錄。

第17章同樣改為「來源入口」「未宣稱逐段核對」，不要寫「依題目附註已核對」。

---

# 二、應一併修正的實作問題

## 10. 第16章未拒絕非有限輸入與參數

### 原句

```python
def forward(self, X, mask=None):
    if X.ndim != 3 or X.shape[-1] != self.D:
        ...
```

### 原因

第14、15、17、18章都明確處理非有限值；第16章卻允許 `NaN` 或 `inf` 進入 QKV、scores 與 softmax。若無 mask，全遮罩檢查也無法阻止 NaN，最後會靜默輸出非有限值。

### 最小修法

檢查 `X` 及全部參數有限，並在 softmax 前檢查 `S` 的非遮罩位置有限。加入 NaN 輸入與 inf 參數故障測試。

---

## 11. 第16章 `D_out`、`D`、`H` 的型別契約不完整

### 原句

```python
if D <= 0 or H <= 0:
...
self.D_out = D if D_out is None else D_out
```

### 原因

布林值在 Python 中是整數子類別，`D=True,H=True` 可能通過比較及整除；`D_out=0` 可建立空輸出，負數或浮點數則在更晚的 shape 建立才出錯。第17章已特別拒絕布林冒充整數，第16章應保持一致。

### 最小修法

加入本地正整數檢查，拒絕 bool，並要求 `D_out` 為正整數。

---

## 12. 第16章的 $H=1$ 描述自相矛盾

### 原句

> 「模型與單頭縮放點積注意力等價（權重不同）。」

### 原因

若 QKV 投影、輸出投影、mask 與縮放相同，$H=1$ 就是同一運算，注意力權重不應不同。若初始化不同，則不是數學等價比較。

### 最小修法

改為：

> 當參數與遮罩相同時，$H=1$ 退化為帶 QKV 與輸出投影的單頭縮放內積注意力。

---

# 三、其餘章的核對結果

第13章的 `np.add.at` 正確累加重複索引；共享矩陣梯度也正確相加。第14章的分數梯度每列和為零、非方形 $T_q\ne T_k$ 與固定 mask 有限差分設計均合理。第15章絕對位置 causal 規則與第17章 cache 位置偏移一致。第17章相鄰偶奇配對、$d_h$ 偶數限制及共同位置平移測試正確。第18章 LN 與 RMSNorm 的 reduction 軸、$\epsilon$ 位置、共享參數梯度及殘差兩路累加皆正確。

因此不需重寫整部章稿；但第16章仍缺少本卷要求的多頭完整梯度驗收，且跨框架轉接、mask 軸契約和來源聲明均有實質錯誤。在這些問題修正前不能批准。

VERDICT: REVISE