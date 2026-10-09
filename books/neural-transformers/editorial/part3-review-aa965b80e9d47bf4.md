## 審稿範圍與重算結論

本次只核對第13至18章的跨章座標、符號、介面、依賴與推導，不把風格差異當作拒稿理由，也未執行任何程式或使用外部工具。手算重核結果如下：

- 第13章 embedding 查表、重複索引累加、共享輸入／輸出矩陣的兩路梯度，主要公式正確。例二的 $L=E_0\cdot E_1$ 確實給出第0列梯度 $E_1=(3,4)$、第1列梯度 $E_0=(1,2)$。
- 第14章 SDPA 的 $Q,K,V$ 形狀、softmax 軸、$dV=A^{\mathsf T}dY$、softmax VJP、$dQ=dSK/\sqrt{d_k}$、$dK=dS^{\mathsf T}Q/\sqrt{d_k}$ 均正確。兩個手算例的主要數值亦相符。
- 第15章以絕對位置建立非方形 causal mask 的原理正確，例中的 $(2,6)$ cache mask 也正確。
- 第16章拆頭／合頭互逆的索引推導與兩個手算例主要正確，但程式與測試契約存在數個實質缺口，其中 PyTorch 交叉驗證片段按原文會遇到 dtype 不一致。
- 第17章 RoPE 的相鄰偶奇配對、範數保持以及
  $$
  (R_pq)^{\mathsf T}(R_rk)=q^{\mathsf T}R_{r-p}k
  $$
  的符號與證明正確；cache 起始位置測試方向也正確。
- 第18章 LN、RMSNorm 的前向與輸入梯度重算正確。尤其 LN 在 $\epsilon>0$ 時仍可使用
  $$
  dx=s^{-1}\left(u-\overline u-z\,\overline{uz}\right),
  $$
  不需要假設 $\sum_jz_j^2=D$；程式與正文在這點一致。

以下列出需要修訂的可定位問題。

---

# 必須修訂

## 1. 第16章的 PyTorch 交叉驗證程式有確定的 dtype 錯誤

### 原句

> `m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0)`

以及：

> `X = torch.randn(B, T, D, dtype=torch.float64)`

又稱：

> 「在相同 `X`（float64）下比較。」

### 原因

PyTorch 模組建立後，參數預設通常是 `float32`；這段程式卻建立 `float64` 輸入。若直接呼叫 `m(X, X, X)`，參數與輸入 dtype 不一致，不能完成所述交叉驗證。原文雖未展示實際 forward 呼叫，但習題要求正是比較 PyTorch 與 NumPy 輸出，因此片段缺少必要的 dtype 轉換，不是單純展示省略。

此外，原文預期：

> 「float64 下若切分正確，最大絕對誤差在 `1e-10` 量級或更小」

要使這項預期有合理前提，PyTorch 模組本身也必須轉為 float64。

### 最小修法

在建立模組後增加：

```python
m = m.double()
```

或直接建立後寫：

```python
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
).double()
```

並補出最小必要的 PyTorch forward：

```python
with torch.no_grad():
    y_torch, _ = m(X, X, X, need_weights=False)
```

仍須保持「預期」措辭，不能聲稱誤差已實測。

---

## 2. 第16章多頭注意力程式沒有完成其自身宣稱的梯度驗收

### 原句

學習目標寫：

> 「推導合併與拆分的反向接口。」

程式則只給：

> `def split_backward(...)`

> `def merge_backward(...)`

測試中稱：

> 「如果有限差分在重排函式上不通過，問題必在軸操作而不在公式。」

但實際程式沒有多頭注意力的完整 backward，也沒有可執行的 split／merge 有限差分測試。

### 原因

本章主題是多頭注意力，而提供的 `MultiHeadAttention` 只有前向。第14章已完成單頭 $Q,K,V$ 梯度，但第16章加入了：

1. $XW_Q,XW_K,XW_V$ 三條共享輸入路徑；
2. split／merge；
3. 各頭 SDPA；
4. 輸出投影 $OW_O+b_O$；
5. 同一 $X$ 經 Q、K、V 三路返回後的梯度累加。

這些新組合正是容易出現 transpose、reshape 與漏加梯度的地方。只列 split／merge 的重排函式，不能驗證多頭模組的參數梯度、輸入三路累加或 mask 下梯度。也未滿足本部驗收要求中的「多頭 attention 的形狀／mask／梯度」這一核心項目。

這不要求把本章重寫成完整 Transformer，但至少需要本章自足地覆蓋自身多頭模組的反向。

### 最小修法

保留現有前向，增加一個最小完整 backward，至少返回：

- `dX`，其中 Q、K、V 三路貢獻相加；
- `dW_Q,dW_K,dW_V,dW_O`；
- `db_Q,db_K,db_V,db_O`。

然後用極小的 `B=1,T=2,D=4,H=2`、`float64`、關閉 dropout，對一个非對稱上游標量目標逐項或抽樣核對 `X` 與四個權重矩陣的中央差分。不能只用全一上游，因為對稱方向可能掩蓋 transpose 錯誤。部分遮罩位置也應有一組梯度測試，確認禁止位置的 attention 權重及分數梯度為零。

若篇幅限制不容納完整 MHA backward，則至少不能把本章描述成已涵蓋多頭梯度驗收；但依本卷明定的階段驗收，較妥當的最小修法仍是補齊。

---

## 3. 第16章四維 mask 的 batch/head 軸契約未檢查

### 原句

```python
elif m.ndim != 4:
    raise ValueError(...)
if m.shape[-2:] != (T, T):
    raise ValueError(...)
if np.any(np.all(~m, axis=-1)):
    raise ValueError(...)
S = np.where(m, S, -np.inf)
```

### 原因

程式只檢查 mask 最後兩軸，不檢查四維 mask 的前兩軸是否能合法對應到 $(B,H)$。例如分數是 `(2,2,T,T)`，卻傳入 `(2,3,T,T)`；程式自己的 shape 檢查不會拒絕，而是在 `np.where` 由 NumPy 丟出廣播錯誤。更危險的是某些意外可廣播形狀可能被接受，但其語義不是本章明示的：

- `(1,1,T,T)`：所有 batch、head 共用；
- `(B,1,T,T)`：每個 batch 共用各頭；
- `(1,H,T,T)`：各 batch 共用同一組 head mask；
- `(B,H,T,T)`：逐 batch、head 指定。

本章既然強調嚴密 shape 契約，就不應把前兩軸交給隱式 NumPy 行為。

全遮罩檢查也最好在廣播到實際 `S.shape` 後執行，這樣拒絕契約直接針對每個 $(b,h,q)$ 列，而不是針對未展開的 mask 表示。

### 最小修法

在轉成四維後明確檢查：

```python
if m.shape[0] not in (1, B) or m.shape[1] not in (1, self.H):
    raise ValueError("mask batch/head axes are not broadcastable")
m = np.broadcast_to(m, S.shape)
if np.any(np.all(~m, axis=-1)):
    raise ValueError("存在全遮罩 query 列")
```

並增加：

- `(B,1,T,T)` 正常測試；
- `(1,H,T,T)` 正常測試；
- `(B,H+1,T,T)` 故障測試；
- `(B+1,H,T,T)` 故障測試。

---

## 4. 第15章會把非布林 mask 靜默轉成布林，與跨章遮罩介面不一致

### 原句

```python
allowed = np.asarray(allowed, dtype=bool)
```

而第14、16章都明確拒絕非布林 mask。第16章寫：

> 「非布林遮罩：傳入 `float32` mask → 拋 `ValueError`；這避免用 0/1 混過布林約定。」

### 原因

第15章的 `attention` 會把浮點、整數甚至部分其他資料靜默轉成 bool。例如 `0.2` 會成為 `True`，而不只是精確的 0/1 被接受。這使同一本卷中相同的 `True=允許` 介面具有不同型別契約：

- 第14章：必須是 NumPy bool；
- 第15章：任何可轉成 bool 的陣列都接受；
- 第16章：必須是 bool。

這是實質跨章接口不一致，也可能掩蓋把加性 mask 誤傳成布林 mask 的錯誤。

### 最小修法

在轉換前保留原 dtype 並拒絕非 bool：

```python
allowed = np.asarray(allowed)
if allowed.dtype != np.bool_:
    raise TypeError("allowed 必須是布林；True=允許")
```

再進行 shape 檢查。補一個浮點 mask 故障測試，例如：

```python
attention(x, x, x, np.ones((3, 3), dtype=np.float64))
```

預期拋出 `TypeError` 或 `ValueError`，全卷統一即可。

---

## 5. 第16章宣稱的參數數量只在特定條件下成立

### 原句

> 「參數數量：多頭用 `4(D^2 + D)` 個參數（含偏置），與單頭在相同維度下相同。」

但同章已定義：

> `W_O ∈ ℝ^{D×D_out}`

> `b_O ∈ ℝ^{D_out}`

並允許：

> 「`D_out ≠ D`」

### 原因

在一般 $D_{\text{out}}$ 下，三個 QKV 投影各有 $D^2+D$，輸出投影有 $DD_{\text{out}}+D_{\text{out}}$，總數應為

$$
3D^2+3D+DD_{\text{out}}+D_{\text{out}}.
$$

只有當 $D_{\text{out}}=D$ 時，才化為

$$
4D^2+4D=4(D^2+D).
$$

正文目前先給一般 $D_{\text{out}}$，隨即給無條件的特殊情形公式，會誤導讀者。

### 最小修法

把原句改為：

> 一般參數數量為 $3D^2+3D+DD_{\text{out}}+D_{\text{out}}$；當 $D_{\text{out}}=D$ 時，化為 $4(D^2+D)$。在總投影寬度固定為 $D$ 的比較下，改變頭數 $H$ 本身不改變這個參數總數。

---

## 6. 第16章將固定身份投影的手算結果稱為「學到了」不同模式

### 原句

> 「兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的『模式』。」

### 原因

該例明確設定：

> `W_Q = W_K = W_V = W_O = I_4`

且沒有訓練流程。兩頭輸出不同，只能說明依固定特徵分塊後，兩個頭執行了不同的數值計算；不能說「學到了」。這屬於能力／執行敘述不實，而不是單純措辭偏好。

### 最小修法

改成：

> 兩個頭對同一 token 的輸出不同，說明固定的兩個特徵子空間在此輸入上產生不同注意力結果；本例沒有訓練，因此不能說兩頭已學到語義模式。

「頭塌縮」也宜改為「兩頭數值相同的退化示例」，避免把單一構造等同於訓練後現象。

---

## 7. 第16章包含無法由章稿支持的來源取得日期與查證聲明

### 原句

> 「2026-10-06 取得 N1、N2 摘要頁，未完整閱讀論文；N3 已取得明確 2.14 API 全文並核對……」

### 原因

章稿沒有提供瀏覽紀錄或其他可核對證據，且寫作規約明定不得虛構來源查閱與執行能力。來源清單只提供條目與 URL，不能反推作者確實在特定日期取得頁面或「已核對全文」。這也與其他章較審慎的「作為入口」「待逐條核對」口徑不一致。

即使相關 API 語義本身可能正確，也不能以沒有證據的個人查閱紀錄來包裝。

### 最小修法

刪除日期和「已取得／已核對全文」敘述，改為：

> N3 是題目提供的 PyTorch 2.14 API 來源入口；本文依所列介面契約說明 mask 與 dropout 注意事項，但不把來源連結當成本機版本或執行證據。

若確有外部編輯保存的查證紀錄，應由卷外來源管理系統承擔，不宜在章稿內自行聲稱。

---

# 建議修訂但不單獨構成拒稿

## 8. 第16章沒有拒絕非有限輸入與參數

### 原句

`forward` 只檢查：

```python
if X.ndim != 3 or X.shape[-1] != self.D:
    raise ValueError(...)
```

### 原因

第14、15、17、18章都明確拒絕非有限輸入；第16章卻會讓含 `NaN` 或 `inf` 的 $X$、權重或偏置進入 score 與 softmax，最後靜默產生非有限輸出。跨章數值策略不一致。

### 最小修法

在 `forward` 中檢查 `X` 及參數有限；至少在 softmax 前檢查允許位置的 `S` 有限。增加 NaN 輸入故障測試。

---

## 9. 第16章 `D_out` 缺少正整數檢查

### 原句

```python
self.D_out = D if D_out is None else D_out
```

### 原因

若 `D_out=0`，NumPy 可以建立空輸出軸，與章中「模型輸出維度」通常應為正數的語義不合；負數會由較晚的陣列建立拋出底層錯誤。若傳浮點數，也會在 shape 建立時才失敗。入口應明確定義契約。

### 最小修法

驗證 `D_out` 是非布林正整數，否則拋出 `ValueError`。

---

## 10. 第16章的「H=1 等價」句尾自相矛盾

### 原句

> 「`H=1`：退化為單頭，`A.shape == (B,1,T,T)`，模型與單頭縮放點積注意力等價（權重不同）。」

### 原因

若使用相同的 QKV 投影、輸出投影、縮放與 mask，$H=1$ 就是同一運算，權重不應不同；若參數初始化不同，任何兩個模型都可能權重不同，這不是 $H=1$ 的數學性質。

### 最小修法

改為：

> 在參數與遮罩相同時，$H=1$ 退化為帶輸入／輸出投影的單頭縮放內積注意力。

---

## 11. 第15章的 `q = x_cache = ...` 雖已說明，仍是不必要別名

### 原句

```python
q = x_cache = np.arange(12, dtype=np.float64).reshape(6, 2)[4:6]
```

### 原因

正文承認 `x_cache` 未使用。這不造成數值錯誤，但會讓 cache 測試看似存在額外狀態，實際沒有。最小化示範程式宜避免無用途名稱。

### 最小修法

照正文建議直接改為：

```python
q = np.arange(12, dtype=np.float64).reshape(6, 2)[4:6]
```

---

# 跨章一致性總評

第13、14、17、18章的核心數學推導大致可靠。第15章的絕對位置 causal mask 也與第17章的 RoPE cache 位置契約一致：兩章都正確強調 cache 的局部張量索引不能替代絕對位置。第13章的有效 token 平均、第15章的 masked token mean、第17章的 PPL 定義以及第18章「上游梯度只除一次」也相互一致，未發現重複平均問題。

主要阻礙集中在第16章：它是本部由單頭過渡到多頭的關鍵整合章，但目前只有完整前向，沒有完整多頭 backward；四維 mask 的前導軸契約不完整；交叉驗證片段存在確定的 dtype 錯誤；另有不受支持的來源查證敘述。這些均屬 shape、梯度、故障測試與能力陳述的實質問題，不能以風格理由略過。

修訂時不必重寫各章。最小範圍是：統一第15、16章 mask 型別契約；補第16章 mask 前導軸檢查、有限值檢查與完整多頭梯度驗證；修正參數數量及 PyTorch dtype；刪除未有紀錄支持的來源取得聲明和「已學到」措辭。其餘章可維持現有架構。

VERDICT: REVISE