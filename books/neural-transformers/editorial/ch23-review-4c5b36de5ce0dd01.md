# 第23章 KV cache與推論等價性（repair_generation 17，round 498）審稿意見

## 一、本輪修訂焦點

本輪與前一稿相比，唯一實質改動是故障測試段：先前以 `try/except ValueError: pass/else: raise AssertionError` 的結構撰寫，並在正文另加一段「此程式中的 `pass` 只表示測試已預期捕獲例外，不是未完成的函式實作」辯護。本稿改以布林旗標，`pass` 完全消失：

```python
try:
    model.step(ids[:, 2:3], cache, start=2)
except ValueError:
    length_mismatch_rejected = True
else:
    length_mismatch_rejected = False
if not length_mismatch_rejected:
    raise AssertionError("cache length/start mismatch was accepted")
```

第二個故障測試採同一結構，旗標名為 `batch_mismatch_rejected`。這對應本卷「程式不得留 `pass`／TODO／omitted」的硬性要求：之前的辯護在文本上可接受，但在自動掃描上 `pass` 仍是禁字；本輪直接移除，符合公約，且不再需要正文另外解釋。這是正確且必要的修訂。

## 二、重算與逐項核對

### 1. 定義、遮罩、softmax

$Q\in\mathbb R^{B\times H\times m\times d_h}$、$K,V\in\mathbb R^{B\times H\times(L+m)\times d_h}$、$M_{i,j}=[j\le L+i]$。$i$ 為區塊列索引，絕對位置為 $L+i$；$j$ 為 key 絕對位置；每列至少含自 key（$j=L+i$），故分母非零。$A_{b,h,i,j}$ 中 $(S-c)$ 為穩定平移，求和的 $r$ 限於 $M_{i,r}=\text{True}$，與程式 `np.where(..., scores, -np.inf)` 及 `weights.sum(axis=-1, keepdims=True)` 一致。掩蓋方式在 softmax 前，符合「遮罩在 softmax 前施加」。

### 2. 小命題與證明

歸納鏈：輸入層位置一致 → 逐詞元投影同一 $Q,K,V$ → 遮罩選出同一組 key → softmax 及加權和相同 → 逐詞元輸出投影、殘差、逐詞元非線性相同 → 逐層至 logits。結論限定實數算術。dropout、SDPA `dropout_p=0`、布林遮罩語義、跨平台逐位一致的分界均正確聲明。

### 3. 規模尺度

單層點積：完整重算 $\sum_{t=1}^T t^2=\Theta(T^3)$；cache $\sum_{t=1}^T t=\Theta(T^2)$。prefill $P$ 加生成 $G$：$O(P^2)+O(PG+G^2)$。每層 $2BN_{\rm layer}TD$ 個數。$L$ 為前綴長度、$N_{\rm layer}$ 為層數，符號切換正確。

### 4. 手算例

例一：$d_h=1$，batch 與頭數均 $1$；位置 $(2,3)$ 對 key $(0,1,2,3)$；遮罩 $\begin{pmatrix}1&1&1&0\\1&1&1&1\end{pmatrix}$；輸出 $4$、$5$；誤用 `tril(2,4)` 得 $2$。例二：位置錯位，$x_2=2+3=5$ 對 $2+0=2$。均與程式一致。

### 5. 程式逐函式

`layer_norm` 沿最後特徵軸，`eps` 在根號內。`masked_attention` 檢查 `q.ndim/k.ndim/v.shape`、`k.shape[:2]==(b,h)`、`k.shape[3]==dh`、`query_start<0`、`tq==0`、`tk==0`；`allow = kpos[None,:] <= qpos[:,None]`；允許位置有限性檢查；`peak` 後 exp、正規化、`@ v`。前幾輪我曾誤稱應補 `tk >= query_start + tq` 檢查，現更正：`allow.any(axis=-1)` 於 key 不足以覆蓋任何 query 時為 False，函式即以 `all-masked query` 拒絕，已足以攔截該情形。

`TinyDecoder.step` 檢查 `ids.ndim==2`、整數 dtype、範圍，`start` 整數且 `start+tnew<=max_len`，`len(cache)==self.layers`，每層 K/V 形狀為 `(b, heads, start, dh)`。`x = token[ids] + position[start:start+tnew]`。逐層投影、拆頭、拼接舊 K/V、`masked_attention`、殘差、FFN、存新 K/V。`full` 另建空 cache，不沿用先前樣本。`cache_decode` 先 prefill，再逐詞元 step，`np.concatenate` 拼回。

`split` 為 `reshape(b,t,H,dh).transpose(0,2,1,3)`，`merge` 為 `transpose(0,2,1,3).reshape(b,t,D)`，兩者互逆；`reshape` 與 `transpose` 分開陳述，符合本卷「reshape 不是 transpose」的規定。

### 6. 測試

正常：`full(ids)` 對 `cache_decode(..., prefill=2)`，形狀 $(2,5,11)$，cache 每層 $(2,2,5,2)$。邊界：$B=T=1$。因果：逐位置掃 $j=1,\ldots,T-1$，每次擾動後檢查早位置 logits 不變，並對擾動序列再比較完整與 cached 前向。故障：`start=2` 對長度 5 cache 拒絕；$b=1$ 新樣本對 $b=2$ cache 拒絕。本輪改用 `length_mismatch_rejected`、`batch_mismatch_rejected` 旗標，取代先前 `pass`。

### 7. 自足、來源、執行紀錄

模型、資料、遮罩、故障測試在章內自足；未下載權重、未載入 pickle、未執行外部工具。N1 為背景、N3 為版本化 API 資料，均非本機執行證據；N4–N6 標為待核對。無虛構執行、時間、設備、收斂率。養殖案例標明合成、無真實閾值、不授權控制設備。習題涵蓋手算、程式、反例、整合，解答完整。

### 8. 字數

仍為 4038 字，高於 3000 下限，未達 4500 目標。目標是軟性建議，不阻擋。

## 三、最小修法（不阻擋）

1. 字數若欲接近 4500，可於反例段補「多層 cache 誤傳」小例，或於測試段補「prefill=1 時無 decode 步驟，仍可檢查拼接與形狀」。屬風格選項。
2. 布林旗標寫法雖已移除 `pass`，`try` 塊本身亦可考慮改為 `with self.assertRaises` 形式；但 NumPy 腳本無 unittest 依賴，本寫法已達目的，不必改。

## 四、結論

我重算定義、遮罩、規模、命題證明、兩個手算、程式逐函式與測試語義；本輪唯一實質修改是移除 `pass`、以旗標取代 `try/except/else`。此修改回應本卷「程式不得留 `pass`」的硬性要求，未引入新問題。旗標邏輯正確：捕獲例外則 `..._rejected=True`，未捕獲則為 False 並拋 `AssertionError`。未發現新的形狀、梯度、遮罩、位置或來源問題。

VERDICT: APPROVE