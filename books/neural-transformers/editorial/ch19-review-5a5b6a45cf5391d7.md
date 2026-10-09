## 獨立審稿結論

本次提交的 `draft` 與上一輪在關鍵位置仍相同：PAD embedding 測試仍使用整合語言模型 loss，`view` 仍被描述成可能給出錯誤映射，因果測試仍可能把普通 token 改成 PAD，生成器仍沒有非法參數驗證。因此先前的阻擋問題尚未實際修復。

我未執行任何程式；以下為靜態重算與故障注入推理，所有測試結果只能稱為「預期」。

---

# 一、重新核算結果

## 1. 端到端 shape 正確

主路徑為：

\[
(B,T)\rightarrow(B,T,D)\rightarrow(B,H,T,d_h)
\rightarrow(B,H,T,T)\rightarrow(B,H,T,d_h)
\rightarrow(B,T,D)\rightarrow(B,T,V).
\]

Q/K/V 先由 \((B,T,D)\) reshape 為 \((B,T,H,d_h)\)，再 transpose 為 \((B,H,T,d_h)\)。scores 的最後兩軸是 query、key，shape 為 \((B,H,T,T)\)。二維 causal mask \((T,T)\) 正確廣播至 batch 與 head 軸，softmax 沿 `dim=-1`，即 key 軸。此部分無 shape、broadcast 或 reduction 錯誤。

## 2. causal mask 正確

```python
mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
```

採 True=允許，保留 \(j\le i\)，並在 softmax 前遮蔽未來。因每列至少包含對角線，方形 self-attention 不會出現全遮罩 query。

## 3. PAD loss 正確

程式等價於：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf1[Y_{b,t}\ne PAD]
}.
\]

`ignore_index=PAD_ID` 控制分子，`valid_count` 使用同一 target mask 控制分母，且只平均一次。全 PAD target 在除法前拒絕。此部分符合契約。

## 4. 右側 padding 驗證正確

input 與 target 均要求每列形如：

\[
[\text{非 PAD 前綴},\text{PAD 後綴}].
\]

另外：

```python
idx.eq(PAD_ID) & targets.ne(PAD_ID)
```

正確拒絕 PAD input 對應有效 target。合法的：

\[
X=[1,2,3],\quad Y=[2,3,PAD]
\]

不會被誤拒絕。新增的 padding 正常與故障案例覆蓋了主要契約。

## 5. 手算與證明正確

注意力例題重算後，第三列權重約為：

\[
(0.2482,0.5035,0.2482),
\]

輸出為：

\[
(0.4964,0.7517).
\]

FFN 例題輸出為 \([2,5]\)。兩者均正確。

因果性證明已包括 pre-norm 的 `LN1`、attention、殘差、`LN2` 與 position-wise FFN；LayerNorm 和 FFN 不混合 time 軸，故歸納論證成立。

---

# 二、阻擋問題一：PAD embedding 梯度測試仍是假陽性

## 逐字原句

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

以及：

```python
pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
```

## 原因

PAD 位於最後位置，而其 target 也是 PAD，所以最後位置的 loss 被 `ignore_index` 排除。有效 loss 只來自位置 0 與 1。由 causal mask：

- 位置 0 不依賴位置 2；
- 位置 1 不依賴位置 2；
- 因此位置 2 的 PAD embedding 沒有通向有效 loss 的路徑。

即使把：

```python
nn.Embedding(vocab_size, d_model, padding_idx=0)
```

故意改成：

```python
nn.Embedding(vocab_size, d_model)
```

這個測試中的 PAD row 梯度仍會是零。故該測試不能辨識缺少 `padding_idx` 的故障實作。

## 最小修法

增加獨立 embedding probe：

```python
model.zero_grad(set_to_none=True)
probe_ids = torch.tensor([[0, 1]], dtype=torch.long)
probe = model.tok_emb(probe_ids).sum()
probe.backward()

assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

之後清空梯度，再做端到端梯度測試：

```python
model.zero_grad(set_to_none=True)
_, loss_grad = model(x_grad, t_grad)
loss_grad.backward()
```

獨立 probe 才能區分有無 `padding_idx=0`。這是核准前必須修復的故障測試問題。

---

# 三、阻擋問題二：`view`／stride 說明仍錯

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

## 原因

目前程式先 `.contiguous()` 再 `.view()` 是正確的。但 stride 不相容時，`view` 應拋錯，不應說可能靜默給出錯誤映射。若使用 `reshape`，框架可在必要時建立副本，因此也不能無條件說 `.contiguous()` 不可省略。

## 最小修法

改成：

> 「本程式在 transpose 後使用 `view`，所以先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常與所求 shape 不相容，使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

---

# 四、因果測試仍混入 PAD 語義

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

若原值為 \(V-1\)，修改後變成 0，即 PAD。這雖不違反右側 padding 契約，卻使測試不再是單純修改未來普通 token。

## 最小修法

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

由於原 token 位於 \(1,\ldots,V-1\)，新 token 保持非 PAD 且一定與原值不同。

---

# 五、梯度測試的宣稱範圍不一致

## 逐字原句

> 「所有參數必須具有正確的梯度。」

但程式只抽查第一層 block 的部分權重；測試說明又只寫：

> 「檢查 `out.weight.grad` 是否存在且有限。」

三者不一致。未覆蓋的參數包括 `pos_emb`、第二層 block、LayerNorm、FFN bias 與 `final_norm`。

## 最小修法

若要宣稱所有參數均受檢查：

```python
for name, p in model.named_parameters():
    assert p.grad is not None, name
    assert p.grad.shape == p.shape, name
    assert torch.isfinite(p.grad).all(), name
```

不應要求所有梯度元素非零。若只保留目前抽查，則把正文改成「代表性參數」，並同步測試說明。

---

# 六、生成器與模型邊界不完整

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

此函數實際要求：

\[
batch\_size\ge1,\quad seq\_len\ge1,\quad vocab\_size\ge2.
\]

但沒有自行驗證。constructor 甚至允許 `vocab_size=1`，此時詞表只有 PAD，無非 PAD token。forward 也未拒絕 \(B=0\)。

## 最小修法

```python
if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
if seq_len < 1:
    raise ValueError("seq_len must be >= 1")
if vocab_size < 2:
    raise ValueError("vocab_size must include PAD and a non-PAD token")
```

constructor 應要求 `vocab_size >= 2`；forward 應增加：

```python
if b < 1:
    raise ValueError("Batch size must be >= 1")
```

並加入對應故障測試。

---

# 七、reshape／transpose 習題無鑑別力

## 逐字原句

> 「輸入 \(X\in\mathbb R^{1\times2\times4}\)，\(D=4,H=2\)」

此時 \(T=H=2\)，所以 reshape 後與 transpose 後都顯示為：

\[
(1,2,2,2).
\]

學生漏掉 transpose 也會得到相同 shape 數字。

## 最小修法

改成 \(T=3\)：

\[
(1,3,4)
\rightarrow(1,3,2,2)
\rightarrow(1,2,3,2),
\]

並要求 scores shape：

\[
(1,2,3,3).
\]

---

# 八、反例敘述過度絕對

## 逐字原句

> 「因果性測試將失敗。」

上三角 mask 破壞因果保證，但隨機權重下可能因未來路徑貢獻恰為零、抵消或小於容差，單次數值測試不一定觀察到失敗。

## 最小修法

改成：

> 「上三角 mask 破壞因果性保證；一般非退化權重下，修改未來 token 預期使較早輸出改變。確定性反例需固定一組使未來 value 有非零貢獻的權重與輸入。」

另一原句：

> 「Softmax 飽和，梯度不穩定。」

應改為條件式：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、來源、執行與能力宣稱

稿件明確寫：

> 「程式結果為預期行為，非已執行紀錄。」

因此沒有虛構實際執行、通過、訓練或收斂。程式中的 `print("Passed")` 是未來執行到該分支時的訊息，不是執行證據。

N3 的已核對範圍是特定 PyTorch 2.14 SDPA API，其中布林 True 表示參與注意。參考來源處應限定為：該特定 API 與本卷一致，其他 API 或版本不可直接類推。本章手寫 attention，核心程式不依賴 N3。

---

# 十、整體驗收

目前已具備：

- 超過最低正文篇幅；
- 完整 embedding、位置、MHA、norm、FFN、logits；
- 完整合成資料、loss、optimizer 與訓練 loop；
- CPU、自足、無外部下載；
- 正確 next-token 位移；
- 正確有效 token 平均；
- 正確右側 padding 契約；
- 正確因果性證明；
- 兩個正確手算；
- 正常、邊界與多項故障測試；
- 四類習題與答案；
- 合成養殖案例及安全界線；
- 無虛構執行、來源或模型能力。

但兩個核心問題仍未修復：PAD embedding 測試仍不能辨識故障，且 `view` 的定義敘述仍不正確。因本輪稿件沒有實際套用此前的最小修法，不能核准。

VERDICT: REVISE