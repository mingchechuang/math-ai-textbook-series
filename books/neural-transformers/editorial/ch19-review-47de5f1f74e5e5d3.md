## 獨立審稿結果

本次稿件的關鍵內容與前次相同，先前要求的局部修正尚未套用。本人重新核算 shape、mask、loss、梯度路徑與測試鑑別力後，結論仍是：模型主體、有效 token loss、右側 padding 契約、因果性證明、手算與完整訓練 loop 基本正確；但 PAD embedding 梯度測試仍是假陽性，且 `view`／stride 說明仍有技術錯誤。這兩項屬本卷明定審查範圍，不能以風格問題略過。

我沒有執行程式；以下只依靜態推導判斷預期行為。

---

# 一、重新核算：已正確部分

## 1. 張量形狀與廣播

輸入索引為 \((B,T)\)，token embedding 為 \((B,T,D)\)，position embedding 切片為 \((1,T,D)\)，兩者相加只沿 batch 軸廣播，結果仍為 \((B,T,D)\)。

每個 Q/K/V 投影輸出 \((B,T,D)\)，再依序：

\[
(B,T,D)\rightarrow(B,T,H,d_h)\rightarrow(B,H,T,d_h).
\]

其中 \(d_h=D/H\)。scores 為：

\[
QK^T\in\mathbb R^{B\times H\times T\times T}.
\]

mask 為 \((T,T)\)，正確廣播到 batch 與 head 軸。softmax 使用 `dim=-1`，即沿 key 軸。與 V 相乘後為 \((B,H,T,d_h)\)，合頭後回到 \((B,T,D)\)，最終 logits 為 \((B,T,V)\)。沒有發現錯誤 transpose、matmul 或 reduction 軸。

## 2. 因果遮罩

程式使用：

```python
mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
```

其語義是 True=允許，符合本卷約定。下三角保留 \(j\le i\)，未來位置 \(j>i\) 在 softmax 前設為 \(-\infty\)。方形 self-attention 每列至少允許自身位置，因此本實作沒有全遮罩 query 列。

## 3. loss

程式實際計算：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf1[Y_{b,t}\ne PAD]
}.
\]

`ignore_index=PAD_ID` 排除分子中的 PAD，`reduction='sum'` 後再除以有效 target 數，故平均只做一次。全 PAD target 亦在除法前拒絕。此部分正確。

## 4. padding 契約

input 與 target 都被限制為右側連續 padding。另有：

```python
idx.eq(PAD) & targets.ne(PAD)
```

用來拒絕 PAD input 對應有效 target。這是正確的單向條件：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

它不會錯誤拒絕合法的序列結束：

\[
X=[1,2,3],\qquad Y=[2,3,PAD].
\]

新增的合法右側 padding、input 內部 PAD、target 內部 PAD、PAD input 對應有效 target 測試，均符合此契約。

## 5. 證明與手算

多層因果性證明已包含 `LN1`、attention、殘差、`LN2` 與 position-wise FFN。由於 LayerNorm 與 FFN 不混合時間軸，而位置 \(i\) 的 attention 只使用 \(j\le i\)，歸納結論成立。

注意力手算重算得到第三列權重約為：

\[
(0.2482,0.5035,0.2482),
\]

輸出為：

\[
(0.4964,0.7517).
\]

FFN 手算輸出為 \([2,5]\)。兩者均正確。

---

# 二、主要阻擋問題：PAD 梯度測試仍是假陽性

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

PAD 雖然實際出現在 lookup 中，但位於最後位置。該位置 target 也是 PAD，因此它的 loss 被忽略。有效 loss 只來自位置 0、1。

因果模型中：

- 位置 0 只依賴 input 位置 0；
- 位置 1 只依賴 input 位置 0、1；
- 最後的 input PAD 位於兩者未來，不可能影響它們。

所以從總 loss 到 PAD embedding 沒有梯度路徑。即使故意移除：

```python
padding_idx=self.PAD_ID
```

PAD row 仍會得到零梯度。當測試在正確實作與故障實作上都通過，它就沒有驗證所聲稱的性質。

## 最小修法

將 `padding_idx` 測試與語言模型 loss 分離：

```python
model.zero_grad(set_to_none=True)
probe_ids = torch.tensor([[0, 1]], dtype=torch.long)
probe = model.tok_emb(probe_ids).sum()
probe.backward()

assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

這個 probe 讓 PAD 與非 PAD row 都直接參與標量。若沒有 `padding_idx=0`，PAD row 會得到非零梯度；有正確設定時才為零。

之後再清空梯度並執行端到端測試：

```python
model.zero_grad(set_to_none=True)
_, loss_grad = model(x_grad, t_grad)
loss_grad.backward()
```

這是本次最主要的必要修正。

---

# 三、`view` 的說明仍有事實錯誤

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

## 原因

目前程式先 `.contiguous()` 再 `.view()` 是正確的。但 stride 不相容時，`view` 應拋錯，不應描述成可能靜默給出錯誤映射。另外，若改用 `reshape`，框架可以在必要時建立副本，因此 `.contiguous()` 不是所有等價寫法都必須顯式使用。

## 最小修法

改為：

> 「本程式在 transpose 後使用 `view`，因此先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常與要求的 shape 不相容，使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

此項只需改文字，不必改模型程式。

---

# 四、因果性測試仍可能引入 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

## 原因

當原 token 為 \(V-1\) 時，新值為 0，即 PAD。雖然位於最右側而不違反 padding 契約，但測試同時改變普通 token 與 padding 身分，不是純粹的未來 token 擾動。

## 最小修法

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

原 token 來自 \(1,\ldots,V-1\)，此寫法保持結果非 PAD，且一定改變 token。

目前測試只改最後位置，作為整合冒煙測試尚可；一般因果性仍由證明支撐。

---

# 五、梯度覆蓋說明不一致

## 逐字原句一

> 「所有參數必須具有正確的梯度。」

## 逐字原句二

> 「檢查 `out.weight.grad` 是否存在且有限。」

實際程式則檢查第一層 block 的代表性參數，不只 `out.weight`，但也沒有涵蓋所有參數，例如：

- `pos_emb`
- 第二層 block
- LayerNorm 權重與偏置
- FFN bias
- `final_norm`

## 最小修法

若保留「所有參數」，使用：

```python
for name, p in model.named_parameters():
    assert p.grad is not None, name
    assert p.grad.shape == p.shape, name
    assert torch.isfinite(p.grad).all(), name
```

只檢查存在、同 shape、有限即可，不應要求所有元素非零。

若只想做代表性抽查，正文與測試說明都應改成「代表性參數」，並列明 embedding、Q/K/V、attention output、FFN 與 logits projection。

---

# 六、生成器與空 batch 未定義

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

## 原因

函數隱含需要：

\[
batch\_size\ge1,\quad seq\_len\ge1,\quad vocab\_size\ge2.
\]

`vocab_size=1` 時沒有非 PAD token，且取樣上下界相同。constructor 目前卻只要求 `vocab_size > 0`。forward 也沒有明確拒絕 \(B=0\)。

## 最小修法

生成器加入：

```python
if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
if seq_len < 1:
    raise ValueError("seq_len must be >= 1")
if vocab_size < 2:
    raise ValueError("vocab_size must include PAD and a non-PAD token")
```

forward 加入：

```python
if b < 1:
    raise ValueError("Batch size must be >= 1")
```

模型 constructor 最好也要求 `vocab_size >= 2`。再為這些條件增加故障測試。

---

# 七、習題無法區分 reshape 與 transpose

## 逐字原句

> 「輸入 \(X\in\mathbb R^{1\times2\times4}\)，\(D=4,H=2\)」

此時 \(T=H=2\)，所以：

\[
(B,T,H,d_h)=(1,2,2,2)
\]

與：

\[
(B,H,T,d_h)=(1,2,2,2)
\]

數字完全一樣。漏做 transpose 仍能寫出相同 shape。

## 最小修法

改成 \(T=3\)：

\[
(1,3,4)
\rightarrow(1,3,2,2)
\rightarrow(1,2,3,2),
\]

並補問：

\[
scores=(1,2,3,3).
\]

這才能實際檢驗 reshape 與 transpose 的差別。

---

# 八、反例與縮放措辭過強

## 逐字原句

> 「因果性測試將失敗。」

上三角 mask 會破壞因果保證，但隨機權重下，未來貢獻可能恰為零、抵消或低於容差。最小修法：

> 「上三角 mask 破壞因果性保證；一般非退化權重下，修改未來 token 預期使較早輸出改變。確定性反例應固定使未來 value 有非零貢獻的參數與輸入。」

另一原句：

> 「Softmax 飽和，梯度不穩定。」

未縮放並不必然造成飽和。最小修法：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差會隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、來源與執行證據

稿件明確聲明：

> 「程式結果為預期行為，非已執行紀錄。」

這符合證據契約。稿中沒有捏造已訓練、已收斂、實測速度、設備或外部下載。程式內的 `print("Passed")` 是預期執行流程，不等於宣稱作者已執行。

N3 的來源註記已說明特定 PyTorch 2.14 SDPA API 中 True 表示參與注意。參考來源處宜改成：該特定 API 與本卷 True=允許一致，但其他 API、mask 參數或版本不可直接類推。本章未直接使用 SDPA，故不影響核心實作。

---

# 十、總評

本章已符合：

- 最低正文篇幅；
- 完整 decoder-only 模型；
- embedding、位置、MHA、norm、FFN、logits 全部自足；
- 完整資料、loss、optimizer、全域梯度裁切與訓練 loop；
- 正確 next-token 位移；
- 正確有效 token 平均；
- 正確右側 padding 契約；
- 正常、邊界與 padding 故障測試；
- 完整因果性命題證明；
- 兩個正確手算；
- 四類習題及答案；
- 合成養殖資料與安全界線；
- 無下載、GPU、shell 或未提供模組；
- 無虛構執行或模型能力。

但 PAD embedding 測試仍無法區分有無 `padding_idx`，而 `view` 的文字仍有錯誤。這兩項已有極小修法，實際套用後才可核准。

VERDICT: REVISE