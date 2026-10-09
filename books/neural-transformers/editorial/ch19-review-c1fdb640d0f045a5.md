## 獨立審稿結論

本輪新增的 next-token 定義修正了先前容易混淆的長度記號：明確從長度 \(T+1\) 的原始序列建立長度 \(T\) 的 input 與 target。其餘核心程式與已核准版本一致。重新核算後，未發現會推翻前次核准的新問題。

我沒有執行程式；以下均為靜態推導與預期行為，不把程式中的 `Passed` 訊息當成實際執行證據。

---

# 一、next-token 定義已更精確

本輪原句：

> 「對一份長度 \(T+1\) 的原始序列 \((s_0,\ldots,s_T)\)，next-token prediction 取長度同為 \(T\) 的輸入 \(X_t=s_t\) 與目標 \(Y_t=s_{t+1}\)，其中 \(0\le t<T\)。」

這與程式：

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
x = seq[:, :-1]
targets = seq[:, 1:]
```

完全一致。若 `seq_len=T`，則：

- `seq` 為 \((B,T+1)\)；
- `x` 為 \((B,T)\)；
- `targets` 為 \((B,T)\)；
- \(Y_t=s_{t+1}\)。

補充說明「只有 \(T\) 個原始詞元時只能建立 \(T-1\) 個預測位置」亦正確。這消除了先前把原始序列長度與模型輸入長度都寫成 \(T\) 的潛在歧義。

---

# 二、shape、broadcast 與 reduction

端到端 shape 為：

\[
(B,T)\rightarrow(B,T,D)
\rightarrow(B,T,H,d_h)
\rightarrow(B,H,T,d_h)
\rightarrow(B,H,T,T)
\rightarrow(B,H,T,d_h)
\rightarrow(B,T,D)
\rightarrow(B,T,V).
\]

核對如下：

- token embedding：\((B,T,D)\)；
- position embedding：\((1,T,D)\)，沿 batch 軸廣播；
- Q/K/V projection：\((B,T,D)\)；
- split heads：\((B,T,H,d_h)\)；
- transpose：\((B,H,T,d_h)\)；
- scores：\((B,H,T,T)\)；
- softmax：沿最後 key 軸；
- attention output：\((B,H,T,d_h)\)；
- merge heads：\((B,T,D)\)；
- logits：\((B,T,V)\)。

正文對 `view`、stride 與 `reshape` 的說明已正確：transpose 後直接 `view` 通常因 stride 不相容而拋錯；使用 `reshape` 時框架可在必要時建立副本。程式的 `.contiguous().view(...)` 與文字一致。

---

# 三、mask 與因果性

程式使用下三角 mask：

```python
mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
```

其語義為：

\[
M_{ij}=\mathrm{True}\iff j\le i.
\]

符合本卷 True=允許的約定。mask 在 softmax 前施加，且 softmax 沿 key 軸。方形 self-attention 每列至少保留自身 key，因此不存在全遮罩 query 列。

多層因果性證明亦完整處理：

- `LN1` 不混合時間位置；
- causal attention 只讀取 \(j\le i\)；
- residual 只合併同一位置；
- `LN2` 與 FFN 均為 position-wise；
- final norm 與 output projection 不混合時間軸。

所以位置 \(i\) 的 logits 只依賴 \(X_0,\ldots,X_i\)。有限數值測試只作實作檢查，沒有取代證明。

---

# 四、loss 與 PAD

程式計算：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf1[Y_{b,t}\ne PAD]
}.
\]

`ignore_index=PAD_ID` 排除分子中的 PAD；`valid_count` 以相同 target mask 控制分母；`reduction='sum'` 後只除一次。全 PAD target 在除法前拋錯。flatten 後 logits 是 \((BT,V)\)，target 是 \((BT)\)，符合 cross-entropy 介面。

右側 padding 契約也保持正確：

\[
[\text{有效前綴},\text{PAD 後綴}].
\]

另有：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

這不會誤拒絕合法的序列結束：

\[
X=[1,2,3],\qquad Y=[2,3,PAD].
\]

在此受限資料契約下，有效 query 不可能看到位於未來的 PAD key，因此不另加 padding key mask 可以成立。

---

# 五、梯度

PAD embedding 的獨立 probe：

```python
emb = model.tok_emb(torch.tensor([[0, 1]], dtype=torch.long))
emb.sum().backward()
assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

具有真正的故障鑑別力：

- 若移除 `padding_idx=0`，PAD row 預期收到非零梯度；
- 若所有 embedding 梯度都被切斷，非 PAD row assertion 預期失敗。

probe 後重新 `zero_grad(set_to_none=True)`，不會污染端到端 loss 測試。

整合梯度測試覆蓋 embedding、Q/K/V、attention output projection、FFN 與 vocabulary projection，並核對梯度存在、shape 相同且有限。符合本章的整合冒煙測試目的。

---

# 六、手算

注意力例題重算：

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}.
\]

第三列 softmax 約為：

\[
(0.2482,0.5035,0.2482),
\]

故輸出：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

稿中正確。

FFN 例題：

\[
[1,2]W_1=[1,2,1,2],
\]

\[
[1,2,1,2]W_2=[2,5].
\]

數值與 shape 均正確。

---

# 七、程式完整性與測試

本章自足定義：

- `TinyDecoder`
- `TransformerBlock`
- `MultiHeadAttention`
- `FFN`
- 合成資料生成器
- loss
- optimizer
- backward
- 全域梯度裁切
- optimizer step
- 完整有限步訓練 loop
- `main()` 入口

沒有 `pass`、TODO、未定義模組、外部資料、權重下載、tokenizer 下載、網路、shell 或 GPU 操作。

測試含：

- \(T=1\)；
- 超過 `max_len`；
- \(D\) 不可被 \(H\) 整除；
- PAD loss 與手動有效 target loss 比較；
- 全 PAD target；
- PAD embedding probe；
- 代表性梯度；
- token 越界及負值；
- input／target dtype；
- `max_len=0`；
- 合法與非法右側 padding。

正常、邊界與故障類別均已具備。`max_len=0` 測試使用顯式旗標與 assertion，預期行為清楚。

---

# 八、資料洩漏、能力與來源

資料為固定 seed 的 IID 合成 token，只用於 plumbing。正文沒有把十步訓練描述成已收斂，也沒有由訓練 loss 推論語法、泛化或安全能力。

養殖案例：

- 明示資料合成；
- 沒有真實操作閾值；
- 不控制設備；
- 不作現場專業決策。

正文逐字聲明：

> 「程式結果為預期行為，非已執行紀錄。」

所以沒有虛構執行、測試成功、訓練結果、時間或設備效能。來源也明確限制 N1、N3、N4 的核對範圍，沒有虛構前章模組或外部能力。

---

# 九、非阻擋改善

下列仍可選擇性修改，但不影響核准：

1. 因果測試可保證修改後仍為非 PAD：
   ```python
   x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
   ```
2. shape 習題可採 \(T=3,H=2\)，避免 reshape 與 transpose 的數字相同。
3. 生成器可顯式驗證 `batch_size>=1`、`seq_len>=1`、`vocab_size>=2`。
4. 「因果性測試失敗」宜寫成「一般非退化權重下預期失敗」。
5. 「Softmax 飽和」宜寫成「可能使 softmax 過尖」。
6. 梯度測試說明可同步列出實際抽查的各類參數與 PAD probe。

上述是措辭或健全性增強，不構成數學、shape、loss、梯度、mask 或自足性缺陷。

---

# 最終判定

本章達到最低篇幅，具備完整小命題證明、兩個逐步手算、自足 CPU Transformer、完整訓練 loop、正常／邊界／故障測試，以及手算／程式／反例／整合四類習題與答案。核心 shape、broadcast、reduction、梯度、概率、loss、mask、position、padding 與 next-token 定義均成立，也沒有虛構執行、來源或模型能力。

VERDICT: APPROVE