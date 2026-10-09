## 審稿結論

本次提交的關鍵程式與上一版相同，先前指出的阻擋問題尚未修正。重新核算後，模型的主體形狀、因果遮罩、有效 token loss、右側 padding 契約、多層因果性證明、兩個手算及完整訓練 loop 均基本正確；但 PAD embedding 梯度測試仍是假陽性，`view` 的 stride 說明仍有事實錯誤。這兩項直接涉及本卷要求的梯度與故障測試，不能視為純風格問題。

以下不假定程式曾被執行，也不把程式中的 `Passed` 輸出當作實際通過紀錄。

---

# 一、先行重算

## 1. 端到端 shape

輸入為：

\[
X\in\mathbb Z^{B\times T}.
\]

token embedding 為：

\[
E[X]\in\mathbb R^{B\times T\times D}.
\]

position embedding 切片為：

\[
P_{0:T}\in\mathbb R^{1\times T\times D},
\]

沿 batch 軸廣播後，兩者相加仍為：

\[
H_0\in\mathbb R^{B\times T\times D}.
\]

Q/K/V 的線性投影均保持最後一軸 \(D\)：

\[
Q,K,V\in\mathbb R^{B\times T\times D}.
\]

因 \(D=Hd_h\)，拆分最後一軸後為：

\[
(B,T,H,d_h),
\]

再交換 time 與 head 軸得到：

\[
(B,H,T,d_h).
\]

因此：

\[
QK^T\in\mathbb R^{B\times H\times T\times T}.
\]

二維 mask \((T,T)\) 向前導的 batch、head 軸廣播；softmax 沿 `dim=-1`，正是 key 軸。與 V 相乘後：

\[
A V\in\mathbb R^{B\times H\times T\times d_h}.
\]

transpose 回 \((B,T,H,d_h)\) 並合併最後兩軸後：

\[
(B,T,D).
\]

最終輸出：

\[
Z\in\mathbb R^{B\times T\times V}.
\]

此部分正確，未發現錯誤 broadcast 或錯誤 reduction 軸。

## 2. causal mask

程式：

```python
mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
```

採用 True=允許。下三角位置 \(j\le i\) 為 True，未來位置 \(j>i\) 經 `~mask` 後填入 \(-\infty\)。每一 query 至少允許自身位置，因此目前方形 self-attention 不會出現全遮罩列。遮罩在 softmax 前施加，正確。

## 3. PAD loss

程式先定義：

```python
valid_mask = targets.ne(self.PAD_ID)
valid_count = valid_mask.sum().item()
```

再計算：

```python
ce_loss_sum = nn.functional.cross_entropy(
    logits.reshape(-1, self.vocab_size),
    targets.reshape(-1),
    ignore_index=self.PAD_ID,
    reduction='sum'
)
loss = ce_loss_sum / valid_count
```

故實際 loss 是：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf 1[Y_{b,t}\ne PAD]
}.
\]

分子與分母使用相同 target 有效集合，且只平均一次。全 PAD target 在除法前拋出：

```python
"No valid tokens in targets (all PAD?)"
```

這部分正確。

## 4. 右側 padding 契約

以 input 為例，程式把 non-PAD 指示量反轉，透過 suffix `cummax` 判斷是否在某個 PAD 的原始右方又出現非 PAD。對：

\[
[1,0,2,0]
\]

會偵測違規；對：

\[
[1,2,0,0]
\]

不會誤判。target 使用同樣邏輯。

另有：

```python
if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
    raise ValueError("PAD input positions must have PAD targets")
```

它正確實作單向條件：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

它沒有錯誤地要求反向條件，因此：

\[
X=[1,2,3],\quad Y=[2,3,PAD]
\]

仍合法。新增的合法與故障案例與此契約一致。

## 5. 手算

稿中注意力矩陣重算為：

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}.
\]

縮放並遮罩後，第二列 softmax 約為：

\[
(0.3302,0.6698,0),
\]

第三列約為：

\[
(0.2482,0.5035,0.2482).
\]

故輸出第三列：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

稿中結果正確。

FFN 手算亦有：

\[
[1,2]W_1=[1,2,1,2],
\]

\[
[1,2,1,2]W_2=[2,5].
\]

正確。

## 6. 因果性證明

證明已明確處理 `LN1`、attention、殘差、`LN2` 與 position-wise FFN。LayerNorm 只沿 feature 軸運算，不混合不同時間位置；attention 對位置 \(i\) 只使用 \(j\le i\)；FFN 也不混合 time 軸。因此歸納結論與程式的 pre-norm block 相符。

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

雖然 PAD 這次確實出現在 embedding lookup，但它位於最後位置。該位置的 target 也是 PAD，所以該位置的 loss 被 `ignore_index` 排除。

有效 loss 僅來自位置 0、1。由因果性：

- 位置 0 的 logits 只依賴 input 位置 0；
- 位置 1 的 logits 只依賴 input 位置 0、1；
- input 位置 2 的 PAD 不可能影響前兩個位置。

因此，從有效 loss 到位置 2 的 PAD embedding 沒有計算圖路徑。即使把：

```python
nn.Embedding(vocab_size, d_model, padding_idx=0)
```

故意改成：

```python
nn.Embedding(vocab_size, d_model)
```

PAD row 在這個測試中仍會得到零梯度。測試不能區分正確實作和缺少 `padding_idx` 的故障實作，故屬假陽性。

這不是可忽略的測試強度偏好，而是測試所聲稱驗證的性質實際沒有被驗證。

## 最小修法

先用獨立 embedding probe 驗證 `padding_idx`：

```python
model.zero_grad(set_to_none=True)
probe_ids = torch.tensor([[0, 1]], dtype=torch.long)
probe = model.tok_emb(probe_ids).sum()
probe.backward()

assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

這裡 PAD row 與非 PAD row 都直接參與同一個標量。若沒有 `padding_idx=0`，PAD row 會得到非零梯度；正確設定時 PAD row 才為零。

之後必須清空 probe 梯度，再做端到端測試：

```python
model.zero_grad(set_to_none=True)
_, loss_grad = model(x_grad, t_grad)
loss_grad.backward()
```

如此才能分別驗證：

1. embedding 的 `padding_idx` 行為；
2. 完整模型的梯度流。

---

# 三、阻擋問題二：`view` 說明仍有事實錯誤

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

## 原因

在目前寫法：

```python
out.transpose(1, 2).contiguous().view(...)
```

中，先建立 contiguous layout 再 `view` 是正確的。

但「直接 `view` 會……給出錯誤的記憶體映射」不精確。當 stride 與要求的 view 不相容時，`view` 應拋出錯誤，而不是靜默改變元素語義。另若使用 `reshape`，框架可以在必要時建立副本，因此 `.contiguous()` 不是所有等價寫法都必須顯式呼叫。

## 最小修法

替換為：

> 「本程式在 transpose 後使用 `view`，因此先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常與所求 shape 不相容，使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

這不需更動模型程式。

---

# 四、因果性測試會偶爾引入 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

## 原因

若原 token 為 \(V-1\)，修改後成為 0，也就是 PAD。PAD 位於最右側，雖不違反右側 padding 契約，但測試同時改變了 token 身分和 padding 身分，不能算純粹的「未來普通 token 修改」。

## 最小修法

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

對原本位於 \(1,\ldots,V-1\) 的 token，此變換保持結果非零，且必定與原值不同。

這是次要問題，不單獨構成拒稿理由，但應與梯度測試一併修正。

---

# 五、梯度覆蓋的文字與程式不一致

## 逐字原句

> 「所有參數必須具有正確的梯度。」

實際程式只檢查第一層 block 的代表性參數，沒有檢查：

- `pos_emb`
- 第二層 block
- LayerNorm 參數
- FFN bias
- `final_norm`

而「測試與預期結果」又只寫：

> 「檢查 `out.weight.grad` 是否存在且有限。」

三處範圍彼此不一致。

## 最小修法

若要符合「所有參數」：

```python
for name, p in model.named_parameters():
    assert p.grad is not None, name
    assert p.grad.shape == p.shape, name
    assert torch.isfinite(p.grad).all(), name
```

不應要求每個梯度元素非零，因為合法計算也可能產生零元素。

若只想保留代表性抽查，則把正文改成「代表性參數」，並把測試說明列成 embedding、Q/K/V、attention output、FFN 和 logits projection，而不是只寫 `out.weight`。

---

# 六、生成器與空 batch 缺少參數契約

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

## 原因

函數實際要求：

\[
batch\_size\ge1,\quad seq\_len\ge1,\quad vocab\_size\ge2.
\]

`vocab_size=1` 時不存在非 PAD token，取樣區間也為空。程式沒有以章內明確錯誤訊息拒絕這些輸入。

`TinyDecoder.forward` 只檢查 \(T\ge1\)，沒有拒絕 \(B=0\)。後續空張量經 attention 的行為不應成為未定義的隱含契約。

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

並為四個條件增加故障測試。

模型 constructor 最好也要求 `vocab_size >= 2`，因為 ID 0 已保留為 PAD，而可訓練資料需要至少一個非 PAD token。

---

# 七、reshape／transpose 習題仍無法鑑別錯誤

## 逐字原句

> 「輸入 \(X\in\mathbb R^{1\times2\times4}\)，\(D=4,H=2\)」

此時 \(T=H=2\)。因此：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

漏做 transpose 的學生仍會寫出同一組 shape 數字。

## 最小修法

改成 \(T=3\)：

\[
(1,3,4)
\to(1,3,2,2)
\to(1,2,3,2),
\]

並要求列出：

\[
scores=(1,2,3,3).
\]

如此才能實際測出 reshape 不等於 transpose。

---

# 八、反例措辭過度絕對

## 逐字原句

> 「因果性測試將失敗。」

使用上三角 mask 會破壞因果保證，但單次隨機模型不保證一定觀察到超過容差的差異；未來 value 的貢獻可能為零、抵消或極小。

## 最小修法

改成：

> 「上三角 mask 破壞因果性保證；一般非退化權重下，修改未來 token 預期會使較早位置的輸出改變。若要構成確定性反例，應固定使未來 value 對較早 query 有非零權重的參數與輸入。」

另一原句：

> 「Softmax 飽和，梯度不穩定。」

亦應改成條件式：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、來源與執行聲明

稿件明確寫：

> 「程式結果為預期行為，非已執行紀錄。」

這一點合格。沒有捏造實測數值、訓練成功、收斂率、硬體速度或下載模型。程式內的 `print("Passed")` 是未來執行時的控制流程輸出，不等於作者宣稱已經執行。

N3 的來源註記已限定到特定 PyTorch 2.14 API，但參考來源正文仍寫：

> 「框架的布林遮罩語義可能與本卷……不同」

最小修法是說明：N3 所列特定 SDPA API 中 True 表示參與，與本卷一致；其他 API、參數和版本不可直接類推。本章沒有直接使用 SDPA，因此不影響核心程式。

---

# 十、最終評估

已符合的要件包括：

- 正文超過最低篇幅；
- 完整 embedding、位置、MHA、LayerNorm、FFN 與 logits；
- 完整 optimizer、全域梯度裁切與訓練 loop；
- 自足 CPU 程式；
- 正確 next-token 位移；
- 正確有效 token loss；
- 右側 padding 契約及正常／故障測試；
- 完整小命題證明；
- 兩個逐步手算；
- 四類習題及答案；
- 合成資料與養殖安全界線；
- 無外部下載、shell、GPU 或未提供模組；
- 無虛構執行、來源查證範圍或模型能力。

仍未符合的核心要件是：PAD embedding 故障測試不具有故障鑑別力，以及 `view` 的文字定義不精確。兩者均有很小的局部修法，但在實際套用前不能核准。

VERDICT: REVISE