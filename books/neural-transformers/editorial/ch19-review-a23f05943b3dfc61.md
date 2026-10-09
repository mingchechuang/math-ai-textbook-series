## 獨立審稿結論

本輪已新增右側 padding 的正常／故障測試，並把梯度測試輸入改成實際包含 PAD；這是實質進展。重新推導後，模型、loss、padding 契約、因果證明、兩個手算及訓練 loop 均大致正確。

但是新梯度測試仍不能驗證 `padding_idx`：雖然 PAD 出現在輸入中，它位於最右側，對應 target 又被 loss 忽略；因果方向使 PAD 位置不能影響更早的有效 loss。因此即使移除 `padding_idx=0`，PAD embedding row 仍會得到零梯度。這仍是假陽性，只是原因由「PAD 未出現」變成「PAD 出現但沒有任何 loss 路徑」。此外，正文仍保留 `view` 可能產生錯誤映射的錯誤說法，因果測試仍可能把 token 改成 PAD，生成器邊界未驗證，習題仍無法區分 reshape 與 transpose。故仍需局部修訂。

---

# 一、重新核算後已正確的部分

## 1. 主張量形狀

模型的實際 shape 流是：

\[
(B,T)
\to(B,T,D)
\to(B,H,T,d_h)
\to(B,H,T,T)
\to(B,H,T,d_h)
\to(B,T,D)
\to(B,T,V).
\]

具體而言：

- token embedding 是 \((B,T,D)\)；
- position embedding 是 \((1,T,D)\)，沿 batch 軸廣播；
- Q/K/V 先拆成 \((B,T,H,d_h)\)，再 transpose 成 \((B,H,T,d_h)\)；
- scores 為 \((B,H,T,T)\)；
- 二維 mask \((T,T)\) 對齊最後兩軸；
- softmax 沿 `dim=-1`，即 key 軸；
- 合頭後回到 \((B,T,D)\)；
- logits 為 \((B,T,V)\)。

此主鏈正確。

## 2. attention 概率公式

目前公式：

\[
A\in\mathbb R^{B\times H\times T\times T},
\]

\[
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_{k=0}^{T-1}\exp(S'_{b,h,i,k})}
\]

已正確區分張量與元素，也寫清楚 reduction 軸。禁止位置在 softmax 前設為 \(-\infty\)，因此其權重為零。

## 3. next-token loss

程式使用：

```python
ignore_index=self.PAD_ID,
reduction='sum'
```

並除以：

```python
valid_count = targets.ne(self.PAD_ID).sum().item()
```

因此：

\[
\mathcal L
=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf 1[Y_{b,t}\ne PAD]
}.
\]

PAD 同時從分子和分母排除，平均只做一次；零有效 target 也被拒絕。正確。

## 4. padding 契約及新增測試

程式現在限制：

\[
X=[\text{有效 prefix},\text{PAD suffix}],
\]

\[
Y=[\text{有效 prefix},\text{PAD suffix}],
\]

並要求：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

新增測試正確涵蓋：

- 合法 input／target 右側 padding；
- 合法的最後有效 input 預測 PAD；
- input 內部 PAD；
- target 內部 PAD；
- PAD input 對應有效 target。

這些測試與資料契約一致。因為有效 query 位於 PAD 後綴之前，causal mask 使其無法看到未來 PAD key，所以本章在此限制下不實作 padding key mask，可以成立。

## 5. 因果性證明

證明已納入 pre-norm：

\[
\widetilde H_{l-1}^{(j)}
=
\operatorname{LN}_1(H_{l-1}^{(j)}).
\]

LayerNorm 與 FFN 都只沿 feature 軸逐位置處理；位置 \(i\) 的 causal attention 只使用 \(j\le i\) 的 K/V；殘差只合併同一時間位置。因此由層數歸納可得，位置 \(i\) 的 logits 只依賴 \(X_0,\ldots,X_i\)。證明主體成立。

## 6. 手算

注意力第三列權重約為：

\[
(0.2482,0.5035,0.2482),
\]

故第三列輸出為：

\[
(0.4964,0.7517).
\]

FFN 計算：

\[
[1,2]W_1=[1,2,1,2],
\]

\[
[1,2,1,2]W_2=[2,5].
\]

均正確。

---

# 二、主要阻擋問題：PAD 梯度測試仍是假陽性

## 逐字原句

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

接著：

```python
pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
assert torch.all(pad_grad == 0)
```

表面上 PAD 已實際出現在 lookup 中，但這仍無法驗證 `padding_idx=0`。

原因如下：

1. PAD 位於最後位置 \(t=2\)。
2. 該位置 target 也是 PAD，所以其 loss 被 `ignore_index` 排除。
3. 位置 0、1 的有效 loss 因 causal mask 只能依賴位置不超過自身的輸入。
4. 最後位置的 PAD 不可能影響位置 0、1。
5. 因此從總 loss 到最後 PAD embedding 的計算圖沒有梯度路徑。

即使刪掉：

```python
padding_idx=self.PAD_ID
```

PAD row 在這個整合 loss 下仍會得到零梯度。故測試仍然無法辨識正確與故障實作。

## 最小修法

把「embedding padding row」測試與語言模型 loss 測試分開。直接構造一個會對所有 lookup 輸出求和的輔助標量：

```python
model.zero_grad(set_to_none=True)
emb = model.tok_emb(torch.tensor([[0, 1]], dtype=torch.long))
probe = emb.sum()
probe.backward()

assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

此時 PAD 與 token 1 都直接參與 `probe`。若沒有 `padding_idx=0`，PAD row 應收到非零梯度；有 `padding_idx=0` 時 PAD row 為零。因此這才是能區分正確與故障實作的測試。

然後另建一個新的模型或先清空梯度，再執行端到端 language-model loss 的梯度流測試。不要讓 embedding probe 的梯度污染後續參數檢查。

---

# 三、`view` 的文字說明仍不正確

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

目前程式：

```python
out.transpose(1, 2).contiguous().view(...)
```

是正確寫法。但說明仍有兩個問題：

- stride 不相容時，`view` 通常會拋錯，不應說可能靜默給出錯誤映射；
- 若改用 `reshape`，框架可在必要時建立副本，所以 `.contiguous()` 並非所有等價寫法中都不可省略。

## 最小修法

改成：

> 「本程式在 transpose 後使用 `view`，因此先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常與所求 shape 不相容，使 `view` 拋錯。若使用 `reshape`，框架可在必要時建立副本。」

---

# 四、因果性測試仍可能將 token 改成 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

若原 token 是 \(V-1\)，新值會變成 0，即 PAD。因其位於最右側，資料契約仍允許，但這使測試同時改變 token 內容和 padding 身分。

## 最小修法

改成：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

由於原 token 來自 \(1,\ldots,V-1\)，新值仍是非 PAD，且與原值不同。

此外，測試只比較最後 token 的改動。它能檢查一個常見故障，但更強的測試是任選分界 \(c\)，改變所有位置 \(c+1,\ldots,T-1\)，驗證 logits 的 \(0,\ldots,c\) 不變。數學一般性仍由歸納證明提供。

---

# 五、梯度流測試說明與程式不同步

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

程式實際檢查：

- embedding；
- Q/K/V；
- attention output projection；
- FFN 兩層；
- final output projection。

## 最小修法

將文字改為：

> 「檢查代表性 embedding、Q/K/V、注意力輸出投影、FFN 與 logits projection 的梯度存在、shape 與參數一致且全部有限；PAD row 則以獨立 embedding probe 驗證。」

如果學習目標堅持「所有參數」，可巡訪：

```python
for name, p in model.named_parameters():
    assert p.grad is not None
    assert p.grad.shape == p.shape
    assert torch.isfinite(p.grad).all()
```

不要求所有元素非零。

---

# 六、習題仍無法檢查 transpose

## 逐字原句

> 「輸入 \(X\in\mathbb R^{1\times2\times4}\)，\(H=2\)」

因為 \(T=H=2\)，reshape 後：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

即使學生漏掉 transpose，shape 的數字也完全相同。

## 最小修法

把 \(T\) 改為 3：

- projection：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改解答。

---

# 七、生成器缺少輸入契約

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

這要求：

- `vocab_size >= 2`
- `batch_size >= 1`
- `seq_len >= 1`

但函數沒有檢查。`vocab_size=1` 時取樣區間為空；空 batch 是否支援也未說明。

## 最小修法

加入：

```python
if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
if seq_len < 1:
    raise ValueError("seq_len must be >= 1")
if vocab_size < 2:
    raise ValueError("vocab_size must include PAD and a non-PAD token")
```

forward 若不支援 \(B=0\)，也應在 `b,t=idx.shape` 後拒絕空 batch。

---

# 八、反例與縮放敘述過強

## 逐字原句一

> 「因果性測試將失敗。」

上三角 mask 會破壞因果保證，但任意隨機初始化下，未來貢獻可能恰為零、抵消或低於容差。應說「存在未來依賴，通常預期失敗」，而不是單次測試必然觀察到失敗。

## 最小修法

改成：

> 「上三角 mask 破壞因果性保證；一般非退化權重下，修改未來 token 的測試預期產生非零差異。確定性反例應固定一組使未來 value 對較早輸出有非零貢獻的權重與輸入。」

## 逐字原句二

> 「Softmax 飽和，梯度不穩定。」

忘記 scale 不保證必然飽和。

## 最小修法

改成：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、來源語義仍應精確

N3 的 source notes 已核對特定 SDPA API／版本中布林 True 表示參與注意。稿中仍寫：

> 「框架的布林遮罩語義可能與本卷『True=允許注意』的約定不同」

對其他 API 或版本確實不能類推，但對已核對的 N3 本身，這句過度模糊。

## 最小修法

改成：

> 「N3 所列特定 SDPA API／版本中布林 True 表示參與注意，與本卷約定一致；其他 API、mask 參數或版本仍須分別核對。」

本章實際使用手寫 `masked_fill(~mask,-inf)`，其語義由程式本身明確定義。

---

# 十、執行證據與完整性

稿中寫：

> 「程式結果為預期行為，非已執行紀錄。」

這符合要求，沒有虛構已執行、已訓練、收斂率、時間或設備效能。程式中的 `print("Passed")` 是執行期訊息，只有實際走到該行才會顯示，不構成作者聲稱已執行。

本章目前已具備：

- 超過最低正文篇幅；
- 完整 decoder-only 模型；
- 完整 optimizer 與訓練 loop；
- 自足 CPU 程式；
- 正確主 shape 與 broadcast；
- 正確 causal mask 與 softmax 軸；
- 正確 PAD loss；
- 正確 input／target padding 契約；
- 正確 pre-norm 因果證明；
- 兩個正確手算；
- 正常、邊界與故障測試；
- 四類習題與答案；
- 無下載與外部模組；
- 合成資料及安全界線；
- 無虛構執行或能力。

距離核准僅剩局部問題，但 PAD embedding 測試仍不能區分有無 `padding_idx`，屬實質測試錯誤；`view` 說明仍有事實錯誤。修正這兩項，並同步處理因果 token 修改與生成器邊界後，才適合核准。

VERDICT: REVISE