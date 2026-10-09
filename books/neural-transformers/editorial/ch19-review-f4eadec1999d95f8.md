## 獨立審稿結論

本輪章稿與上一輪在實質內容上相同。先前已修正的兩項核心問題仍保持正確：targets 已受右側 padding 契約約束，且 pre-norm LayerNorm 已納入多層因果性證明。重新核算後，完整模型、資料、loss、optimizer、訓練 loop、注意力主 shape、兩個手算與有效 token reduction 均成立。

但仍有數個尚未修正的明確技術問題：attention 單一元素被標成四維張量；logits 被錯稱為「未歸一化對數概率」；`view` 被描述成可能靜默給出錯誤記憶體映射；PAD embedding 梯度測試仍未讓 PAD 出現在 lookup 中，因此是無效測試；新增的 input／target padding 契約沒有測試；因果測試仍可能把普通 token 改成 PAD。這些都可最小幅度修補，但在定義與測試仍不一致時不宜核准。

---

# 一、重新核算與可接受部分

## 1. 注意力手算正確

稿中的

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}
\]

正確。除以 \(\sqrt2\) 並施加 causal mask 後，三列 softmax 約為：

\[
(1,0,0),
\]

\[
(0.3302,0.6698,0),
\]

\[
(0.2482,0.5035,0.2482).
\]

所以第三列輸出是：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

稿中數值正確。

## 2. FFN 手算正確

\[
[1,2]W_1=[1,2,1,2],
\]

經 ReLU 不變，再乘 \(W_2\) 得：

\[
[1,2,1,2]W_2=[2,5].
\]

中間 shape \((1,4)\) 與輸出 shape \((1,2)\) 亦正確。

## 3. 有效 token loss 正確

程式使用：

```python
ignore_index=self.PAD_ID,
reduction='sum'
```

再除以：

```python
valid_count = targets.ne(self.PAD_ID).sum().item()
```

其結果是：

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

PAD 同時排除於分子及分母，且只平均一次。全 PAD target 也在除法前被拒絕。此部分正確。

## 4. padding 契約已正確補足

程式現在檢查：

- `idx` 只能右側連續 PAD；
- `targets` 只能右側連續 PAD；
- `idx==PAD` 的位置不得有有效 target。

新增的：

```python
if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
    raise ValueError("PAD input positions must have PAD targets")
```

方向正確。反方向不應強制，因為合法的序列結束可以是：

\[
X=[a,b,c],\qquad Y=[b,c,PAD].
\]

## 5. pre-norm 因果性證明已正確修補

證明現在明列：

\[
\widetilde H_{l-1}^{(j)}
=
\operatorname{LN}_1(H_{l-1}^{(j)}),
\]

並說明 LayerNorm 沿 feature 軸逐位置運算，不擴大時間依賴集合。FFN 子層也使用：

\[
H_l^{(i)}
=
\operatorname{FFN}(\operatorname{LN}_2(H_{\mathrm{mid}}^{(i)}))
+
H_{\mathrm{mid}}^{(i)}.
\]

因此證明與程式的 pre-norm 結構一致。此項不再構成問題。

---

# 二、必須修正：attention 元素與張量 shape 混寫

## 逐字原句

> \(A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T}\)

若 \(A_{ij}\) 表示單一 query-key 權重，它仍省略 batch 與 head 索引；若所有索引固定，它是 scalar。只有整體張量 \(A\) 才屬於 \(\mathbb R^{B\times H\times T\times T}\)。

這一章的核心之一就是 shape 與 reduction，不能讓元素和張量型別混在同一式中。

## 最小修法

改為：

\[
A\in\mathbb R^{B\times H\times T\times T},
\]

且

\[
A_{b,h,i,j}
=
\frac{
\exp(S'_{b,h,i,j})
}{
\sum_{k=0}^{T-1}
\exp(S'_{b,h,i,k})
}.
\]

此時分母明確沿最後的 key 軸 \(k\) 求和。

前面的 masked score 最好同步寫成：

\[
S'_{b,h,i,j}
=
\begin{cases}
Q_{b,h,i,:}\cdot K_{b,h,j,:}/\sqrt{d_h},&j\le i,\\
-\infty,&j>i.
\end{cases}
\]

---

# 三、必須修正：logits 的概率語義錯誤

## 逐字原句

> 「表示每個位置對詞彙表中每個詞元的未歸一化對數概率。」

logits 是未正規化的實數分數，不必先被定義成某個概率的對數。沿詞彙軸套用 softmax 後才得到概率：

\[
p_v
=
\frac{e^{z_v}}{\sum_u e^{z_u}},
\]

套用 log-softmax 後才得到對數概率：

\[
\log p_v
=
z_v-\log\sum_u e^{z_u}.
\]

## 最小修法

改成：

> 「Logits \(Z\in\mathbb R^{B\times T\times V}\) 是每個位置對詞彙中各 token 的未正規化實數分數；沿詞彙軸套用 softmax 後得到條件機率。」

這也與 `cross_entropy` 直接接收 logits 的程式契約一致。

---

# 四、必須修正：`view` 行為描述不正確

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

目前程式：

```python
out.transpose(1, 2).contiguous().view(...)
```

是正確且清楚的。但 `view` 面對不相容 stride 時通常會拋錯，不應描述成可能靜默給出錯誤映射。另外，如果改用 `reshape`，框架可在需要時建立副本，所以 `.contiguous()` 並非所有寫法下都絕對不可省略。

## 最小修法

改成：

> 「本程式在 transpose 後使用 `view`，所以先呼叫 `.contiguous()`；否則 stride 通常與所求 shape 不相容而使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

---

# 五、PAD embedding 梯度測試仍是假陽性

## 逐字原句

```python
x_grad = torch.randint(1, V, (2, 5))
```

隨後：

```python
pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
assert torch.all(pad_grad == 0)
```

`torch.randint(1,V,...)` 永遠不會產生 PAD_ID 0。因此 PAD embedding row 完全沒有參與 lookup。即使移除 `padding_idx=0`，未被索引的第 0 row 梯度仍然為零。這個 assert 不能驗證 `padding_idx` 的效果。

## 最小修法

使用實際含 PAD 且符合右側 padding 契約的資料：

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

在 backward 前明確清空梯度：

```python
model.zero_grad(set_to_none=True)
```

再檢查：

```python
assert torch.all(model.tok_emb.weight.grad[0] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

此外保留 Q/K/V、output projection、FFN 及 logits projection 的 `grad is not None`、同 shape 與 finite 檢查。

如果不想依賴特定 token row 必為非零，至少應檢查出現過的所有非 PAD row 中存在一個非零 row，而不是要求每一 row 都非零。

---

# 六、新增的 padding 契約仍沒有測試

本輪程式新增 target 右側 padding及 input-target 相容檢查，但 `run_tests()` 沒有任何案例觸發這些分支。這會讓未來回歸無法被發現。

## 最小必要測試

### 1. 合法右側 padding

```python
idx = torch.tensor([[1, 2, 0, 0]])
targets = torch.tensor([[2, 3, 0, 0]])
```

預期接受並得到有限 loss。

### 2. input 內部 PAD

```python
idx = torch.tensor([[1, 0, 2, 0]])
```

預期拋出右側 padding 的 `ValueError`。

### 3. target 內部 PAD

```python
idx = torch.tensor([[1, 2, 3, 4]])
targets = torch.tensor([[2, 0, 3, 0]])
```

預期拋出 target padding 錯誤。

### 4. PAD input 對應有效 target

```python
idx = torch.tensor([[1, 2, 0]])
targets = torch.tensor([[2, 3, 4]])
```

預期拋出：

> `"PAD input positions must have PAD targets"`

### 5. 合法結束 target

```python
idx = torch.tensor([[1, 2, 3]])
targets = torch.tensor([[2, 3, 0]])
```

預期接受。這可防止日後錯誤加入反向條件。

---

# 七、因果測試仍可能改出 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

當最後 token 為 \(V-1\) 時，新值會變成 0，即 PAD。它因位於右端而仍符合契約，但測試同時改變 token 內容與 padding 身分。

## 最小修法

保持修改後仍為非 PAD：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

對 \(1,\ldots,V-1\) 的任何輸入，結果都會是另一個有效 token。

此外，這個有限 seed、有限容差測試只驗證特定程式案例。一般性的無未來洩漏由本章歸納證明支持，兩者不可混為同一證據。

---

# 八、測試說明與程式不一致

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

程式實際還檢查：

- embedding；
- Q/K/V projection；
- attention output projection；
- FFN `fc1`、`fc2`；
- final output projection。

## 最小修法

文字同步改成：

> 「檢查代表性 embedding、Q/K/V、注意力輸出投影、FFN 與 logits projection 參數的梯度存在、shape 與參數一致且全部有限；另用真的含 PAD 的輸入檢查 PAD embedding row 梯度為零。」

若學習目標堅持「所有參數」都具有梯度，則最直接的測試是：

```python
for name, p in model.named_parameters():
    assert p.grad is not None
    assert p.grad.shape == p.shape
    assert torch.isfinite(p.grad).all()
```

非零性只對人工固定的非退化案例檢查，不應要求所有元素都非零。

---

# 九、習題仍無法辨識 transpose

## 逐字原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

因為 \(T=H=2\)，reshape 前後的數字 shape 都是：

\[
(1,2,2,2).
\]

雖然軸語義不同，但答案僅看 shape 數字時無法發現漏掉 transpose。

## 最小修法

把 \(T\) 改成 3：

- projection：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改答案。這能真正檢驗 reshape 與 transpose 的差異。

---

# 十、反例與 scale 的因果措辭仍過強

## 逐字原句一

> 「因果性測試將失敗。」

錯誤上三角 mask 會破壞因果保證，但單次隨機初始化可能因權重退化、貢獻抵消或差異低於容差而沒有觀察到失敗。

## 最小修法

改成：

> 「上三角 mask 允許未來依賴，因此因果保證失效；一般非退化權重下測試預期失敗。確定性故障測試應固定一組使未來 value 對過去輸出具有非零貢獻的權重與輸入。」

## 逐字原句二

> 「Softmax 飽和，梯度不穩定。」

忘記縮放不保證必然飽和。此論證依賴 Q/K 分量的方差近似。

## 最小修法

改成：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差隨 \(d_k\) 增長，可能令 softmax 過尖並使部分梯度變小或不穩定。」

---

# 十一、生成器與空 batch 邊界尚未定義

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

此函數要求 `vocab_size>=2`，但沒有檢查。`batch_size<=0` 或 `seq_len<=0` 亦未定義。

## 最小修法

加入：

```python
if batch_size < 1 or seq_len < 1 or vocab_size < 2:
    raise ValueError(...)
```

forward 也沒有拒絕空 batch \(B=0\)。若不支援，應加：

```python
if b < 1:
    raise ValueError("Batch size must be >= 1")
```

並加入故障測試。

---

# 十二、來源與執行宣稱

稿中明確寫：

> 「程式結果為預期行為，非已執行紀錄。」

此處符合規則，沒有虛構已執行。程式中的 `print("Passed")` 只有在使用者未來實際執行並到達該行時才輸出，不等同作者聲稱本次已執行。

N3 的 source notes 已明確說特定 SDPA 版本中布林 True 表示參與注意。正文的：

> 「框架的布林遮罩語義可能與本卷……不同」

應限定為其他 API、不同 mask 參數或不同版本不可直接類推；不可讓讀者誤以為已核對的 N3 本身與本卷約定相反。本章實際手寫 `masked_fill(~mask,-inf)`，不依賴 SDPA。

---

# 十三、總評

本稿目前已滿足多數核心條件：

- 正文達最低字數；
- 模型與訓練 loop 完整；
- 不依賴未提供模組；
- 無下載、GPU、shell 或網路；
- 主 shape 與 broadcast 正確；
- next-token 位移正確；
- PAD loss 正確；
- input／target 右側 padding 契約正確；
- 因果證明已對齊 pre-norm；
- 兩個手算正確；
- 正常、邊界、故障測試存在；
- 四類習題及答案存在；
- 沒有虛構已訓練或收斂結果；
- 養殖案例明確限於合成資料且不控制設備。

剩餘問題均可局部修改，不必重寫章稿；但 attention shape 與 logits 定義屬正文數學錯誤，PAD 梯度測試與未測 padding 契約則屬故障測試不充分。修正後才適合核准。

VERDICT: REVISE