## 獨立審稿結論

本輪稿件與上一輪的實質內容相同：logits 定義、attention 權重 shape、target padding 契約及 pre-norm 因果證明均已修正；但上一輪留下的程式測試與張量 layout 問題仍未處理。重新核算後，模型本身、loss、主 shape、因果 mask、兩個手算與訓練 loop 沒有發現新的致命錯誤。

目前阻擋核准的核心是「宣稱測到但實際未測到」：PAD embedding 梯度測試沒有讓 PAD 出現在輸入，因此無法驗證 `padding_idx`；新加入的 input／target 右側 padding 契約完全沒有測試；因果測試仍可能把一般 token 改成 PAD。此外，正文仍錯誤描述非連續張量 `view` 可能產生錯誤映射。依本卷要求核對自足程式及正常／邊界／故障測試，仍需修訂。

---

# 一、重新核算確認正確之處

## 1. 端到端 shape

程式的實際 shape 流為：

- `idx`: \((B,T)\)
- `tok_emb(idx)`: \((B,T,D)\)
- `pos_emb[:, :t, :]`: \((1,T,D)\)
- 相加時沿 batch 軸廣播：\((B,T,D)\)
- Q/K/V 線性投影：\((B,T,D)\)
- `view(B,T,H,d_h)`：\((B,T,H,d_h)\)
- `transpose(1,2)`：\((B,H,T,d_h)\)
- scores：\((B,H,T,T)\)
- mask：\((T,T)\)，向 batch 與 head 軸廣播
- attention output：\((B,H,T,d_h)\)
- 合頭：\((B,T,D)\)
- logits：\((B,T,V)\)

程式中的主 shape 正確，且 softmax 沿最後 key 軸 `dim=-1`。

## 2. attention 定義已修正

現在寫為：

\[
A\in\mathbb R^{B\times H\times T\times T},
\]

\[
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_{k=0}^{T-1}\exp(S'_{b,h,i,k})}.
\]

整體張量與 scalar 元素已正確區分，reduction 軸也明確。

## 3. loss 計算

程式使用 `ignore_index=PAD_ID, reduction="sum"`，再除以有效 target 數。因此：

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

PAD 不進入分子，分母也只計有效 target，平均只做一次。全 PAD target 在除法前拒絕。正確。

## 4. padding 契約

目前程式正確實作三條條件：

1. input PAD 只能在右側形成連續後綴；
2. target PAD 只能在右側形成連續後綴；
3. `idx==PAD` 的位置不得有有效 target。

第三條採單向蘊含是正確的。合法序列可以是：

\[
X=[a,b,c],\qquad Y=[b,c,PAD],
\]

所以不能反向要求 target PAD 時 input 也必須 PAD。

## 5. 因果性證明

證明已正確納入：

\[
\widetilde H_{l-1}^{(j)}
=
\operatorname{LN}_1(H_{l-1}^{(j)}).
\]

LayerNorm 和 FFN 只混合 feature 軸，不混合 time 軸；causal attention 在位置 \(i\) 只使用 \(j\le i\) 的 K/V；殘差只相加同一位置。因此歸納證明成立。

## 6. 兩個手算

注意力第三列權重約為：

\[
(0.2482,0.5035,0.2482),
\]

輸出：

\[
(0.4964,0.7517).
\]

FFN 輸出為：

\[
[1,2]W_1=[1,2,1,2],\qquad
[1,2,1,2]W_2=[2,5].
\]

均正確。

---

# 二、必須修正：非連續張量 `view` 的行為描述仍錯

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

本程式使用：

```python
out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
```

此程式寫法本身正確。問題在說明：

- transpose 後 stride 通常不符合合頭所需的 `view`；
- 此時 `view` 應拋錯，而不是靜默給出錯誤映射；
- 若改用 `reshape`，框架可以在必要時建立副本，顯式 `.contiguous()` 並非所有寫法中都不可省略。

## 最小修法

改成：

> 「本程式在 transpose 後使用 `view`，因此先呼叫 `.contiguous()`；否則 stride 通常與所求 shape 不相容，使 `view` 拋錯。若使用 `reshape`，框架可在必要時建立副本。」

不必更動目前程式。

---

# 三、必須修正：PAD embedding 梯度測試是假陽性

## 逐字原句

```python
x_grad = torch.randint(1, V, (2, 5))
```

然後：

```python
pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
assert torch.all(pad_grad == 0)
```

`torch.randint(1,V,...)` 的下界是 1，因此 input 不含 PAD_ID 0。PAD embedding row 沒有參與 lookup；任何未被索引的 embedding row 梯度本來就為零。

也就是說，即使把：

```python
padding_idx=self.PAD_ID
```

刪掉，這個測試仍可能通過。它不能證明 `padding_idx` 阻止 PAD row 接收 lookup 梯度。

## 最小修法

改用真正含 PAD 的合法資料：

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

在 backward 前加入：

```python
model.zero_grad(set_to_none=True)
```

再檢查：

```python
assert torch.all(model.tok_emb.weight.grad[0] == 0)
```

並對實際出現的非 PAD row 檢查至少有一個非零梯度，例如：

```python
used = torch.unique(x_grad[x_grad != 0])
assert torch.any(model.tok_emb.weight.grad[used] != 0)
```

不能要求所有參數元素均非零，因為合法抵消、對稱或特定輸入可產生零梯度。

---

# 四、新增的 padding 契約沒有測試

程式已加入 input 與 target 的 right-padding 檢查，但 `run_tests()` 沒有覆蓋任何相關錯誤分支。現有 PAD loss 測試只有 target 最後一項是 PAD，input 完全沒有 PAD；它只能測 `ignore_index`，不能測資料契約。

## 最小測試集合

### 正常案例

```python
idx = torch.tensor([[1, 2, 0, 0]])
targets = torch.tensor([[2, 3, 0, 0]])
```

預期接受，loss 有限。

### input 內部 PAD

```python
idx = torch.tensor([[1, 0, 2, 0]])
```

預期拋出 input right-padding 錯誤。

### target 內部 PAD

```python
idx = torch.tensor([[1, 2, 3, 4]])
targets = torch.tensor([[2, 0, 3, 0]])
```

預期拋出：

> `"Target PAD tokens must be at the right side."`

### PAD input 對應有效 target

```python
idx = torch.tensor([[1, 2, 0]])
targets = torch.tensor([[2, 3, 4]])
```

預期拋出：

> `"PAD input positions must have PAD targets"`

### 合法序列結束

```python
idx = torch.tensor([[1, 2, 3]])
targets = torch.tensor([[2, 3, 0]])
```

預期接受。這一例可防止日後誤加反向約束。

由於本章以「只允許右側 padding」取代一般 padding key mask，該契約是模型正確性的前提，不能只有程式分支而沒有故障測試。

---

# 五、因果性測試仍可能把普通 token 改成 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

如果原 token 是 \(V-1\)，修改後是 0，也就是 PAD。雖然最右側 PAD 仍符合 right-padding 契約，但測試同時改變了：

- token 值；
- 是否為 PAD。

這讓「只改變未來內容 token」的測試不夠純粹。

## 最小修法

使用：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

原值位於 \(1,\ldots,V-1\)，新值仍位於該範圍，而且與原值不同。

另應明說：`diff < 1e-5` 是特定實作的故障測試，不是因果性的普遍證明；一般性由前面的歸納證明提供。

---

# 六、梯度測試說明與實際範圍不一致

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

程式實際還檢查：

- token embedding；
- Q/K/V projection；
- attention output projection；
- FFN `fc1` 與 `fc2`；
- final output projection。

## 最小修法

測試說明應改為：

> 「檢查代表性 embedding、Q/K/V、注意力輸出投影、FFN 與 logits projection 的梯度存在、shape 與參數一致且全部有限；另以真的含 PAD 的輸入檢查 PAD embedding row 梯度為零。」

如果要符合學習目標中「所有參數必須具有正確梯度」的字面要求，可巡訪：

```python
for name, p in model.named_parameters():
    assert p.grad is not None
    assert p.grad.shape == p.shape
    assert torch.isfinite(p.grad).all()
```

但非零測試只應用在人工固定的非退化參數，不應要求所有元素非零。

---

# 七、手算習題仍無法測出漏做 transpose

## 逐字原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

因為 \(T=H=2\)，reshape 後：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

兩者數字相同。即使學生漏做 transpose，仍可能寫出一樣的 shape，無法驗證軸語義。

## 最小修法

改用 \(T=3\)：

- Q/K/V 投影：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改解答。

這不是模型程式錯誤，但與本章核心「reshape 不是 transpose」直接相關。

---

# 八、反例與縮放敘述仍過度絕對

## 逐字原句一

> 「因果性測試將失敗。」

上三角 mask 確實破壞因果性保證，但單次隨機模型可能因未來投影為零、抵消或小於容差而沒有觀察到差異。

## 最小修法

改成：

> 「上三角 mask 允許未來依賴，因此因果性保證失效；一般非退化權重下，該測試預期失敗。若要建立確定反例，應固定使未來 value 對較早輸出有非零貢獻的權重與輸入。」

## 逐字原句二

> 「Softmax 飽和，梯度不穩定。」

未縮放不保證必然飽和。該論證依賴 Q/K 分量的方差假設。

## 最小修法

改成：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用近似下，未縮放 score 的方差隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、資料生成器邊界未驗證

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

此函數要求：

- `vocab_size >= 2`
- `batch_size >= 1`
- `seq_len >= 1`

但沒有輸入檢查。模型建構器允許 `vocab_size=1`，生成器卻無法從空區間 \([1,1)\) 取樣。

## 最小修法

加入：

```python
if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
if seq_len < 1:
    raise ValueError("seq_len must be >= 1")
if vocab_size < 2:
    raise ValueError("vocab_size must contain PAD and a non-PAD token")
```

forward 也沒有拒絕空 batch \(B=0\)。若空 batch 不在支援範圍，加入：

```python
if b < 1:
    raise ValueError("Batch size must be >= 1")
```

並增加故障測試。

---

# 十、來源與執行宣稱

稿中明確寫：

> 「程式結果為預期行為，非已執行紀錄。」

因此沒有虛構實際執行、訓練成功、收斂率或硬體效能。程式內的 `print("Passed")` 是未來執行到該行後的訊息，不等於作者聲稱已在寫作流程中執行。

N3 的 source notes 已核對特定 SDPA API／版本中布林 True 表示參與注意。正文：

> 「框架的布林遮罩語義可能與本卷……不同」

應限定為其他 API、不同 mask 參數或其他版本不可直接類推。本章實際使用手寫 `masked_fill(~mask,-inf)`，其 True=允許由程式本身定義，不直接依賴 SDPA。

---

# 十一、總評

已符合的核心要求包括：

- 正文超過最低 3000 中文字；
- 模型、資料、loss、optimizer、loop 完整；
- 不依賴未提供模組；
- 無下載、GPU、shell 或網路；
- 主 shape、broadcast、mask 與 reduction 正確；
- next-token 資料錯位正確；
- PAD loss 正確；
- input／target padding 契約正確；
- pre-norm 因果證明完整；
- 兩個手算正確；
- 四類習題及答案存在；
- 正常、邊界與故障測試已有基本集合；
- 無虛構執行或能力；
- 合成資料與養殖安全範圍清楚。

剩餘問題都是局部可修，但 PAD 梯度測試目前沒有測到它聲稱驗證的性質，而右側 padding 是取代 key mask 的核心前提卻完全未被測試；再加上 `view` 行為的錯誤說明，仍需修訂後才可核准。

VERDICT: REVISE