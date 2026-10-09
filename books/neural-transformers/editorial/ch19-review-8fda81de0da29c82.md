## 獨立審稿結論

本輪已正確修正兩項上一輪的正文錯誤：

1. logits 現在被定義為「未正規化實數分數」，不再錯稱為未歸一化對數概率。
2. attention 權重現在正確區分整體張量 \(A\in\mathbb R^{B\times H\times T\times T}\) 與 scalar 元素 \(A_{b,h,i,j}\)。

連同前輪已修正的 target padding 契約與 pre-norm 因果證明，本章數學主體已接近完成。重新核算後，模型、loss、主 shape、因果性、兩個手算與訓練 loop 均成立。

目前仍有一項明確錯誤及若干驗收缺口：`.contiguous()`／`view` 的行為敘述仍錯；PAD embedding 梯度測試仍是假陽性；新增的右側 padding 契約沒有被測試；因果測試仍可能把普通 token 改成 PAD；習題仍因 \(T=H\) 而無法辨識 transpose；生成器邊界未驗證。尤其 PAD 梯度測試聲稱驗證一項實際未測到的性質，依本卷故障測試契約仍應修訂。

---

# 一、重新核算後已正確的內容

## 1. logits 定義已修正

新原句：

> 「Logits \(Z\in\mathbb R^{B\times T\times V}\) 是每個位置對詞彙表中各詞元的未正規化實數分數；沿詞彙軸套用 softmax 後得到條件機率。」

這是正確定義。若一個位置的 logits 為 \(z_v\)，則：

\[
p_v=\frac{\exp(z_v)}{\sum_u\exp(z_u)}.
\]

`cross_entropy` 直接接受 logits，也與此定義一致。

## 2. attention 元素與張量 shape 已修正

新公式：

\[
A\in\mathbb R^{B\times H\times T\times T},
\qquad
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_{k=0}^{T-1}\exp(S'_{b,h,i,k})}
\]

正確區分整體張量與 scalar 元素，並明確寫出 softmax 沿最後 key 軸 \(k\) reduction。此項已不再構成問題。

## 3. 有效 token loss

程式使用：

```python
ignore_index=self.PAD_ID,
reduction='sum'
```

並除以：

```python
valid_count = targets.ne(self.PAD_ID).sum().item()
```

所以實際 loss 是：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf 1[Y_{b,t}\ne PAD]
}.
\]

分子、分母採同一 target mask，只平均一次；全 PAD target 也在除法前被拒絕。正確。

## 4. input／target padding 契約

目前程式同時檢查：

- `idx` 只能右側連續 PAD；
- targets 只能右側連續 PAD；
- PAD input 位置不得對應有效 target。

其單向條件：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD
\]

是正確的。不能加入反向條件，因為合法的序列結束可以是：

\[
X=[a,b,c],\qquad Y=[b,c,PAD].
\]

## 5. 因果性證明

證明已把：

\[
\widetilde H_{l-1}^{(j)}
=
\operatorname{LN}_1(H_{l-1}^{(j)})
\]

納入 Q/K/V 的來源，也正確指出 LayerNorm 與 FFN 不混合 time 軸。注意力權重依賴 \(Q_i\) 及所有 \(j\le i\) 的 \(K_j\)，而這些前層表示依歸納假設都只依賴 prefix。因此多層因果性結論成立。

## 6. 手算

注意力第三列權重約為：

\[
(0.2482,0.5035,0.2482),
\]

故輸出：

\[
(0.4964,0.7517).
\]

FFN 輸出為：

\[
[1,2]W_1=[1,2,1,2],\qquad
[1,2,1,2]W_2=[2,5].
\]

兩例均正確。

---

# 二、仍須修正：`view` 不會靜默給出錯誤映射

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

目前程式：

```python
out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
```

本身正確。但文字有兩個問題：

1. 對 stride 不相容的非連續張量，`view` 通常會拋錯，不應描述成可能靜默產生錯誤映射。
2. 若使用 `reshape`，框架可以在必要時建立副本，所以 `.contiguous()` 並非所有等價寫法中都不可省略。

## 最小修法

改成：

> 「本程式在 transpose 後使用 `view`，因此先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常與所求 shape 不相容，使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

這不需改程式，只改一段說明。

---

# 三、PAD embedding 梯度測試仍是假陽性

## 逐字原句

```python
x_grad = torch.randint(1, V, (2, 5))
```

以及：

```python
pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
assert torch.all(pad_grad == 0)
```

由於 `torch.randint(1,V,...)` 不可能產生 0，PAD row 從未參與 embedding lookup。任何未被索引的 embedding row，其 lookup 梯度原本就是零。即使刪掉：

```python
padding_idx=self.PAD_ID
```

這個測試仍可能通過，因此它沒有驗證所聲稱的「padding_idx 使 PAD row 不由 lookup 更新」。

## 最小修法

使用真的包含 PAD、且符合右側 padding 契約的輸入：

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

在 backward 前清空梯度：

```python
model.zero_grad(set_to_none=True)
```

然後檢查：

```python
pad_grad = model.tok_emb.weight.grad[0]
assert torch.all(pad_grad == 0)
```

再檢查至少一個實際出現的非 PAD row 有梯度：

```python
used = torch.unique(x_grad[x_grad != 0])
assert torch.any(model.tok_emb.weight.grad[used] != 0)
```

不應要求每個出現 token 的每個梯度元素都非零，因為可能存在抵消或合法零值。

原有 Q/K/V、attention output projection、FFN 及 output projection 的 `.grad is not None`、shape 與 finite 檢查可保留。

---

# 四、padding 契約已實作但未測試

程式已新增重要故障分支，但 `run_tests()` 沒有任何案例觸發它們。這不會令模型本身錯誤，卻使「故障測試」無法驗證章中著重的資料契約。

## 最小必要測試

### 1. 合法右側 padding

```python
idx = torch.tensor([[1, 2, 0, 0]])
targets = torch.tensor([[2, 3, 0, 0]])
```

預期接受且 loss 有限。

### 2. 非法 input 內部 PAD

```python
idx = torch.tensor([[1, 0, 2, 0]])
```

預期拋出 input right-padding 錯誤。

### 3. 非法 target 內部 PAD

```python
idx = torch.tensor([[1, 2, 3, 4]])
targets = torch.tensor([[2, 0, 3, 0]])
```

預期拋出：

> `"Target PAD tokens must be at the right side."`

### 4. PAD input 對應有效 target

```python
idx = torch.tensor([[1, 2, 0]])
targets = torch.tensor([[2, 3, 4]])
```

預期拋出：

> `"PAD input positions must have PAD targets"`

### 5. 合法序列結束

```python
idx = torch.tensor([[1, 2, 3]])
targets = torch.tensor([[2, 3, 0]])
```

預期接受。此案例可防止日後錯誤加入 `target PAD -> input PAD` 的反向限制。

目前的 PAD loss 測試只有 target 最後一項為 PAD，而 input 沒有 PAD；它能驗證 `ignore_index`，但不能代替上述資料契約測試。

---

# 五、因果測試仍可能引入 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

若原 token 是 \(V-1\)，新 token 會成為 0，即 PAD。因為它在最右側，仍符合右側 padding，但測試同時改變：

- token 身分；
- padding 身分。

這使測試目的不夠單純。

## 最小修法

保持修改結果在 \(1,\ldots,V-1\)：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

對任何原有效 token，此式都會得到另一個有效 token。

此外，`diff < 1e-5` 是特定模型、seed 與 dtype 下的實作測試，不是一般因果性的證明。一般結論已由歸納證明支持；測試只用於抓 mask 實作錯誤。

---

# 六、梯度測試說明仍與程式不同步

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

程式實際檢查了：

- token embedding；
- Q/K/V projection；
- attention output projection；
- FFN 的兩層；
- final output projection。

## 最小修法

把說明改成：

> 「執行 backward，檢查代表性 embedding、Q/K/V、注意力輸出投影、FFN 與 logits projection 的梯度存在、shape 與參數一致且全部有限；另用真的含 PAD 的輸入檢查 PAD embedding row 梯度為零。」

若正文堅持「所有參數必須具有正確梯度」，可直接巡訪：

```python
for name, p in model.named_parameters():
    assert p.grad is not None
    assert p.grad.shape == p.shape
    assert torch.isfinite(p.grad).all()
```

非零性則只應針對固定非退化案例中的少數參數檢查。

---

# 七、習題仍掩蓋 reshape／transpose 差異

## 逐字原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

這裡 \(T=H=2\)，因此：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

而 transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

shape 的數字完全相同。學生即使忘記 transpose，仍可能寫出相同答案，無法驗證軸順序。

## 最小修法

將序列長度改成 \(T=3\)：

- projection：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改答案。

此問題不破壞模型程式，但與本章「reshape 不是 transpose」的核心教學目標直接相關。

---

# 八、反例答案仍把「存在失敗」寫成「必然觀察失敗」

## 逐字原句

> 「因果性測試將失敗。」

使用上三角 mask 會允許未來依賴，因此因果性保證確實失效。但對任意隨機權重，未來路徑可能因投影為零、貢獻抵消或低於容差而未被單次數值測試觀察到。

## 最小修法

改成：

> 「上三角 mask 破壞因果性保證；一般非退化權重下，修改未來 token 的測試預期產生非零差異。要建立確定反例，應固定一組使未來 value 對較早輸出具有非零貢獻的 Q/K/V 或投影權重。」

同樣，原句：

> 「Softmax 飽和，梯度不穩定。」

也過於絕對。應改為：

> 「在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 九、生成器與空 batch 的邊界未定義

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

這隱含要求：

- `vocab_size >= 2`
- `batch_size >= 1`
- `seq_len >= 1`

但生成器沒有檢查。若 `vocab_size=1`，取樣區間 \([1,1)\) 為空。

## 最小修法

加入：

```python
if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
if seq_len < 1:
    raise ValueError("seq_len must be >= 1")
if vocab_size < 2:
    raise ValueError("vocab_size must include PAD and at least one non-PAD token")
```

forward 也未拒絕 \(B=0\)。如果不支援空 batch，應在取得 `b,t` 後加入：

```python
if b < 1:
    raise ValueError("Batch size must be >= 1")
```

並提供故障測試。

---

# 十、來源與執行證據

稿中明確寫：

> 「程式結果為預期行為，非已執行紀錄。」

因此沒有虛構本地執行、訓練成功或收斂結果。程式裡的 `print("Passed")` 是未來執行到該行時才會顯示的訊息，不是作者宣稱本次已執行。

N3 的 source notes 已指出特定 SDPA API／版本中布林 True 表示參與注意。正文仍寫：

> 「框架的布林遮罩語義可能與本卷……不同」

最小修法是限定範圍：N3 所列特定 SDPA API 中 True 表示參與，但其他 API、不同 mask 參數或其他版本不可直接類推。本章手寫 `masked_fill(~mask,-inf)`，本身清楚採 True=允許，不直接使用 SDPA。

---

# 十一、完整性總評

本章目前已具備：

- 超過最低 3000 中文字；
- 完整 decoder-only 模型；
- 自足資料、loss、optimizer 與訓練 loop；
- 無下載、GPU 或未提供模組；
- embedding、位置、MHA、pre-norm、FFN 與 logits 主鏈；
- 正確的 causal mask 與 softmax 軸；
- 正確的 input／target 右側 padding 契約；
- 正確的有效 token 平均；
- 完整多層因果性證明；
- 兩個正確手算；
- 正常、邊界與故障測試；
- 四類習題與答案；
- 合成資料及能力限制；
- 養殖案例安全界線；
- 無虛構執行或能力。

本輪已修正數學定義的主要問題，距離核准很近。但 PAD embedding 梯度測試目前仍不能驗證其文字宣稱，padding 契約也缺少任何對應測試；加上仍存在錯誤的 `view` 行為說明，因此尚不宜核准。這些均是局部修訂，不需重寫章稿。

VERDICT: REVISE