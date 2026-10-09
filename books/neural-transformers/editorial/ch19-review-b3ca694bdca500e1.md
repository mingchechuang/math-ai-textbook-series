## 獨立審稿結論

本輪已修正上一版最重要的 PAD loss 錯誤：`ignore_index=PAD_ID` 使 PAD 不再進入 NLL 分子，`valid_count` 又以相同 target mask 計算分母，數學公式、程式與手動核對測試現在一致。多層因果性證明亦補足了注意力權重對 \(Q_i\) 與 \(K_j\) 的依賴，主要推理成立。模型、optimizer、資料產生器及多步訓練 loop 均已自足列出，沒有省略核心組件。

目前仍不可核准，主要原因有二。第一，正文測得約 2572 中文字，仍低於本卷每章最低 3000 字的硬性要求。第二，模型接受含 PAD 的 `idx`，但注意力只實作 causal mask，沒有 padding key mask，也沒有明定只接受右側 padding 的資料契約；這使 PAD 位置可能作為 key/value 影響後續有效 token。除此之外，尚有 reshape／transpose 說明錯誤、測試覆蓋與文字宣稱不一致、來源引用不精確等問題。

---

# 一、先行重算

## 1. 有效 token loss 已修正正確

程式現在計算：

```python
valid_mask = targets.ne(self.PAD_ID)
valid_count = valid_mask.sum().item()
```

並使用：

```python
ce_loss_sum = nn.functional.cross_entropy(
    logits.reshape(-1, self.vocab_size),
    targets.reshape(-1),
    ignore_index=self.PAD_ID,
    reduction='sum'
)
loss = ce_loss_sum / valid_count
```

令有效位置集合為

\[
\mathcal I=\{(b,t):Y_{b,t}\ne PAD\},
\]

則 `cross_entropy(..., ignore_index=PAD, reduction="sum")` 對應

\[
\mathcal L_{\mathrm{sum}}
=\sum_{(b,t)\in\mathcal I}
-\log p(Y_{b,t}\mid X_{b,0:t}),
\]

而 `valid_count` 是 \(|\mathcal I|\)，故

\[
\mathcal L=\frac{\mathcal L_{\mathrm{sum}}}{|\mathcal I|}.
\]

分子、分母使用同一 target mask，且只除一次，現在正確。

測試中的：

```python
valid_logits = logits_pad.reshape(-1, V)[targets_pad.reshape(-1) != 0]
expected_loss = nn.functional.cross_entropy(
    valid_logits, valid_targets, reduction='mean'
)
```

與上述定義等價，因此能抓出「只排除分母、沒有排除分子」的錯誤。這部分可以保留。

## 2. 注意力與 FFN 手算正確

注意力例題中：

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}
\]

正確。第三列 softmax 更精確約為

\[
(0.2483,0.5035,0.2483),
\]

因此輸出約為

\[
(0.4966,0.7518).
\]

稿中的 \((0.4964,0.7517)\) 是近似誤差，不構成實質問題。

FFN 例題中：

\[
[1,2]W_1=[1,2,1,2],\qquad
[1,2,1,2]W_2=[2,5],
\]

數值與 shape 均正確。

## 3. 模型 shape 正確

實作中的 shape 流為：

\[
(B,T)\to(B,T,D),
\]

位置參數 \((1,T,D)\) 沿 batch 軸廣播後仍為 \((B,T,D)\)。Q/K/V 經：

```python
.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
```

得到

\[
(B,H,T,d_h).
\]

分數為

\[
(B,H,T,T),
\]

二維 causal mask \((T,T)\) 對齊最後兩軸，廣播到 batch 與 head 軸。合併後回到 \((B,T,D)\)，logits 為 \((B,T,V)\)。核心 shape 沒有錯。

---

# 二、必須修正：padding key mask 或明確限制資料契約

## 問題原句

程式只建立：

> `mask = torch.tril(torch.ones((t, t), device=x.device)).bool()`

而模型明確保留：

> `self.PAD_ID = 0`

並允許 `idx` 包含 0。

`padding_idx=0` 只影響 token embedding lookup 的 PAD row 梯度，不會自動建立 attention key mask。PAD 位置還會加上：

```python
pos = self.pos_emb[:, :t, :]
h = tok + pos
```

因此 PAD 位置的隱狀態一般不是零。即使 token embedding 的 PAD row 是零，位置 embedding、殘差、LayerNorm 與 FFN 仍會使該位置具有非零 K/V。

若輸入為內部 padding，例如

\[
[a,\mathrm{PAD},b,c],
\]

則位置 2 的有效 query 可以注意位置 1 的 PAD key。模型目前沒有禁止這件事。

## 最小修法一：限定右側 padding

如果本章不想加入 padding key mask，必須明定：

1. `idx` 只允許連續右側 padding；
2. 第一個 PAD 出現後不得再出現有效 token；
3. target loss mask 仍獨立由 `targets != PAD_ID` 決定；
4. 有效 query 因 causal mask 不會看到位於其未來的右側 PAD。

還應加入資料驗證。例如檢查每列是否存在 `PAD, non-PAD` 的反向轉換；若有便拋錯。這是最小改動，但限制較強。

## 最小修法二：加入 key padding mask

較完整的作法是把 `idx.ne(PAD_ID)` 傳給每個 block 與 attention。允許矩陣可構造成：

\[
M_{b,1,i,j}
=
(j\le i)\land(X_{b,j}\ne PAD).
\]

其 shape 為 \((B,1,T,T)\)，可廣播到 head 軸。必須再處理全遮罩 query 列；例如左側 PAD 的第一個 query 可能沒有任何有效 key。依本卷規則，核心實作應明確拒絕全遮罩列，不能讓 softmax 對全 \(-\infty\) 產生 NaN。

無論選哪種修法，都必須明說：

- causal mask 控制未來資訊；
- key padding mask 控制哪些輸入位置可被注意；
- target loss mask 控制哪些預測位置進入 loss；
- 三者不是同一個 mask。

這些內容也可用來合理補足目前不足的正文篇幅。

---

# 三、reshape 與 transpose 的說明錯誤

## 原句

> 「這裡使用 `reshape` 而非 `transpose` 來改變記憶體布局的邏輯視圖」

這句不正確。實際程式明明同時使用：

```python
.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
```

`view`／`reshape` 把最後一個 \(D\) 軸拆為 \((H,d_h)\)，得到：

\[
(B,T,H,d_h).
\]

`transpose(1,2)` 才交換 time 與 head 軸，得到：

\[
(B,H,T,d_h).
\]

reshape 不會替代 transpose，也不會自行把 \((B,T,H,d_h)\) 變成 \((B,H,T,d_h)\)。此外，`reshape` 不宜被概括成「改變記憶體布局」；它改變 shape，可能回傳 view，也可能複製，取決於 layout。

## 最小修法

逐步改寫為：

1. `view/reshape`：拆分最後的 model feature 軸，\((B,T,D)\to(B,T,H,d_h)\)；
2. `transpose`：交換 T 與 H，\((B,T,H,d_h)\to(B,H,T,d_h)\)；
3. 合頭時先 transpose 回 \((B,T,H,d_h)\)；
4. 因 transpose 後通常非 contiguous，程式使用 `.contiguous().view(B,T,D)`。

這是本章 shape／layout 契約的核心，不只是文字風格問題。

---

# 四、因果證明仍可做一項最小對齊

多層因果性證明的邏輯已成立，但證明中的 block 方程沒有精確反映程式的 pre-norm attention：

程式是：

```python
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
```

證明則直接將注意力視為前層表示的投影，沒有明寫 attention 前的 `norm1`。由於 LayerNorm 只沿 feature 軸作用，這不會破壞因果性，所以不是結論錯誤；但既然本章主張證明完整模型，方程應與程式一致。

**最小修法：**

將注意力輸入記為

\[
\widetilde H_{l-1}^{(i)}
=\mathrm{LN}_1(H_{l-1}^{(i)}),
\]

再由 \(\widetilde H\) 產生 Q/K/V。接著寫：

\[
H_{\mathrm{mid}}^{(i)}
=
H_{l-1}^{(i)}
+
\mathrm{MHA}(\mathrm{LN}_1(H_{l-1}))_i,
\]

以及

\[
H_l^{(i)}
=
H_{\mathrm{mid}}^{(i)}
+
\mathrm{FFN}(\mathrm{LN}_2(H_{\mathrm{mid}}^{(i)})).
\]

如此證明與實際 pre-norm block 完全一致。

---

# 五、測試覆蓋與文字宣稱不一致

## 1. 梯度測試只檢查輸出層

**原句：**

> 「梯度流動至嵌入層與各投影矩陣。」

但程式只檢查：

```python
assert model.out.weight.grad is not None
assert torch.isfinite(model.out.weight.grad).all()
```

這不能證明梯度已流到 embedding、Q/K/V、輸出投影及 FFN。

**最小修法：**

至少對以下代表性參數檢查 `.grad is not None`、shape 相同及全部有限：

- `tok_emb.weight`
- 第一層 `w_q.weight`
- 第一層 `w_k.weight`
- 第一層 `w_v.weight`
- 第一層 `w_o.weight`
- `fc1.weight`
- `fc2.weight`
- `out.weight`

對 embedding 還可核對：

- PAD row 的梯度為零；
- 至少一個實際出現的非 PAD token row 有非零梯度。

不能要求每個元素都非零，因為對稱或數值抵消可能產生合法零值。

## 2. 缺少 token 越界與錯誤 dtype 測試

forward 已加入檢查，這很好，但 `run_tests()` 沒有驗證這些故障路徑。章稿又把程式描述為包含完整參數驗證。

**最小修法：**

加入：

- `idx` 含 \(V\)；
- `idx` 含 \(-1\)；
- `idx` 為 float；
- target 含 \(V\)；
- target dtype 為 float；
- `max_len=0`。

每項都應預期 `ValueError`，而不是依賴底層 embedding 或 CE 的偶然錯誤訊息。

## 3. 未提供實際 mask 方向故障測試

反例段落說：

> 「因果性測試失敗。」

但程式沒有注入上三角 mask 的故障測試。全 PAD 測試可算故障測試，因此形式要求未完全缺失；只是 mask 方向的具體主張沒有由自足程式驗證。

若補測試，不宜依賴隨機初始化保證差異一定大於閾值。應用固定 Q/K/V 的小張量，讓未來 value 明確非零，構造確定反例。

---

# 六、因果性測試的數值表述

測試以：

```python
assert diff < 1e-5
```

作判準。在此模型沒有 dropout，且兩次前向的前綴輸入完全相同；理論上前綴計算圖相同。不過平行矩陣運算的浮點實作可能有環境差異，因此容忍閾值合理。

文字應避免把單一 seed 的通過當成證明。數學證明負責一般性；程式測試只負責抓出此特定實作的常見錯誤。章稿目前兩者都有，可再明確說明分工。

另外：

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

可能把 \(V-1\) 改成 PAD_ID 0。這仍是合法的「改變未來 token」測試，但會混入 PAD 語義。若想讓測試只比較兩個有效 token，可用：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

使結果保持在 \(1,\ldots,V-1\)。

---

# 七、概率與表述細節

## 1. softmax 公式的 shape 標記不嚴謹

**原句：**

> \(A_{ij}=\cdots\in\mathbb R^{B\times H\times T\times T}\)

單一元素 \(A_{ij}\) 是 scalar；整個張量 \(A\) 才屬於該張量空間。

**最小修法：**

寫成：

\[
A\in\mathbb R^{B\times H\times T\times T},
\qquad
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_k\exp(S'_{b,h,i,k})}.
\]

這也能清楚呈現 softmax reduction 是沿最後 key 軸。

## 2. 忘記縮放的後果仍過度絕對

**原句：**

> 「Softmax 飽和，梯度不穩定。」

未縮放不保證每個輸入都飽和。標準解釋依賴 Q/K 分量近似零均值、獨立且方差相近，使內積方差隨 \(d_h\) 增長。

**最小修法：**

改為「在常用初始化的近似假設下，未縮放的 score 方差會隨 \(d_h\) 增長，可能令 softmax 過尖並使部分梯度變小或不穩定」。

## 3. 「未歸一化對數概率」不夠精確

**原句：**

> 「未歸一化對數概率」

logits 一般稱為未正規化分數；經 log-softmax 後才是正規化後的 log probability。最小修法是改成「未正規化分數（logits）」。

---

# 八、資料、訓練與能力範圍

合成資料已正確標明為 IID 隨機 token，且明說不能證明語法或泛化，這是改進。訓練 loop 對固定同一小批資料重複十步，只能作為前向、反向與 optimizer plumbing 示範，不能視為一般訓練成果。稿中沒有宣稱收斂，這點合格。

但：

> `print("Training simulation finished.")`

是程式執行到結尾時的執行期訊息，沒有問題；正文則不可改成「模型已訓練」。目前依賴聲明已明確說沒有執行紀錄，可以保留。

`generate_synthetic_data` 隱含要求 `vocab_size>=2`，因為它呼叫 `torch.randint(1, vocab_size, ...)`。模型建構器卻只要求 `vocab_size>0`。如果直接用 `vocab_size=1` 呼叫生成器，隨機整數範圍為空。

**最小修法：**

在生成器中檢查 `vocab_size >= 2`、`batch_size >= 1`、`seq_len >= 1`。模型本身若允許只有 PAD 的詞表雖數學上可建構，但沒有實用 next-token target；也可直接要求模型 `vocab_size>=2`。

---

# 九、習題解答的嚴謹性

第 1 題使用 \(B=1,T=2,H=2,d_h=2\)，因此 \((B,T,H,d_h)\) 與 \((B,H,T,d_h)\) 恰巧都是 \((1,2,2,2)\)。這會掩蓋 transpose 的作用，與本章希望教導 reshape／transpose 區別的目標相反。

**最小修法：**

把題目改成 \(T=3,H=2\)。如此：

- projection 後為 \((1,3,4)\)；
- reshape 後為 \((1,3,2,2)\)；
- transpose 後為 \((1,2,3,2)\)；
- scores 為 \((1,2,3,3)\)。

這能實際檢查軸順序，而不是只有相同數字的 shape。

第 3 題答案說因果性測試「將失敗」，對一般錯誤 mask 的概念結論正確，但任意隨機權重可能偶然令未來路徑貢獻為零。更嚴謹的答案是：「錯誤 mask 允許未來依賴，故存在輸入與權重使測試失敗；一般隨機初始化下通常也會觀察到非零差異。」

---

# 十、來源與引用

## 原句

> `PyTorch Documentation (General). [N3]`

但 N3 的實際標題是 `scaled_dot_product_attention API`。本章沒有呼叫 SDPA，而是手寫 matmul、mask、softmax。不能把特定 SDPA 頁面標成 PyTorch 一般文件，也不能以其 API mask 語義直接替手寫布林 mask 背書；本章手寫 mask 的 True=允許是作者自己的明確定義。

N4 是 NumPy broadcasting 指南，但本章程式完全使用 PyTorch，且來源註記說 N4 尚未逐條核對。

**最小修法：**

- N1 可作 Transformer 架構來源，但來源註記既然只讀過摘要頁，就不要宣稱已逐條由論文核實實作細節。
- N3 若保留，只列為 SDPA 延伸比較，本章未直接使用。
- N4 若保留，只標「延伸入口，尚未逐條核對」，不可作已查證依據。
- PyTorch `nn.Linear`、`nn.LayerNorm`、`Embedding`、`cross_entropy` 若無精確已核對來源，可不虛構成 N3 所涵蓋內容。

---

# 十一、篇幅硬性要求

自動測量顯示正文約 2572 中文字，低於最低 3000 字。即使程式很長，依本卷規則，公式、程式、英文與參考來源不充作中文字數，因此不能靠增加程式註解或英文 API 名稱補足。

建議以必要內容補足至少約 500 至 800 中文字：

1. 增加 causal mask、key padding mask、target loss mask 的比較；
2. 解釋右側 padding 契約或實作 key mask；
3. 展開 reshape、transpose、contiguous 的差異；
4. 補充 loss 手算，例如兩個有效 target 加一個 PAD，明算分子只含兩項、分母為 2；
5. 說明數學證明與有限數值測試的證據範圍；
6. 補全 embedding、attention 與 FFN 梯度測試的預期。

這些都是實質內容，不是為達字數而填充。

---

# 十二、總評

可保留的部分包括：

- 完整 decoder-only 模型結構；
- pre-norm 殘差順序；
- Q/K/V 及 logits shape；
- causal mask 方向與 softmax 軸；
- 修正後的 PAD NLL 分子與有效 token 分母；
- 多層因果性歸納證明主體；
- 兩個正確手算；
- CPU optimizer 與多步 loop；
- 無下載、無 GPU、無外部資料；
- 未執行證據與 IID 資料能力限制；
- 養殖案例的合成資料與禁止設備控制界線。

阻擋核准的主因是最低篇幅仍未達標，以及 padding 輸入已被模型接受卻沒有 key-mask 或右側 padding 契約。修正這兩點，並同步修正 reshape／transpose 說明、梯度測試與來源標示後，章稿才符合本卷的完整性要求。

VERDICT: REVISE