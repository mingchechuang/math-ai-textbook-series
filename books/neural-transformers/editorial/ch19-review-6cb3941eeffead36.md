## 獨立審稿結論

本輪已達正文最低 3000 中文字，上一輪的兩個主要阻擋點也大致修正：

1. PAD loss 現在正確地由 `ignore_index=PAD_ID` 排除分子，並以相同 target mask 的有效 token 數作分母。
2. 程式明定只接受連續右側 padding，且加入輸入驗證，因此在此受限契約下，有效 query 不會注意到未來的 PAD key。

完整模型、合成資料、loss、optimizer、訓練 loop、正常／邊界／故障測試均已提供；兩個手算正確，四類習題與解答也已具備。然而仍有數項定義與程式契約未完全閉合，其中最重要的是：右側 padding 契約只檢查 `idx`，沒有檢查 `targets` 與 `idx` 的相容關係；多層因果性證明仍沒有忠實寫入 pre-norm attention；另有兩處明確的張量／概率表述錯誤。這些不需重寫章稿，但應修正後再核准。

---

# 一、重算確認

## 1. 右側 padding 驗證邏輯

程式使用：

```python
non_pad = (idx != self.PAD_ID).long()
flipped = torch.flip(non_pad, dims=[1])
cummax = torch.cummax(flipped, dim=1).values
violation = (flipped == 0) & (cummax == 1)
```

對合法序列 `[1, 1, 0, 0]`：

- `non_pad=[1,1,0,0]`
- `flipped=[0,0,1,1]`
- `cummax=[0,0,1,1]`
- 沒有 `flipped==0` 且 `cummax==1` 的位置

所以合法。

對非法內部 PAD `[1,0,1]`：

- `flipped=[1,0,1]`
- `cummax=[1,1,1]`
- 中間位置滿足 `flipped==0` 且 `cummax==1`

所以會拋錯。此檢查本身正確。

## 2. 為何右側 padding 可免 key mask

若一列有效 prefix 長度為 \(m\)，則：

\[
X_0,\ldots,X_{m-1}\ne PAD,\qquad
X_m,\ldots,X_{T-1}=PAD.
\]

任一有效 query 位置 \(i<m\) 透過 causal mask 只能注意 \(j\le i<m\)，因此不會看到右側 PAD。PAD query 位置本身雖可能產生非零 logits，但只要相應 target 是 PAD，便由 loss mask 排除。這個論證成立，但它同時要求 target 與輸入 padding 對齊；目前程式沒有完整驗證後半條件。

## 3. PAD loss

目前實作等價於：

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

分子與分母一致，且全 PAD 時先拒絕，正確。

## 4. 注意力手算

稿中第三列 softmax 約為：

\[
(0.2482,0.5035,0.2482),
\]

故輸出約為：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

計算正確。FFN 的 \([2,5]\) 亦正確。

---

# 二、必須修正：輸入與 target 的 padding 契約沒有閉合

## 原句

> 「本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。」

程式只驗證 `idx`。但模型允許例如：

```python
idx     = [[4, 5, 0, 0]]
targets = [[5, 6, 7, 0]]
```

`idx` 符合右側 padding，`targets` 也含有效 token 7；位置 2 的輸入卻是 PAD，而該位置 loss 有效。這時位置 2 的 query 表示主要來自位置 embedding 與前層運算，程式仍會訓練它預測 7。這不符合所宣稱的 next-token 右側 padding 契約。

對正常的錯位序列，若完整序列是：

\[
[a,b,c,PAD,PAD],
\]

則可能取：

\[
X=[a,b,c,PAD],\qquad
Y=[b,c,PAD,PAD].
\]

因此至少應滿足：

\[
X_{b,t}=PAD\implies Y_{b,t}=PAD.
\]

target 本身也應為右側連續 PAD，而不能出現 `[有效, PAD, 有效]`。

## 最小修法

在 targets 檢查後加入：

1. target 必須右側連續 padding；
2. `idx == PAD` 的位置，target 也必須是 PAD。

例如概念上檢查：

```python
if torch.any((idx == self.PAD_ID) & (targets != self.PAD_ID)):
    raise ValueError("A PAD input position must have a PAD target")
```

並重用右側 padding 驗證函數檢查 targets。最好把重複邏輯抽成章內自足的小函數，而不是複製兩份。

還應加入一個故障測試，明確拒絕：

- `idx=[1,0,2]`
- `targets=[2,0,3]`
- `idx=[1,2,0]`、`targets=[2,3,4]`

否則「安全免 key mask」只是一半契約。

---

# 三、因果證明仍未忠實對應 pre-norm 程式

## 原句

證明寫：

> \(V_j^{(l)}\) 是第 \(l-1\) 層輸出 \(H_{l-1}^{(j)}\) 的線性投影。

但程式實際是：

```python
x = x + self.attn(self.norm1(x))
```

所以 Q/K/V 並不是直接由 \(H_{l-1}\) 投影，而是由：

\[
\widetilde H_{l-1}^{(i)}
=
\operatorname{LN}_1(H_{l-1}^{(i)})
\]

投影。

這不會推翻因果性結論，因為 LayerNorm 只沿 feature 軸作用；但本章標題與命題都聲稱處理「完整模型」，證明應與所實作的 pre-norm block 一致。

## 最小修法

將注意力步驟寫成：

\[
\widetilde H_{l-1}^{(i)}
=
\operatorname{LN}_1(H_{l-1}^{(i)}),
\]

\[
A_l^{(i)}
=
\operatorname{MHA}(\widetilde H_{l-1})_i,
\]

\[
H_{\mathrm{mid}}^{(i)}
=
H_{l-1}^{(i)}+A_l^{(i)}.
\]

再指出 LayerNorm 不混合 time 軸，故 \(\widetilde H_{l-1}^{(i)}\) 與 \(H_{l-1}^{(i)}\) 有相同的時間依賴集合。第二個 sublayer 同理使用 \(\operatorname{LN}_2\)。

目前證明的邏輯骨架可全部保留，只需補這個位置式步驟。

---

# 四、softmax 元素與張量 shape 混寫

## 原句

> \(A_{ij}=\frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1}\exp(S'_{ik})}\in\mathbb R^{B\times H\times T\times T}\)

\(A_{ij}\) 若表示固定 batch、head、query、key 的元素，它是 scalar，不屬於四維張量空間。若省略 batch 與 head 索引，至少也不能把單一元素標為整個張量 shape。

## 最小修法

改為：

\[
A\in\mathbb R^{B\times H\times T\times T},
\]

且

\[
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_{k=0}^{T-1}\exp(S'_{b,h,i,k})}.
\]

如此也精確表示 reduction 沿最後的 key 軸 \(k\) 進行。

---

# 五、logits 定義不精確

## 原句

> 「未歸一化對數概率」

logits 是未正規化分數。一般只有經過 log-softmax 後的量才是 log probability。不存在「每個 logit 本身就是未正規化概率的對數」這個必要定義；線性層輸出可以是任意實數分數。

## 最小修法

改為：

> 「Logits \(Z\in\mathbb R^{B\times T\times V}\) 是每個詞元的未正規化實數分數；對詞彙軸套用 softmax 後得到條件機率。」

---

# 六、`.contiguous()` 的敘述過度且部分錯誤

## 原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

在目前這條使用 `.view(...)` 的程式路徑中，transpose 後通常需要 `.contiguous()`，否則 `view` 會因 stride 不相容而拋出錯誤。PyTorch 的 `view` 不應靜默給出錯誤的記憶體映射；它通常是成功給出符合 stride 的 view，或直接拒絕。

另外，若改用：

```python
out = out.transpose(1, 2).reshape(b, t, self.d_model)
```

則可不顯式呼叫 `.contiguous()`，因為 `reshape` 必要時可複製。

## 最小修法

改成：

> 「本程式後續使用 `view`，因此先呼叫 `.contiguous()`，避免 transpose 後 stride 不相容而拋錯；若改用 `reshape`，框架可在必要時建立副本。」

---

# 七、梯度測試尚可再避免假陽性

本輪梯度測試已大幅改善，確實覆蓋 embedding、Q/K/V、輸出投影、FFN 與輸出層，並檢查 shape 及有限性。

但：

```python
assert torch.all(pad_grad == 0)
```

在此測試中 `x_grad` 是由 `torch.randint(1,V,...)` 產生，本來就完全沒有 PAD。即使 embedding 沒有設定 `padding_idx`，未被索引的 row 梯度也會是零。因此這項測試不能驗證 `padding_idx` 的效果。

## 最小修法

另建一個符合右側 padding 契約的輸入，例如：

```python
x = [[1, 2, 0]]
targets = [[2, 0, 0]]
```

執行 backward 後再檢查 PAD row 梯度為零。這才能證明 PAD 實際出現在 embedding lookup 中而該 row 仍不更新。

同時可對至少一個實際出現的非 PAD token row 檢查存在非零梯度，以避免只證明 gradient tensor 被建立。

---

# 八、測試段落與程式內容不同步

## 原句

測試說明中的梯度項仍只寫：

> 「檢查 `out.weight.grad` 是否存在且有限。」

但程式已檢查多個代表性參數。正文應同步列出實際範圍，否則讀者無法由測試說明得知 Q/K/V、FFN 與 embedding 也被檢查。

此外，右側 padding 驗證已是本輪新增的重要安全契約，卻沒有任何測試：

- 合法 `[1,2,0,0]` 應接受；
- 非法 `[1,0,2,0]` 應拒絕；
- 全 PAD 輸入是否允許也應明定。

全 PAD `idx` 在目前程式中被允許；若沒有 targets，會輸出 logits；若 targets 全 PAD，則由 loss 檢查拒絕。這個策略本身可以成立，但應寫清楚「全 PAD 前向可產生無評估意義的 logits，帶 loss 時拒絕零有效 target」，或乾脆在模型入口拒絕全 PAD 樣本。

---

# 九、因果測試可能把有效 token 改成 PAD

## 原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

當原 token 為 \(V-1\) 時，修改後會變成 0，即 PAD。因為它位於最右側，仍符合右側 padding 契約，所以不會造成程式失敗；但測試同時改變了「token 值」與「padding 身分」，讓測試目的不夠單純。

## 最小修法

保持新 token 在有效範圍：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

對原值 \(1,\ldots,V-1\)，此式會循環到另一個有效 token，不產生 PAD。

---

# 十、手算習題仍掩蓋 transpose 的作用

## 原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

此時 reshape 後：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

因為 \(T=H=2\)，數值 shape 完全相同，讀者即使忘記 transpose 也會得到相同的 shape 字串。這與正文強調 reshape 不等於 transpose 的目標衝突。

## 最小修法

把 \(T\) 改成 3：

- projection：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

這能真正檢驗軸順序。

---

# 十一、反例答案表述過強

## 原句

> 「因果性測試將失敗。」

錯誤的上三角 mask 確實允許未來資訊，因此「存在」權重與輸入使過去 logits 依賴未來。但對任意隨機初始化，未來路徑可能因投影恰為零、權重抵消或容差而沒有被單次測試觀察到。

## 最小修法

改成：

> 「錯誤 mask 破壞了因果性保證；一般非退化權重下測試預期失敗。要構成確定故障測試，應固定一組使未來 value 對輸出有非零貢獻的權重與輸入。」

這也符合「數值測試不代替證明」的全卷原則。

---

# 十二、其他最小修正

1. **縮放後果仍過度絕對**

   原句：

   > 「Softmax 飽和，梯度不穩定。」

   應改為在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 方差隨 \(d_h\) 增長，因而「可能」使 softmax 過尖。

2. **生成器參數未驗證**

   `torch.randint(1, vocab_size, ...)` 要求 `vocab_size>=2`，也應要求 `batch_size>=1`、`seq_len>=1`。目前模型接受 `vocab_size=1`，但生成器會因空取樣範圍失敗。至少在生成器入口明確拋出 `ValueError`。

3. **batch 空軸未拒絕**

   `idx` shape 為 `(0,T)` 時，不帶 targets 的 forward 可能產生空 logits。若章稿不打算支持空 batch，應檢查 `b>=1`。

4. **來源標示**

   N3 僅作 SDPA 延伸比較的說法已比前稿精確；但來源註記已核對 N3 的 True=參與，而正文又寫「可能與本卷約定不同」。較精確的寫法是：給定 N3 所述版本中布林 True 表示參與，但其他 API、不同 mask 類型或版本不可由此類推。本章手寫 mask 不依賴 SDPA。

---

# 十三、可保留內容與最終判定

下列內容已符合要求，可直接保留：

- 正文已超過最低 3000 字；
- 完整 CPU PyTorch decoder-only 模型；
- embedding、位置、多頭注意力、pre-norm、FFN、logits 全部自足；
- optimizer、資料、loss 與十步訓練 loop；
- 正確的 next-token 錯位資料產生；
- 正確的有效 target token 平均；
- causal mask 的 True=允許約定；
- 兩個正確逐步手算；
- 多層因果性歸納證明的主要邏輯；
- 正常、邊界與故障測試；
- 四類習題及答案；
- 無下載、無 GPU、無虛構執行紀錄；
- IID 合成資料不代表泛化的限制；
- 養殖案例不控制設備且不提供真實操作閾值。

目前修改量已不大，但仍涉及 padding 契約、完整模型證明及明確的 shape／概率定義，不能僅視為文風問題。補上 target padding 相容檢查及測試，修正 pre-norm 證明、softmax 元素 shape、logits 定義與 `.contiguous()` 敘述後即可再驗收。

VERDICT: REVISE