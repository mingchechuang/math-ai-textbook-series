## 獨立審稿結論

本輪提供的章稿與上一輪實質相同，先前指出的核心問題仍未修改。重新核算後，模型主體、注意力手算、FFN 手算、有效 token loss、因果 mask 與訓練 loop 均大致正確；正文亦已達 3000 中文字下限。然而，右側 padding 契約仍只驗證 `idx` 而不驗證 targets，因而允許 PAD 輸入位置承擔有效 loss；多層因果性證明仍省略實際 pre-norm attention；數學定義仍把 attention 元素誤標成四維張量；logits 與 `view` 的說明仍不精確；PAD embedding 梯度測試仍是假陽性。這些均是可由局部修改解決的技術問題，但目前不能核准。

---

# 一、重新計算結果

## 1. 單頭注意力手算

稿中

\[
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},\quad
K=
\begin{bmatrix}
1&0\\
1&1\\
0&1
\end{bmatrix}
\]

給出

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix},
\]

此處正確。施加因果 mask 後，各列 softmax 約為：

\[
(1,0,0),
\]

\[
(0.3302,0.6698,0),
\]

\[
(0.2482,0.5035,0.2482).
\]

因此

\[
O_2
=
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=
(0.4964,0.7517),
\]

與稿中一致。

## 2. FFN 手算

\[
[1,2]W_1=[1,2,1,2],
\]

ReLU 不改變結果，而

\[
[1,2,1,2]W_2=[2,5].
\]

數值與 shape 均正確。

## 3. loss reduction

目前程式使用：

```python
ignore_index=self.PAD_ID,
reduction='sum'
```

再除以：

```python
valid_count = targets.ne(self.PAD_ID).sum().item()
```

故其數學結果是

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf 1[Y_{b,t}\ne PAD]
}.
\]

PAD 同時從分子與分母排除，且只平均一次。全 PAD target 在除法前被拒絕。此部分正確。

## 4. shape 與 broadcast

主路徑為：

- `idx`: \((B,T)\)
- token embedding：\((B,T,D)\)
- position embedding：\((1,T,D)\)
- batch 軸廣播後：\((B,T,D)\)
- reshape：\((B,T,H,d_h)\)
- transpose：\((B,H,T,d_h)\)
- scores：\((B,H,T,T)\)
- 二維 mask \((T,T)\) 向 batch 與 head 軸廣播
- 合頭：\((B,T,D)\)
- logits：\((B,T,V)\)

程式中的實際 shape 正確。

---

# 二、主要阻擋問題：padding 資料契約仍不完整

## 逐字原句

> 「本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。」

程式實際只檢查：

```python
if (idx == self.PAD_ID).any():
    ...
```

targets 只檢查 shape、dtype 與索引範圍，未檢查右側 padding，也未檢查它與 `idx` 的關係。

下列輸入目前會被接受：

```python
idx     = torch.tensor([[4, 5, 0, 0]])
targets = torch.tensor([[5, 6, 7, 0]])
```

`idx` 是合法右側 padding，但位置 2 的輸入是 PAD，target 卻是有效 token 7，因此該位置會進入 loss。這違反 next-token 右側 padding 契約。該 query 的表示包含位置 embedding、殘差及其他逐位置運算，不能被解讀為正常的有效輸入位置。

若完整序列為

\[
[a,b,c,PAD,PAD],
\]

合理錯位應是

\[
X=[a,b,c,PAD],\qquad
Y=[b,c,PAD,PAD].
\]

所以至少必須滿足：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

target 也不能出現 `[有效,PAD,有效]` 這種內部 padding。

## 最小修法

抽出一個右側 padding 檢查函數，分別套用於 `idx` 與 `targets`。再加入：

```python
if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
    raise ValueError("PAD input positions must have PAD targets")
```

注意反方向不成立：最後一個有效輸入可以預測 PAD，所以 `target==PAD` 不要求 `idx==PAD`。

還需加入故障測試：

1. `idx=[[1,0,2]]`：拒絕內部 PAD；
2. `targets=[[2,0,3]]`：拒絕 target 內部 PAD；
3. `idx=[[1,2,0]]`、`targets=[[2,3,4]]`：拒絕 PAD input 對應有效 target；
4. `idx=[[1,2,3]]`、`targets=[[2,3,0]]`：應接受。

在完成這些條件前，「右側 padding 使 key mask 不再需要」只閉合了 input key 的一半契約。

---

# 三、多層因果性證明仍與 pre-norm 程式不一致

## 逐字原句

> 「\(V_j^{(l)}\) 是第 \(l-1\) 層輸出 \(H_{l-1}^{(j)}\) 的線性投影。」

但程式是：

```python
x = x + self.attn(self.norm1(x))
```

Q、K、V 實際由

\[
\widetilde H_{l-1}^{(i)}
=
\operatorname{LN}_1(H_{l-1}^{(i)})
\]

投影，而不是直接由 \(H_{l-1}^{(i)}\) 投影。

這不推翻命題，因為 LayerNorm 沿 feature 軸運算、不混合時間位置；但既然本節聲稱證明完整實作，就應把 pre-norm 步驟寫進證明，而不是證明一個略有不同的 block。

## 最小修法

在歸納步驟補上：

\[
\widetilde H_{l-1}^{(i)}
=
\operatorname{LN}_1(H_{l-1}^{(i)}).
\]

由於 LN 不混合 time 軸，\(\widetilde H_{l-1}^{(i)}\) 具有與 \(H_{l-1}^{(i)}\) 相同的時間依賴集合。接著寫：

\[
A_l^{(i)}
=
\operatorname{MHA}(\widetilde H_{l-1})_i,
\]

\[
H_{\mathrm{mid}}^{(i)}
=
H_{l-1}^{(i)}+A_l^{(i)},
\]

\[
H_l^{(i)}
=
H_{\mathrm{mid}}^{(i)}
+
\operatorname{FFN}
\left(
\operatorname{LN}_2(H_{\mathrm{mid}}^{(i)})
\right).
\]

其餘歸納推理可保留。

---

# 四、attention 元素與張量 shape 仍混淆

## 逐字原句

> \(A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1}\exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T}\)

單一 \(A_{ij}\) 是 scalar；整體 \(A\) 才具有四維 shape。由於本章特別強調 shape 契約，這種元素／張量混寫應修正。

## 最小修法

改為：

\[
A\in\mathbb R^{B\times H\times T\times T},
\]

以及

\[
A_{b,h,i,j}
=
\frac{\exp(S'_{b,h,i,j})}
{\sum_{k=0}^{T-1}\exp(S'_{b,h,i,k})}.
\]

此式亦明確表達 softmax reduction 沿最後的 key 軸 \(k\)。

---

# 五、logits 的概率定義仍不精確

## 逐字原句

> 「表示每個位置對詞彙表中每個詞元的未歸一化對數概率。」

logits 是未正規化的實數分數。經 softmax 後才是機率，經 log-softmax 後才是對數機率。將 logits 稱作「未歸一化對數概率」容易混淆 score 與 log probability。

## 最小修法

改成：

> 「Logits 是每個位置對各詞元的未正規化實數分數；沿詞彙軸套用 softmax 後得到條件機率，套用 log-softmax 後得到對數機率。」

---

# 六、`contiguous` 與 `view` 的敘述仍錯

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

就目前程式而言，後面使用 `view`，所以先 `.contiguous()` 是合理的。但 PyTorch 的 `view` 不應靜默給出錯誤映射；當 stride 與 shape 不相容時通常會拋錯。若改用 `reshape`，框架可在必要時建立副本，因此 `.contiguous()` 也不是所有寫法中都不可省略。

## 最小修法

改成：

> 「本程式後續使用 `view`，因此先呼叫 `.contiguous()`；否則 transpose 後的 stride 通常會令 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本。」

---

# 七、PAD embedding 梯度測試仍無法證明所宣稱性質

## 逐字原句

> `assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."`

產生該梯度的輸入是：

```python
x_grad = torch.randint(1, V, (2, 5))
```

其值永遠位於 \(1,\ldots,V-1\)，PAD_ID 0 從未被 lookup。任何未被索引的 embedding row 梯度本來就是零，即使刪掉 `padding_idx=0`，這項測試也可能通過。因此它是假陽性，沒有驗證 `padding_idx`。

## 最小修法

用真正含 PAD 的合法右側 padding 資料：

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

反向後檢查：

- PAD row 梯度為零；
- token 1 或 2 的 row 至少一個具有非零梯度；
- 代表性 Q/K/V、FFN、輸出投影與最終輸出層梯度存在、shape 正確且有限。

目前對代表性參數的 `.grad is not None`、shape 及 finite 檢查可以保留。

---

# 八、因果測試不應把有效 token 改成 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

若原值為 \(V-1\)，新值會是 0，也就是 PAD。因為位置在最右側，仍符合右側 padding 檢查，但測試同時改變 token 內容與 padding 身分，不夠單純。

## 最小修法

保持新值為非 PAD：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

對原值 \(1,\ldots,V-1\)，結果一定是另一個有效 token。

此外，`diff < 1e-5` 是特定實作的故障測試，不是因果性的普遍證明。稿中已有歸納證明，可補一句說明兩者證據範圍不同。

---

# 九、測試說明與實際程式未同步

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

實際程式已檢查 embedding、Q/K/V、attention output projection、FFN 與輸出層。測試說明應同步列出這些範圍，並說明：

- `.grad is not None`
- gradient shape 與參數一致
- gradient 全部有限
- 不要求每個元素都非零

此外，新增的右側 padding 契約沒有相應測試。至少需測：

- 合法 `[1,2,0,0]`
- 非法 `[1,0,2,0]`
- 全 PAD input 的明確策略
- input PAD 與 target 有效的錯配

目前全 PAD target 會被拒絕，但全 PAD `idx` 在不傳 targets 時會得到 logits。若保留此行為，應明說這些 logits 不具有有效 token 的評估意義；否則可直接拒絕每列全 PAD。

---

# 十、習題仍無法辨識 reshape 與 transpose

## 逐字原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

因 \(T=H=2\)，reshape 後：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

數字完全一樣，即使學生漏掉 transpose，也會寫出相同 shape。這與章內強調 reshape 不等於 transpose 的目標相衝突。

## 最小修法

將 \(T\) 改為 3：

- 投影後：\((1,3,4)\)
- reshape 後：\((1,3,2,2)\)
- transpose 後：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改答案。

---

# 十一、反例與縮放結論過強

## 逐字原句一

> 「因果性測試將失敗。」

上三角 mask 確實破壞因果保證，但隨機初始化可能因投影為零、抵消或容差而在某次測試中沒有觀察到差異。數學上正確的結論是「存在未來依賴路徑」，不是任意權重下單次測試必然失敗。

**最小修法：**

改為：

> 「一般非退化權重下測試預期失敗；若要構成確定故障測試，應固定一組令未來 value 對過去輸出具有非零貢獻的權重與輸入。」

## 逐字原句二

> 「Softmax 飽和，梯度不穩定。」

沒有除以 \(\sqrt{d_k}\) 不保證必然飽和。方差論證依賴 Q/K 分量近似獨立、零均值且方差相近。

**最小修法：**

改成：

> 「在常用初始化的近似假設下，未縮放 score 的方差會隨 \(d_k\) 增長，可能使 softmax 過尖並使部分梯度變小或不穩定。」

---

# 十二、生成器與空 batch 的邊界未定義

## 逐字原句

```python
seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
```

這要求 `vocab_size >= 2`。目前模型允許 `vocab_size=1`，生成器也沒有檢查 `batch_size>=1` 或 `seq_len>=1`。

此外，forward 沒有拒絕 \(B=0\)。不帶 target 時，空 batch 可能一路產生空 logits；這是否允許沒有定義。

## 最小修法

生成器入口加入：

```python
if batch_size < 1 or seq_len < 1 or vocab_size < 2:
    raise ValueError(...)
```

模型 forward 若不支援空 batch，加入 `if b < 1: raise ValueError(...)`，並加入故障測試。

---

# 十三、來源範圍仍需精確化

## 逐字原句

> 「框架的布林遮罩語義可能與本卷『True=允許注意』的約定不同」

給定的 source notes 已明說 N3 所核對版本的 SDPA 布林 True 表示參與注意。因此對 N3 的特定 API／版本，不應再模糊說「可能不同」。可說其他框架 API、不同 mask 參數或不同版本不能直接類推。本章是手寫 `masked_fill(~mask, -inf)`，其 True=允許語義由程式本身清楚定義，並不依賴 SDPA。

N4 是未逐條核對且本章沒有使用 NumPy的延伸入口，保留時只能標作延伸閱讀，不能作 PyTorch broadcasting 的已核實來源。目前已寫「不可作為已查證依據」，這點可保留。

---

# 十四、可保留部分

下列內容無需重寫：

- 正文已達最低篇幅；
- 模型、資料、loss、optimizer 與 loop 完整；
- 沒有外部下載或 GPU；
- 注意力與 FFN 手算正確；
- Q/K/V、scores、合頭及 logits 主 shape 正確；
- causal mask 在 softmax 前施加；
- softmax 沿 key 軸；
- PAD loss 分子與分母一致；
- 全 PAD target 明確拒絕；
- 因果性歸納證明的主體推理正確；
- 測試涵蓋正常、邊界與故障類別；
- 四類習題與答案存在；
- IID 合成資料不被宣稱具有泛化能力；
- 未虛構已執行、已收斂或效能數據；
- 養殖案例明確禁止實際設備控制。

本稿距離可核准已不遠，但送審版本仍未修掉明確指出的資料契約、定義與測試問題。尤其 target padding 契約與 PAD 梯度假陽性直接影響「程式驗證了什麼」；pre-norm 證明與元素 shape 則影響數學自洽性。完成上述局部修正後方可核准。

VERDICT: REVISE