## 獨立審稿結論

本稿已符合 3000 中文字下限，完整列出模型、資料、loss、optimizer、訓練 loop 與測試；PAD loss、張量主形狀、因果 mask、兩個手算及多層因果性的主要論證均正確。沒有聲稱已實際執行，也沒有下載、GPU、shell、網路或未定義外部模組。

但本輪稿件仍保留數項上一版已可直接定位的技術問題，並非風格偏好：右側 padding 契約沒有同步約束 targets；pre-norm 證明與實作不完全一致；單一 attention 元素被誤標成四維張量；logits 被錯稱為「未歸一化對數概率」；`view` 的失敗行為描述錯誤；PAD embedding 梯度測試沒有讓 PAD 實際出現在輸入中，因此不能測到所聲稱的性質。這些都有小幅局部修法，但在修正前不宜核准。

---

# 一、重新核算後可確認正確的部分

## 1. 注意力手算

給定

\[
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},\qquad
K=
\begin{bmatrix}
1&0\\
1&1\\
0&1
\end{bmatrix},
\]

則

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}.
\]

除以 \(\sqrt2\) 並施加下三角 mask 後，三列注意力權重約為：

\[
(1,0,0),
\]

\[
(0.3302,0.6698,0),
\]

\[
(0.2482,0.5035,0.2482).
\]

因此第三列輸出約為：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

稿中結果正確。

## 2. FFN 手算

\[
[1,2]W_1=[1,2,1,2],
\]

經 ReLU 後不變，再乘 \(W_2\)：

\[
[1,2,1,2]W_2=[2,5].
\]

數值與 shape 正確。

## 3. PAD loss

目前程式使用：

```python
ignore_index=self.PAD_ID,
reduction='sum'
```

再除以：

```python
valid_count = targets.ne(self.PAD_ID).sum().item()
```

因此實際計算：

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

分子與分母採同一 target mask，只平均一次；全 PAD target 也會在除法前被拒絕。此核心修正正確。

## 4. 右側 padding 檢查

對合法 `[1,1,0,0]`，反轉後為 `[0,0,1,1]`，累積最大值為 `[0,0,1,1]`，不會找到零後仍有一的違規位置。對非法 `[1,0,1]`，反轉後為 `[1,0,1]`，中間零的累積最大值為一，會被拒絕。此演算法正確辨識內部 PAD。

---

# 二、阻擋問題一：`idx` 與 `targets` 的 padding 契約沒有閉合

## 逐字原句

> 「本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。」

以及程式：

```python
if (idx == self.PAD_ID).any():
    ...
```

但 target 檢查只有 dtype、shape 與索引範圍，沒有檢查 target 是否右側 padding，也沒有檢查 PAD 輸入位置是否對應 PAD target。

例如下列輸入目前會被接受：

```python
idx     = [[4, 5, 0, 0]]
targets = [[5, 6, 7, 0]]
```

`idx` 符合右側 padding，但位置 2 的輸入是 PAD，target 卻是有效 token 7，因而該位置會進入 loss。這違反稿中宣稱的 next-token 右側 padding 契約。此時「有效 query 不會看到 PAD key，所以不需 padding key mask」的論證也不足，因為程式把 PAD 輸入位置本身當成有效訓練 query。

正常錯位序列若為：

\[
[a,b,c,PAD,PAD],
\]

則合理切片是：

\[
X=[a,b,c,PAD],\qquad
Y=[b,c,PAD,PAD].
\]

至少應滿足：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

## 最小修法

抽出一個 `check_right_padding(tensor, name)`，同時檢查 `idx` 與 `targets`。此外加入：

```python
if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
    raise ValueError("PAD input positions must have PAD targets")
```

再加入兩個故障測試：

1. `idx=[[1,0,2]]` 應因內部 PAD 被拒絕；
2. `idx=[[1,2,0]]`、`targets=[[2,3,4]]` 應因 PAD input 對應有效 target 被拒絕。

target 比 input 提早一格出現 PAD，例如 `idx=[1,2,3]`、`targets=[2,3,0]`，則是合法的序列結束情況，不應拒絕。

---

# 三、阻擋問題二：因果證明沒有忠實反映 pre-norm

## 逐字原句

> 「\(V_j^{(l)}\) 是第 \(l-1\) 層輸出 \(H_{l-1}^{(j)}\) 的線性投影。」

但實作是：

```python
x = x + self.attn(self.norm1(x))
```

Q/K/V 實際由 \(\operatorname{LN}_1(H_{l-1})\) 投影，而非直接由 \(H_{l-1}\) 投影。這不會使因果性命題變成錯誤，因為 LayerNorm 不混合 time 軸；然而命題標示為完整多層模型的證明，方程應和實際 block 一致。

## 最小修法

在歸納步驟加入：

\[
\widetilde H_{l-1}^{(i)}
=
\operatorname{LN}_1(H_{l-1}^{(i)}).
\]

因 LN 只沿 feature 軸作用，\(\widetilde H_{l-1}^{(i)}\) 與 \(H_{l-1}^{(i)}\) 具有相同的時間依賴範圍。再由 \(\widetilde H\) 產生 Q/K/V，並寫：

\[
H_{\mathrm{mid}}^{(i)}
=
H_{l-1}^{(i)}
+
\operatorname{MHA}(\operatorname{LN}_1(H_{l-1}))_i,
\]

\[
H_l^{(i)}
=
H_{\mathrm{mid}}^{(i)}
+
\operatorname{FFN}(\operatorname{LN}_2(H_{\mathrm{mid}}^{(i)})).
\]

其餘歸納論證可以保留。

---

# 四、attention 元素與整體張量的 shape 混寫

## 逐字原句

> \(A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1}\exp(S'_{ik})}\in\mathbb R^{B\times H\times T\times T}\)

\(A_{ij}\) 是單一元素，若 batch 與 head 也固定，它是 scalar；整體 \(A\) 才具有 \((B,H,T,T)\) shape。目前寫法混淆元素與張量。

## 最小修法

改成：

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

這也明確表達 softmax 沿最後的 key 軸 \(k\) reduction。

---

# 五、logits 的定義不精確

## 逐字原句

> 「表示每個位置對詞彙表中每個詞元的未歸一化對數概率。」

線性輸出 logits 是未正規化實數分數。只有經 log-softmax 後才是 log probability。把 logits 稱為「未歸一化對數概率」容易把模型分數和機率域混為一談。

## 最小修法

改成：

> 「Logits 是每個位置對每個詞元的未正規化實數分數；沿詞彙軸套用 softmax 後得到條件機率，套用 log-softmax 後得到對數機率。」

---

# 六、`contiguous`／`view` 說明有錯

## 逐字原句

> 「`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射」

在目前程式確實先呼叫 `.contiguous()` 才能穩健地使用 `view`。但 PyTorch 的 `view` 不應靜默給出錯誤映射：若 stride 與所求 shape 不相容，通常會拋錯。若改用 `reshape`，則框架可在必要時複製，顯式 `.contiguous()` 並非普遍不可省略。

## 最小修法

改成：

> 「本程式後續使用 `view`，所以先以 `.contiguous()` 取得相容布局；否則 transpose 後的 stride 通常會使 `view` 拋錯。若使用 `reshape`，框架可在必要時建立副本。」

---

# 七、PAD embedding 梯度測試是假陽性

## 逐字原句

> `assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."`

但產生梯度的輸入是：

```python
x_grad = torch.randint(1, V, (2, 5))
```

其取值永遠不含 0，所以 PAD row 根本沒有參與 embedding lookup。即使移除 `padding_idx=0`，未被索引的第 0 row 梯度仍然是零。因此目前測試無法驗證 `padding_idx` 的性質。

## 最小修法

使用真的含 PAD、且符合右側 padding與 target 契約的案例，例如：

```python
x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
```

反向後檢查：

1. PAD row 梯度為零；
2. 至少一個實際出現的非 PAD row 梯度非零；
3. Q/K/V、FFN 與 output 梯度存在、shape 正確且有限。

原本的多參數梯度檢查可以保留。

---

# 八、因果測試會偶然把有效 token 改成 PAD

## 逐字原句

```python
x_mod[:, -1] = (x_mod[:, -1] + 1) % V
```

原 token 若為 \(V-1\)，修改後會成為 PAD_ID 0。由於它在最右側，仍符合 right-padding 檢查，因此程式不一定失敗，但測試同時改變 token 內容與 padding 身分。

## 最小修法

保持在有效詞元範圍：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

這會把 \(1,\ldots,V-1\) 循環到另一個非 PAD token。

此外，有限容差測試不是因果性的數學證明；本章已有歸納證明，因此只需明說數值測試用於發現本實作的 mask 錯誤。

---

# 九、測試說明沒有同步程式

## 逐字原句

> 「執行 `backward()`，檢查 `out.weight.grad` 是否存在且有限。」

實際程式已檢查 embedding、Q/K/V、attention output projection、FFN 與 output 層。文字應同步，否則讀者會誤以為只測最終輸出層。

另一方面，右側 padding 是本章採用的重要資料契約，但 `run_tests()` 沒有測試合法與非法 padding 排列。應補：

- `[1,2,0,0]`：正常；
- `[1,0,2,0]`：拒絕；
- PAD input 對應有效 target：拒絕；
- 全 PAD input 不帶 target 是否允許：明確定義並測試。

目前全 PAD target 被拒絕是正確的，但全 PAD `idx` 單獨前向仍會產生 logits。若保留此行為，應明說這些 logits 不具有效評估意義。

---

# 十、習題沒有真正檢查 transpose

## 逐字原句

> 「給定 \(D=4,H=2\)，輸入 \(X\in\mathbb R^{1\times2\times4}\)」

由於 \(T=H=2\)：

\[
(B,T,H,d_h)=(1,2,2,2),
\]

而 transpose 後：

\[
(B,H,T,d_h)=(1,2,2,2).
\]

兩個 shape 的數字完全相同，忘記 transpose 也看不出來，削弱了本章對軸順序的教學與故障檢查。

## 最小修法

將題目改為 \(T=3,H=2\)：

- projection：\((1,3,4)\)
- reshape：\((1,3,2,2)\)
- transpose：\((1,2,3,2)\)
- scores：\((1,2,3,3)\)

並同步修改答案。

---

# 十一、反例與 scale 敘述過強

## 逐字原句一

> 「因果性測試將失敗。」

錯誤上三角 mask 破壞了因果保證，確實存在輸入與參數令測試失敗；但任意隨機初始化下，未來路徑可能因投影為零、抵消或容差而偶然看不到差異。

**最小修法：**

改成「一般非退化權重下預期失敗；確定性故障測試應固定能使未來 value 產生非零貢獻的權重與輸入」。

## 逐字原句二

> 「Softmax 飽和，梯度不穩定。」

忘記 scale 不保證必然飽和。此結論依賴 Q/K 分量近似獨立、零均值且方差相近。

**最小修法：**

改為「在常用初始化的方差近似下，未縮放 score 的方差會隨 \(d_h\) 增長，可能使 softmax 過尖並令部分梯度變小或不穩定」。

---

# 十二、其他邊界與來源問題

## 1. 合成生成器沒有參數檢查

`torch.randint(1, vocab_size, ...)` 要求 `vocab_size>=2`。目前 `TinyDecoder` 允許 `vocab_size=1`，但生成器會遇到空取樣範圍。生成器也沒有拒絕 `batch_size<=0` 或 `seq_len<=0`。

**最小修法：**

在生成器入口檢查：

```python
if batch_size < 1 or seq_len < 1 or vocab_size < 2:
    raise ValueError(...)
```

## 2. 空 batch

forward 沒有拒絕 \(B=0\)。若不打算支援空 batch，應檢查 `b >= 1`，並加入故障測試。

## 3. N3 的說明不夠精確

來源註記已說明 N3 所列版本中布林 True 表示參與注意，但正文又說：

> 「框架的布林遮罩語義可能與本卷……不同」

更精確的說法應是：N3 所述特定 SDPA API／版本中 True 表示參與；其他 API、mask 類型或版本不能直接類推。本章手寫 mask 不依賴 SDPA。

N4 是未逐條核對且本章未使用 NumPy的延伸入口，保留亦可，但不應讓它看起來支撐本章 PyTorch broadcasting 的具體行為。

---

# 十三、總評

本章現在已具備可接受稿件的大部分必要元素：

- 正文篇幅達下限；
- 模型及訓練 loop 完整；
- shape 主鏈正確；
- causal mask 正確；
- PAD loss 正確；
- 兩個手算正確；
- 多層因果性證明主體成立；
- 測試包含正常、邊界與故障類型；
- 四類習題與解答齊全；
- 無虛構執行、收斂率或設備效能；
- 合成資料與養殖安全範圍已明示。

剩餘問題都可局部修復，不需要重寫整章；但其中 target padding 契約、pre-norm 證明、概率定義與測試假陽性屬實質正確性問題，不能以風格偏好略過。修正後再審可望核准。

VERDICT: REVISE