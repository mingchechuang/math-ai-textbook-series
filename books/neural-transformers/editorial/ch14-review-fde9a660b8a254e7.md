# 第14章獨立審稿意見

## 一、總評

本輪已完成前次大部分必要修訂。手動重算後，核心定義、shape、Softmax reduction、完整 $Q/K/V$ 反向、非對稱長度、布林 hard mask、全遮罩拒絕與有限差分方向均正確。正文3444中文字，超過最低3000中文字；程式無 `pass`、`TODO` 或未定義的正文依賴；章稿也沒有聲稱程式已執行。

目前僅剩一項實質敘述需要修正：在「反例與常見陷阱」中，章稿仍把未縮放的尺度風險寫成確定的權重二值化、梯度極小及訓練收斂緩慢。這與本章其他段落已採用的條件式、風險式表述不一致，也沒有執行或理論證明支持「訓練收斂緩慢」的無條件結論。另有測試說明與實際程式覆蓋範圍不一致、有限差分目標未在 docstring 明列，以及習題1的一維記號容易和 NumPy `.T` 規則混淆。後三項修補很小，但第一項屬實質證據邊界問題。

因此本輪仍判定修訂；不需改動核心數學或重寫章稿。

---

## 二、核心數學重算

給定：

$$
Q\in\mathbb{R}^{B\times T_q\times d_k},\quad
K\in\mathbb{R}^{B\times T_k\times d_k},\quad
V\in\mathbb{R}^{B\times T_k\times d_v},
$$

交換 $K$ 最後兩軸後：

$$
K^T\in\mathbb{R}^{B\times d_k\times T_k}.
$$

所以：

$$
S=\frac{QK^T}{\sqrt{d_k}}
\in\mathbb{R}^{B\times T_q\times T_k}.
$$

逐 query 沿最後 key 軸做 Softmax：

$$
A_{bij}
=
\frac{\exp S_{bij}}
{\sum_{\ell=1}^{T_k}\exp S_{bi\ell}},
$$

故每個合法 query row 都滿足：

$$
\sum_{j=1}^{T_k}A_{bij}=1.
$$

輸出：

$$
Y=AV
\in\mathbb{R}^{B\times T_q\times d_v}.
$$

反向令 $G_Y=\partial L/\partial Y$，可得：

$$
G_V=A^TG_Y
\in\mathbb{R}^{B\times T_k\times d_v},
$$

$$
G_A=G_YV^T
\in\mathbb{R}^{B\times T_q\times T_k}.
$$

Softmax VJP：

$$
G_S
=
A\odot
\left(
G_A-\sum_{\text{key}}A\odot G_A
\right),
$$

其中 reduction 輸出為 $(B,T_q,1)$，再廣播到 $(B,T_q,T_k)$。最後：

$$
G_Q=\frac{G_SK}{\sqrt{d_k}},
\qquad
G_K=\frac{G_S^TQ}{\sqrt{d_k}}.
$$

章稿的公式與程式均符合上述結果。`dK` 使用：

> `np.swapaxes(dScores, -1, -2)`

正確交換最後兩軸，不會誤用三維 `.T` 將所有軸反序。

例14.2中：

$$
G_S\approx
\begin{bmatrix}
0.161&-0.240&0.079\\
-0.240&0.079&0.161
\end{bmatrix}
$$

每列和約為0，符合 Softmax 對共同平移不變的結構。其 $G_Q$ 與 $G_K$ 數值亦與矩陣乘法一致。

---

## 三、方差命題核對

目前命題條件寫為：

> 所有分量 $\{q_i\}$ 與 $\{k_i\}$ 相互獨立，且各自均值為0、方差為1。

這是足夠條件。由相互獨立：

$$
E[q_ik_i]=E[q_i]E[k_i]=0,
$$

$$
E[q_i^2k_i^2]=E[q_i^2]E[k_i^2]=1.
$$

因此：

$$
\operatorname{Var}(q_ik_i)=1.
$$

不同索引的乘積項也彼此獨立，所以：

$$
\operatorname{Var}\left(\sum_iq_ik_i\right)
=
\sum_i\operatorname{Var}(q_ik_i)
=d_k.
$$

縮放後：

$$
\operatorname{Var}\left(
\frac{q\cdot k}{\sqrt{d_k}}
\right)=1.
$$

小結已正確限定：

> 它並非唯一必要的尺度，也不保證不會飽和。

這已修正先前過度絕對的結論，可保留。

---

## 四、必要修正：仍有無證據的收斂斷言

逐字原句：

> **忘記縮放**：若 $d_k=512$，內積標準差約 22.6。Softmax 輸入差異過大，導致權重極端二值化，梯度極小，訓練收斂緩慢。

原因：

在命題條件下，未縮放單一分數的標準差確實是：

$$
\sqrt{512}\approx22.6.
$$

但這只能說明 logits 的尺度可能較大。它不能單獨推出：

1. 每個 query row 的權重都會極端接近 one-hot；
2. 所有相關梯度都極小；
3. 完整模型訓練必然收斂緩慢。

Softmax 的尖銳程度取決於同一 row 中分數的相對差距，不只取決於單一分數的邊際標準差。完整梯度還取決於上游 $G_A$，而 $G_V=A^TG_Y$ 不經 Softmax Jacobian。模型優化也受到初始化、殘差、正規化、資料與優化器影響。

章稿沒有實際訓練紀錄，也沒有證明「訓練收斂緩慢」。這與本卷不得虛構收斂證據的規則衝突，也與本章前文的謹慎說法：

> 可能使 Softmax 權重過於尖銳

不一致。

最小修法：

只需將該點改成：

> **忘記縮放**：在命題的獨立、零均值、單位方差假設下，若 $d_k=512$，未縮放內積的標準差約為22.6。較大的分數尺度會提高 Softmax 權重過度尖銳及傳向 logits 的部分梯度變弱之風險，可能增加優化困難，但不保證每列都飽和或模型必然收斂緩慢。

這是本輪主要必改項。

---

## 五、遮罩與非有限值核對

目前布林 mask 的契約清楚：

> `True=Allowed`

程式先檢查 mask 是 NumPy 陣列、shape 等於 scores、dtype 為布林，再拒絕全 False row。合法 mask 經：

```python
mask_float = np.where(mask, 0.0, -np.inf)
scores = scores + mask_float
```

於 Softmax 前施加，符合章級約定。

若某位置被遮罩，則其穩定 Softmax 指數為0，故：

$$
A_j=0.
$$

Softmax VJP 中：

$$
(G_S)_j
=
A_j\left((G_A)_j-\sum_\ell A_\ell(G_A)_\ell\right)
=0.
$$

因此禁止位置的 score gradient 為0。全遮罩 row 在 Softmax 前拒絕，不會形成 `-inf - (-inf)` 的 NaN。

`stable_softmax` 的 docstring 已更新，明確區分：

- `NaN`、`+inf`：拒絕；
- 整列全 `-inf`：拒絕；
- 部分 `-inf` 且仍有有限 logit：允許。

這部分已自洽。

---

## 六、廣播與梯度 shape

章稿明確說：

> $Q,K,V$ 的前導批次軸必須完全相同，不支援 NumPy 的自動廣播

程式亦檢查：

```python
if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:
    raise ValueError(...)
```

這是合理的教學接口。若允許 batch broadcasting，反向確實必須把梯度沿被擴展的軸求和回原 shape；目前程式選擇拒絕，因此不需要實作該 reduction。

非對稱案例：

$$
Q:(1,5,2),\quad
K:(1,3,2),\quad
V:(1,3,4)
$$

產生：

$$
A:(1,5,3),\qquad
Y:(1,5,4),
$$

程式斷言與此一致。這能避免 $T_q=T_k$ 時錯誤轉置被方形 shape 掩蓋。

---

## 七、測試可重現性已修正

`run_tests()` 現在在開頭建立：

> `rng = np.random.default_rng(14)`

並將正常、梯度 shape、全遮罩、廣播及非對稱案例都改用同一個局部生成器。這已修復全域 `np.random` 狀態造成的非決定性。

章稿沒有宣稱固定 seed 可跨 NumPy 版本、BLAS 或平台逐位重現，因此沒有過度承諾。

測試2也已由：

> 梯度驗證

改成：

> 梯度 shape

與其實際 assertion 一致。

---

## 八、有限差分目標仍應在文件中明列

逐字原句：

> 有限差分驗證 $Q,K,V$ 的梯度。

程式實際使用：

> `dY = np.ones_like(Y)`

及：

> `return np.sum(y)`

原因：

這是在驗證明確的標量目標：

$$
L(Q,K,V)=\sum_{b,t,r}Y_{btr},
$$

其上游梯度才是全一張量。這個驗證完全合法，但函式 docstring 未說明只核對此標量目標的一個 VJP 方向。

最小修法：

把 docstring 改成：

> 以標量目標 `L = sum(Y)` 進行中央有限差分，因此解析反向的 `dY` 固定為全一張量；返回 $Q,K,V$ 三者的最大絕對誤差。

這不需要改程式。若想支援一般方向，可另接受 `dY` 並使用 `sum(Y*dY)`，但不是本章核准的必要條件。

---

## 九、測試文字摘要仍少列新增案例

「測試與預期結果」目前只列測試1至5，但程式另有非對稱、部分遮罩與完整故障集合。

原因：

程式本身已覆蓋 lab 要求，但正文的證據摘要沒有反映實際範圍。讀者若只閱讀該節，可能誤認為非對稱長度和部分 mask 沒有測試。

最小修法：

增加：

> **測試6：非對稱長度與部分硬遮罩。** 預期權重 shape 為 $(1,5,3)$、輸出為 $(1,5,4)$；禁止位置權重為0，每列權重和為1，三路梯度與輸入同 shape，有限差分低於指定門檻。

> **測試7：故障集合。** 預期低 rank、非 NumPy 輸入、$d_k$ 不符、$K/V$ 長度不符、mask shape/dtype 不符、非有限輸入、非法 scale 和錯誤 `dY` shape 均被拒絕。

這屬最小文件同步，不需增加程式。

---

## 十、程式習題3核對

習題3要求新增固定有限 `attn_bias`。解答目前包含：

1. 修改後的函式簽名；
2. bias 必須為 NumPy 浮點陣列；
3. bias shape 必須等於 scores；
4. bias 必須有限；
5. 同一 bias 傳入初次前向與每次正負擾動；
6. 非方形輸入；
7. 非均勻上游梯度；
8. $Q,K,V$ 逐元素中央差分；
9. 未聲稱已執行；
10. 若 bias 可訓練，$G_{\text{bias}}=G_S$；
11. 若未來允許 broadcasting，須 sum 回原 shape。

數學和程式邏輯正確。

仍建議在 `check_fixed_bias()` 前明寫：

> 此函式須搭配完成題目修改後、已接受 `attn_bias` 關鍵字的前向函式；不能直接搭配正文未修改版本執行。

目前上下文已能推知，但一句明示可避免讀者直接複製後遇到 unexpected keyword argument。此項是說明改善，不單獨構成拒稿理由。

---

## 十一、習題1的一維記號

逐字原句：

> $Q=[1,0],K=[1,1],V=[1]$

本卷已明確提醒 NumPy 一維 `.T` 不改變 shape。這是手算題，數學意圖可理解，但最好避免讓讀者把一維 ndarray 當 row matrix。

最小修法：

改成：

$$
Q=\begin{bmatrix}1&0\end{bmatrix},\quad
K=\begin{bmatrix}1&1\end{bmatrix},\quad
V=\begin{bmatrix}1\end{bmatrix},
$$

並列出 shapes 為 $(1,2),(1,2),(1,1)$。答案保持不變。此項屬精確性改善，不是核心錯誤。

---

## 十二、來源與能力邊界

章稿沒有宣稱程式已實際執行。以下表述符合規則：

> 預期所有 `assert` 通過

以及：

> 本章未聲稱已實際執行。

沒有虛構設備、耗時、實測收斂率或模型訓練成果。養殖案例明確是完全合成示意，沒有真實操作閾值，也沒有生成控制設備的指令。

N3只作 API 語義參考，並明說不是本機執行證據；N4標為尚未逐條核對。N1可作書目定位，但來源註記表明未完整閱讀全文，現稿也沒有捏造已核對全文實驗。

本章不是完整 decoder-only Transformer 章，因此不要求在此提供完整模型和訓練 loop；不能因沒有訓練 loop 拒稿。資料切分、cache、position 與 next-token loss 亦非本章核心範圍。

---

## 十三、核准前最小修訂清單

本稿已無核心梯度、shape、mask 或有限差分演算法錯誤。核准前只需：

1. 將「導致權重極端二值化、訓練收斂緩慢」改成條件式風險表述；
2. 在有限差分 docstring 明列標量目標是 `sum(Y)`；
3. 在測試說明補列非對稱／部分 mask 與故障集合；
4. 最好把習題1改成明確二維矩陣；
5. 最好明說 `check_fixed_bias()` 需搭配修改後的前向版本。

其中第1項是唯一明確的實質證據邊界問題：未經執行或證明，不應把可能的優化困難寫成必然的收斂緩慢。其餘均為很小的一致性修補。

VERDICT: REVISE