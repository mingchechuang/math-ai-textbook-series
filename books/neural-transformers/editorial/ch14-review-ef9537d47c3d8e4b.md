# 獨立審稿意見：第14章〈縮放內積注意力與QKV〉

## 一、總結判定

本輪章稿的核心數學已基本正確。重新手算後，前向 $QK^T/\sqrt{d_k}$、逐 query Softmax、$AV$、Softmax VJP，以及 $dQ,dK,dV$ 的矩陣方向都一致；例14.2的數值亦在所示四捨五入精度內成立。程式也已採取明確策略：前導軸不允許 broadcasting、布林 `True=Allowed`、全遮罩行拒絕、輸入須有限、scale 須為有限正數。這些修復有效。

但當前稿仍不能核准，理由有三類：

1. 自動量測正文僅1810中文字，明確低於每章最低3000中文字。
2. 習題3仍把有限的 `-1e9` 模糊地當成 hard mask；習題4仍錯誤宣稱飽和後「僅在最大分數對應的Key上有梯度」。
3. 自足測試仍未涵蓋章級 lab 指定的非對稱長度、部分遮罩與 masked gradient，也未固定 seed；若干已實作的故障分支完全沒有測試。

以下只列仍需修正之處，不因純風格偏好拒稿。

---

## 二、數學與 shape 核對

### 1. 核心前向 shape 正確

逐字原句：

> $Q \in \mathbb{R}^{B \times T_q \times d_k}$，$K \in \mathbb{R}^{B \times T_k \times d_k}$，$V \in \mathbb{R}^{B \times T_k \times d_v}$。

以及：

> $QK^T$ 的形状為 $(B, T_q, T_k)$。

此處成立，但三維張量的 $K^T$ 並不是一般意義下將所有軸反序，而是逐 batch 交換最後兩軸：

$$
(B,T_k,d_k)\longrightarrow(B,d_k,T_k).
$$

最小修法：

在定義後增加一句：「此處 $K^T$ 表示只交換最後兩軸；NumPy 對應 `np.swapaxes(K, -1, -2)`。」同理，$A^T$、$V^T$、$(dS)^T$ 都應採此約定。這可防止讀者將 NumPy 三維陣列的 `.T` 直接套用而得到軸反序。

### 2. 方差命題成立，但條件可再精確

逐字原句：

> 假設 $q,k\in\mathbb{R}^{d_k}$ 的分量 $q_i,k_j$ 是獨立同分布的隨機變數

要使用

$$
\operatorname{Var}\left(\sum_iq_ik_i\right)
=
\sum_i\operatorname{Var}(q_ik_i),
$$

需要不同乘積 $q_ik_i$ 彼此不相關；全部分量互相獨立是足夠條件。現有「$q_i,k_j$ 是獨立同分布」大致可表達此意，但最好明確寫成「所有 $\{q_i\}$ 與 $\{k_i\}$ 分量相互獨立」，以免被理解為只要求同一索引的 $q_i$ 與 $k_i$ 獨立。

命題結論：

> 降低僅因維度增加而導致 Softmax 過度飽和的風險。

此版本已合理，沒有把方差近似誤寫成絕對保證。

但小結又寫：

> 縮放因子 $\sqrt{d_k}$ 的必要性來自於避免 Softmax 梯度飽和

「必要性」和「避免」都過強。方差命題沒有證明此縮放是唯一必要選擇，也沒有保證 Softmax 不飽和。

最小修法：

改成「在所列獨立、零均值、單位方差近似下，除以 $\sqrt{d_k}$ 可抵消內積標準差隨維度增加的尺度，降低過度尖銳的風險」。

### 3. Softmax VJP 正確

逐字原句：

> $$  
> \frac{\partial L}{\partial S} = A \odot \left( G_A - \text{sum}(A \odot G_A, \text{axis}=T_k, \text{keepdims=True}) \right)  
> $$

此公式正確。Reduction 沿 key 軸，結果 shape 為 $(B,T_q,1)$，再廣播回 $(B,T_q,T_k)$。

例題所說：

> 每列和應為0。

也正確，但最好說明原因。由 Softmax 對共同平移不變，

$$
\operatorname{softmax}(s+c\mathbf1)=\operatorname{softmax}(s),
$$

所以 $dS$ 必須與全1方向正交：

$$
\sum_jdS_{ij}=0.
$$

最小修法是補上這一句結構性說明，而不是只以近似小數和為0作驗證。

### 4. $dQ,dK,dV$ 均正確

目前公式：

$$
dV=A^TdY,\qquad
dA=dYV^T,
$$

$$
dQ=\frac1{\sqrt{d_k}}dSK,\qquad
dK=\frac1{\sqrt{d_k}}dS^TQ
$$

均正確，且各梯度與輸入同 shape。手算 $dK$ 為 $(3,2)$，已修正早期轉置錯誤。這部分無須再改。

---

## 三、程式接口與數值策略

### 1. 缺少輸入 rank 驗證

逐字原句：

> `if Q.shape[-1] != K.shape[-1]:`

隨後程式直接使用：

> `K.shape[-2]`  
> `V.shape[-2]`

若傳入一維陣列，例如 `Q=np.array([1.0, 2.0])`，部分 `shape[-2]` 存取會拋出 `IndexError`，而不是接口可預期的 `ValueError`。若傳入 Python list，則連 `.shape` 都不存在。

最小修法：

先明定接口只接受 NumPy 陣列，再檢查：

```python
if not all(isinstance(x, np.ndarray) for x in (Q, K, V)):
    raise TypeError("Q, K, V must be NumPy arrays")
if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
    raise ValueError("Q, K, V must have at least two dimensions")
```

若本章只教單頭 batch 版本，也可更嚴格要求三維；但目前文件使用 `...`，允許一致的多前導軸亦可。

### 2. Softmax docstring 與實作不一致

逐字原句：

> 若輸入含 NaN 或全列為 -inf，會導致未定義行為或錯誤，需在上游檢查。

實作已在 `stable_softmax` 內檢查 `NaN`、`+inf` 及全 `-inf`，不再只是「需在上游檢查」。

最小修法：

將說明改成：

> 輸入含 `NaN`、`+inf`，或某個 reduction 行全為 `-inf` 時，函式明確拋出 `ValueError`；部分元素為 `-inf`、但該行至少有一個有限值時允許計算。

此外，既然先減有限最大值，合法列至少有一個 `exp(0)=1`，則：

> `if np.any(sum_exp == 0):`

主要只是防禦性檢查，而不是正常的全遮罩偵測。可保留，但註解不應再把它當成唯一全遮罩處理。

### 3. 嚴格拒絕前導軸廣播是有效策略

逐字原句：

> $Q,K,V$ 的前導批次軸必須完全相同，不支援 NumPy 的自動廣播

以及：

> `if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:`

此設計自洽。它避免前向發生 batch broadcasting 後，反向還需將梯度 `sum_to_shape`。既然已明確拒絕，就不要求本章另外實作廣播梯度。

### 4. `finite_diff_check` 的標量目標一致，但過於特殊

逐字原句：

> `dY = np.ones_like(Y)`

以及：

> `return np.sum(y)`

兩者一致，對應標量損失 $L=\sum Y$。中央差分也正確。

不過，若 value 向量產生某些對稱或退化情形，$\sum Y$ 對注意力權重的依賴可能很弱甚至消失。固定隨機資料通常不會恰好退化，但更一般的驗證應選定非均勻上游梯度 $G$，令

$$
L=\sum Y\odot G,
$$

解析反向使用 `dY=G`。

最小修法：

令 `finite_diff_check` 接受 `dY`，或在內部建立固定非均勻的 `dY`，並把 `loss` 改成：

```python
return np.sum(y * dY)
```

這可同時驗證不同輸出方向，而不是只驗證全1方向。

### 5. 有限差分只回傳最大絕對誤差

逐字原句：

> `return np.max(np.abs(num_g - bwd_grad))`

對固定小型 `float64` 案例可用，但未考慮梯度尺度。若梯度很大，絕對誤差稍大未必代表相對錯誤；若梯度很小，單看相對誤差也可能不穩。

最小修法：

同時回傳絕對與縮放相對誤差：

```python
abs_err = np.max(np.abs(num_g - bwd_grad))
denom = max(1.0, np.max(np.abs(num_g)), np.max(np.abs(bwd_grad)))
rel_err = abs_err / denom
```

固定案例可對兩者設定預期門檻。無執行紀錄時仍只能說「預期低於門檻」。

---

## 四、測試覆蓋不足

### 1. 隨機測試沒有固定 seed

逐字原句：

> `Q = np.random.randn(2, 3, 4).astype(np.float64)`

`run_tests()` 每次使用不同資料。雖然測的是代數公式，但固定 seed 能讓失敗案例可重現，也能使 $10^{-5}$ 的門檻對應到明確資料。

最小修法：

使用局部生成器：

```python
rng = np.random.default_rng(0)
Q = rng.standard_normal((2, 3, 4))
```

不要使用全域 `np.random.seed` 污染其他程式狀態。章文應明說固定 seed 只控制本次資料生成，不保證跨版本或跨平台逐位相同。

### 2. 沒有可執行的非對稱長度測試

本章 lab 明確要求「對稱／非對稱長度」。但 `run_tests()` 的正常案例是：

> `Q (2,3,4), K (2,3,4), V (2,3,4)`

即 $T_q=T_k$。這種方形案例可能掩蓋錯軸或轉置錯誤。

最小修法：

加入例如：

```python
Q = rng.standard_normal((1, 5, 2))
K = rng.standard_normal((1, 3, 2))
V = rng.standard_normal((1, 3, 4))
Y, cache = scaled_dot_product_attention(Q, K, V)
assert cache["attn_weights"].shape == (1, 5, 3)
assert Y.shape == (1, 5, 4)
```

並對該非方形案例檢查：

```python
assert dQ.shape == Q.shape
assert dK.shape == K.shape
assert dV.shape == V.shape
```

最好也在此案例上做有限差分，這比只測方形資料更能抓出 $dK$ 轉置問題。

### 3. 沒有合法部分遮罩測試

目前只有：

> `mask_full = np.zeros((1, 1, 2), dtype=bool)`

這只驗證全遮罩拒絕，沒有驗證正常的混合 True/False 路徑。至少需檢查：

- 被遮罩位置的注意力權重為0；
- 每個 query 的允許權重和為1；
- 部分遮罩下前向 shape 正確；
- 部分遮罩下 $Q,K,V$ 反向與有限差分一致。

最小修法：

建立例如：

```python
mask = np.array([[
    [True, False, True],
    [False, True, True]
]], dtype=bool)
```

並斷言：

```python
A = cache["attn_weights"]
assert np.all(A[~mask] == 0.0)
assert np.allclose(np.sum(A, axis=-1), 1.0)
```

由於布林 mask 是常數，不對 mask 求梯度；但應對同一 mask 下的 $Q,K,V$ 做有限差分。

### 4. 多個故障分支沒有測試

實作已宣告會拒絕下列輸入，但 `run_tests()` 未涵蓋：

- $Q,K$ 的 $d_k$ 不同；
- $K,V$ 的 $T_k$ 不同；
- mask shape 錯誤；
- mask dtype 非布林；
- $Q,K,V$ 含 `NaN` 或無限值；
- `scale_factor` 為0、負數或 `NaN`；
- `dY` shape 錯誤；
- 輸入 rank 不足。

不必為每行建立龐大框架，但至少應各覆蓋 shape、mask dtype、非有限值及 scale 四種故障類別。

### 5. 沒有實際呼叫測試入口

章稿定義 `run_tests()`，但程式末尾沒有呼叫。函式本身仍可算自足，但若希望讀者直接執行檔案，最小修法是加：

```python
if __name__ == "__main__":
    run_tests()
```

章稿使用：

> 預期所有 `assert` 通過

這是合規措辭，沒有虛構執行；不應改寫成「已通過」。

---

## 五、習題與解答仍有關鍵錯誤

### 1. 習題3混淆 hard mask 與有限 logit bias

逐字原句：

> 支持 `mask` 為浮點數矩陣（例如 0 或 -1e9）

解答：

> 假設 mask 浮點數中，負值極小表示遮罩

原因：

`-1e9` 是有限數，不是數學上的禁止。它只是很大的負偏置；在某些 dtype 或 logit 尺度下可能近似使權重下溢到0，但不能視為精確 hard mask。現有解答也沒有定義「負值極小」的門檻，沒有拒絕全遮罩行，更沒有提供題目要求的梯度驗證。

最小修法二選一：

- 將題目改為 `attn_bias`，允許任意有限浮點偏置，並明說它不是 hard mask；布林 mask 仍負責禁止位置。
- 若確實要浮點 hard mask，只允許元素為 `0.0` 或 `-np.inf`，並檢查每列至少有一個有限位置。

解答還應加入固定小案例的 $Q,K,V$ 有限差分，而不是只寫「梯度驗證邏輯不變」。

### 2. 習題4的梯度結論錯誤

逐字原句：

> 導致梯度在大部分方向上為零，僅在最大分數對應的 Key 上有梯度。

Softmax 導數為：

$$
\frac{\partial A_i}{\partial S_j}=A_i(\delta_{ij}-A_j).
$$

當 $A_m\to1$ 時，最大位置的自身導數也是

$$
A_m(1-A_m)\to0,
$$

並非「只有最大位置有梯度」。非最大位置與交叉導數也趨近0。另一方面，$dV=A^TdY$ 可能主要流向最大權重對應的 value，但這不能和 $dS,dQ,dK$ 混為一談。

最小修法：

改為：

> 未縮放時 Softmax 往往更尖銳，其 Jacobian 的許多分量可能變小，使傳向 logits、$Q$ 與 $K$ 的部分梯度方向變弱；具體梯度仍取決於上游 $dA$。對 $V$ 的梯度則可能主要集中在高權重位置。

### 3. 缺少卷級要求的反例習題

本卷要求手算、程式、反例、整合四類習題。現有四題是手算、理論、程式、整合，沒有明確反例。

最小修法：

增加或替換一題：

> 令所有 $V_j=v$ 完全相同。證明不論注意力權重 $A$ 如何變化，只要每列和為1，輸出均為 $v$。說明為何注意力權重改變不必然造成輸出改變。

完整解答為：

$$
Y_i=\sum_jA_{ij}V_j
=\sum_jA_{ij}v
=v\sum_jA_{ij}
=v.
$$

這也是反駁「權重本身等於最終貢獻」的直接反例。

---

## 六、應用與能力邊界

### 1. 長距依賴敘述過度

逐字原句：

> 注意力機制解決了序列模型中長距離依賴的問題。

注意力提供遠距位置的直接計算路徑，但不保證模型一定學會任意長距關係。

最小修法：

改為「注意力提供直接連結遠距位置的機制，可緩解部分序列模型中資訊需逐步傳遞的問題」。

### 2. 注意力權重仍被當成預測貢獻

逐字原句：

> 僅表示在當前模型表示下該 Token 對當前預測的貢獻較大。

即使某個 $A_{ij}$ 較大，最終影響仍取決於 $V_j$、輸出投影、殘差與後續層。若所有 $V_j$ 相同，權重可大幅改變而輸出完全不變。

最小修法：

改為：

> 只表示該 query 對該 key/value 位置分配較大的混合權重；不能單由權重判定最終貢獻、語義重要性或因果關係。

### 3. 幾何表述不精確

逐字原句：

> $QK^T$ 測量的是向量夾角的餘弦相似性（若向量已歸一化）或投影長度。

未歸一化內積為 $\|q\|\|k\|\cos\theta$，同時受範數與夾角影響；它本身不是一般定義的投影長度。

最小修法：

改為「若兩向量均單位化，內積等於餘弦相似度；未歸一化時，內積同時反映範數與夾角」。

---

## 七、來源、虛構執行與篇幅

章稿沒有聲稱程式已實際執行，也沒有虛構訓練時間、設備、收斂率或實測誤差。使用「預期所有assert通過」符合無執行紀錄的規定。N3亦明確標為 API 語義參考而非本機執行證據；N4標為尚未逐條核對，沒有冒充已驗證來源。

但來源註記說 N1只取得摘要頁、未完整閱讀論文，而參考來源寫：

> 縮放因子的原始來源

這可作書目定位，但不宜暗示本章已逐條核對全文。最小修法是標成「縮放內積注意力的主要原始文獻；本章僅核對可得摘要資訊」，或依實際來源核對狀態調整。

最後，自動量測仍為1810中文字，低於最低3000中文字。公式、程式、英文及參考來源不計字數，所以即使程式區塊很長，也不能補足正文要求。可用下列必要內容補足，而非重複敘述：

1. 以 Frobenius 內積逐步推導 $dQ,dK,dV$；
2. 證明 Softmax VJP 每列和為0；
3. 說明 hard mask 下 masked 權重與 `dScores` 為0；
4. 說明本實作為何拒絕 broadcasting；
5. 補上非對稱長度與部分遮罩的測試解釋；
6. 增加反例習題及完整解答；
7. 精確區分布林 hard mask、`-inf` mask 與有限 additive bias。

核心注意力數學已可保留，但習題錯誤、測試缺口及硬性篇幅不足尚未解除。

VERDICT: REVISE