## 獨立審稿結論

本輪有一項實質修正：`perplexity` 已不再使用硬編碼 700，而改用 `np.log(np.finfo(np.float64).max)`。因此上一輪「把 $e^{701}$ 錯報為無限」的拒稿理由已失效，不能再沿用。

重新核對後，主要手算、nats／bits、one-hot CE、有效 token reduction、序列似然、shape 與主程式基本正確。然而仍有一個確定的程式邊界錯誤：批次 KL 習題答案會在已知 $P>0,Q=0$ 的位置實際除以零。另有三項與本章核心契約直接相關的定義／證明缺口：CE 的無限值域沒有明列；可數空間的 KL 證明沒有說明擴展 Jensen；Fisher 局部展開沒有可定位來源或完整條件。測試也沒有覆蓋本輪修正的 overflow 邊界及批次 KL 的除零故障。

因此本輪已接近批准，但仍需局部修訂。我沒有執行程式，以下均是手算及逐行審查。

---

# 一、先行重算

## 1. 例題 4.1

三筆正確類別機率是 $0.9,0.9,0.6$：

$$
-\ln0.9\approx0.1053605,
\qquad
-\ln0.6\approx0.5108256.
$$

所以：

$$
\text{Total NLL}
=2(0.1053605)+0.5108256
\approx0.7215466,
$$

$$
\text{Mean NLL}
=0.7215466/3
\approx0.2405155\text{ nats}.
$$

轉成 bits：

$$
0.2405155/\ln2\approx0.34699\text{ bits}.
$$

稿中 $0.72155$、$0.24052$、$0.3470$ 正確。

## 2. KL 手算

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.1\log_2(0.5)
 +0.2\log_2(2/3)
 +0.7\log_2(1.4)\\
&\approx-0.1-0.116993+0.339799\\
&\approx0.122806\text{ bits}.
\end{aligned}
$$

稿中 $0.12281$ bits 正確。

## 3. PAD 困惑度

有效 token NLL 為 $2.0,0.5,0.5$：

$$
\overline L=\frac{2.0+0.5+0.5}{3}=1,
$$

$$
PP=e^1\approx2.71828.
$$

錯誤的 batch perplexity 平均為：

$$
\frac{e^2+e^{0.5}}2
\approx4.51889.
$$

稿中數值正確。

## 4. 整合題

$$
\text{Total NLL}=1.0+0.5+2.0=3.5,
$$

$$
\text{Mean NLL}=3.5/3\approx1.16667\text{ nats},
$$

$$
\text{bits/token}=1.16667/\ln2\approx1.68315,
$$

$$
PP=e^{1.16667}\approx3.21127.
$$

答案正確。

---

# 二、本輪已修正且通過的 Perplexity 閾值

**原句：**

```python
if mean_nll > np.log(np.finfo(np.float64).max):
    return np.inf
```

這次已按實際 `float64` 上限決定 overflow 邊界，不再以任意常數 700 提前返回 `inf`。由於 `_as_float_array` 明確把 NLL 轉成 `float64`，這個 dtype 上限和實際計算 dtype 一致。

因此：

- $e^{701}$ 應保持有限；
- 超過 $\ln(\text{float64 max})$ 時回傳 `inf`；
- 先前的有限／無限分類錯誤已修正。

這部分應判定通過。

仍建議增加兩個預期測試，避免未來回歸：

```python
assert np.isfinite(
    perplexity(np.array([701.0]), np.array([True]))
)
```

以及用略高於 `np.log(np.finfo(np.float64).max)` 的值確認回傳 `inf`。這是測試覆蓋要求，不是說目前公式仍錯。

---

# 三、阻斷問題：批次 KL 仍然執行除以零

**原句：**

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
terms = np.zeros_like(p, dtype=np.float64)
terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
```

## 原因

`bad` 已經識別 $P>0,Q=0$ 的支撐集違規位置，但計算 `terms` 時仍然使用全部 `p_support`。題目附的第二列恰有：

$$
P=[0.5,0.5,0],
\qquad
Q=[0.5,0,0.5].
$$

第二類別會實際計算：

$$
0.5/0,
$$

再取 $\log(\infty)$。最後：

```python
out[bad] = np.inf
```

雖然令最終數值符合數學定義，但中間仍會產生 divide-by-zero。這與本章反例段明示的原則衝突：

> `必須使用掩蔽（masking），只計算 $p>0$ 的項`

對 KL 而言，只檢查 $p>0$ 還不夠；安全的有限運算位置應是 $p>0$ 且 $q>0$。支撐集違規列再另外設成 `inf`。

## 最小修法

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
safe = p_support & (q > 0)

terms = np.zeros_like(p, dtype=np.float64)
terms[safe] = p[safe] * (
    np.log(p[safe]) - np.log(q[safe])
)

out = np.sum(terms, axis=-1)
out[bad] = np.inf
return out
```

這樣：

- $P=0,Q=0$ 不參與計算；
- $P=0,Q>0$ 不參與計算；
- $P>0,Q>0$ 計算有限 KL 項；
- $P>0,Q=0$ 由 `bad` 明確設為 `inf`；
- 一筆違規不污染其他 batch；
- 不需要先產生除零 warning。

一維 `kl_divergence` 已在除法前直接返回 `np.inf`，所以一維版本正確；問題只存在於程式習題答案。

---

# 四、KL 證明的聲稱範圍仍需對齊

**原句：**

> `本章定義限於有限或可數離散樣本空間 $\mathcal X$。`

以及：

> `命題 4.1：對於任意兩個概率分布 $P,Q$……`

隨後直接把 Jensen 寫成離散和式。

## 核對結果

在有限樣本空間中，現有證明完整正確。令：

$$
S=\{x:P(x)>0\},
\qquad
Z(x)=\frac{Q(x)}{P(x)}.
$$

則：

$$
\mathbb E_P[Z]
=\sum_{x\in S}Q(x)
\leq1.
$$

由 $-\ln$ 的凸性：

$$
D_{\mathrm{KL}}(P\parallel Q)
=\mathbb E_P[-\ln Z]
\geq-\ln\mathbb E_P[Z]
\geq0.
$$

等號需要：

1. $Z$ 在 $P$ 支撐上為常數；
2. $\sum_{x\in S}Q(x)=1$。

故 $Q=P$。

可數無限情形的結論也成立，因此命題不是假的；但左側期望可能是 $+\infty$，需要說明使用擴展期望版本的 Jensen，或用有限截斷再取極限。稿件目前沒有交代這一步。

## 最小修法

最簡單的修法是把命題改成：

> 對有限離散樣本空間上的概率分布 $P,Q$……

並在證明後說：

> 可數無限情形可由擴展 Jensen 或有限截斷極限處理，本章不展開。

另一方式是保留可數域，補充 $\mathbb E_P[Z]\leq1$ 有限以及 Jensen 允許左側為 $+\infty$ 的說明。

習題 2 也應加上「假設 $\mathcal X$ 有限」或「假設相關熵有限」，否則：

$$
D_{\mathrm{KL}}=H(P,Q)-H(P)
$$

可能形式上涉及 $\infty-\infty$，不能直接按普通實數移項。

---

# 五、交叉熵與零機率的域仍不完整

## 1. 沒有明列 CE 的無限條件

**原句：**

> `若 $P(x)>0$ 且 $Q(x)=0$，則 $D_{KL}(P\|Q)=+\infty$。`

同一條件下：

$$
-P(x)\ln Q(x)=+\infty,
$$

因此：

$$
H(P,Q)=+\infty.
$$

本章核心明列「交叉熵與熵的域」，不能只明列 KL。

## 最小修法

將規則改成：

> 若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則該交叉熵項為 $+\infty$，所以 $H(P,Q)=+\infty$，且 $D_{\mathrm{KL}}(P\parallel Q)=+\infty$。

## 2. 最終指標的值域表述過寬

**原句：**

> `採用擴展實數系 $\mathbb R\cup\{+\infty,-\infty\}$。`

合法離散概率下，熵、CE、KL 與 NLL 的最終值位於：

$$
[0,+\infty].
$$

$-\infty$ 只會作為 $\ln0$ 的中間值，不是合法 KL 或 NLL 的最終值。

## 最小修法

改成：

> 本章的熵、CE、KL 與 NLL 允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln0=-\infty$。

## 3. 零權重規則重複且容易誤讀

**原句：**

> `$0\cdot\ln0$ 定義為 $0$。`
>
> `$P(x)\ln(0)$ 定義為 $0$（若 $P(x)=0$）。`

可合併為：

> 零權重項約定為 $0\ln a=0$，包括 $a=0$；若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$。

---

# 六、困惑度的直覺仍然過強

**原句：**

> `若 $PP=100$，表示模型在平均意義上的預測困難度相當於在 100 個候選中隨機猜測。`

一般情況下：

$$
PP
=
\exp\left(
-\frac1N\sum_t\ln p_t
\right)
=
\left(\prod_tp_t\right)^{-1/N},
$$

即正確 token 機率幾何平均的倒數。它不表示每一步的完整預測分布都是在 100 個候選間均勻猜測。

前一句雖已說明均勻分布才有精確的候選數解釋，後一句又容易讓讀者把特殊直覺一般化。

## 最小修法

改成：

> 若 $PP=100$，則正確 token 機率的幾何平均為 $1/100$；只有每一步都均勻分布在相同數量候選上的特殊情形，才等同於在 100 個等可能候選間猜測。

---

# 七、程式介面與測試

## 1. `safe_log_prob` 的介面契約

**原句：**

```python
def safe_log_prob(p):
    if np.any(p < 0):
```

在 `cross_entropy` 內，`p` 已經由 `_as_float_array` 和 `validate_probability` 處理，所以主流程安全。但若讀者直接呼叫：

```python
safe_log_prob([0.5, 0.5])
```

函數沒有自行轉成陣列，也沒有拒絕 NaN。若它只是一個內部 helper，應用底線命名並明確說明輸入已驗證；若它是公開函數，則應自行轉換及檢查。

## 最小修法

二選一：

```python
def _safe_log_prob(p):
```

或在函數開頭：

```python
p = _as_float_array(p, "p")
if not np.all(np.isfinite(p)):
    raise ValueError(...)
```

## 2. 測試覆蓋仍不足以捕捉本輪邊界

目前測試有正常、邊界、故障三類，這一點合格。但還缺：

1. `mean_nll=701` 為有限值；
2. 超過 `float64` 指數上限時為 `inf`；
3. 空有效 mask 拒絕；
4. NaN 概率拒絕；
5. soft label；
6. Python list index labels；
7. 批次 KL 支撐集違規不先除零；
8. label batch 長度錯誤。

至少應補前四項，因為它們直接對應本章核心的非有限值和 token 計數契約。

**原句：**

> `執行時應輸出 "All expected tests passed."`

這只是預期輸出，沒有宣稱已實際運行。稿件沒有虛構測試成功、設備、耗時或訓練結果。

---

# 八、AI、資訊幾何與養殖案例

## 1. ECE 的限制

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱數量、邊界、樣本量和聚合方式影響，不應被表述成充分校準證據。

## 最小修法

> 可搭配可靠度圖、ECE 等檢查校準，但 ECE 受分箱策略影響，不能單獨證明模型已校準。

## 2. Fisher 公式

**原句：**

> `$$D_{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^T I(θ)\delta$$`

有兩個問題：

1. `I(θ)` 應依 LaTeX 慣例改成 `I(\theta)`；
2. 此式需要可微、固定支撐、可交換微分與期望等正則條件，稿中沒有證明，也沒有可定位來源。

## 最小修法

標成「未證明的延伸結果」，列出基本條件及來源；若無已核對來源，則只保留局部直覺，不展示像已建立的定理公式。

## 3. 養殖風險措辭

**原句：**

> `這是一個高風險信號，觸發人工審查。`

高 NLL 可能來自模型失配、罕見正常事件、資料偏移或感測異常，不等於實際現場風險。

## 最小修法

改成：

> 這是一個高 NLL 的待查信號，可提交人工審查。

稿件已說明門檻由驗證集制定、不能查看測試集後調整，且案例為合成、非實際安全閾值；這些限制正確。

---

# 九、來源與能力聲明

**原句：**

> `本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。`

是否執行程式與是否查閱文獻是兩件事。題目提供的來源查閱紀錄沒有 Cover、Goodfellow、Jurafsky–Martin 的逐章核對資訊，也沒有 Fisher 局部展開的定位來源。

## 最小修法

改成：

> 以下為延伸參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。

若作者確實查閱過，應列出版本、章節或頁碼，而不是用「未執行外部程式」代替文獻查證聲明。

---

# 十、資料洩漏、梯度與 Transformer 範圍

本章沒有實際資料建窗或訓練，因此沒有可直接檢查的 train／validation／test 程式。養殖案例已正確要求門檻只用驗證集制定，不得事後查看測試集調整。

本章不是完整 Transformer 章，所以不需要加入模型、訓練 loop、KV cache、position 或 attention mask。logits 版穩定 CE 和梯度也已排在下一章，不應因本章沒有推導 logits 梯度而拒稿。

目前沒有虛構已執行、已通過、已訓練、收斂速度、設備能力或實測 benchmark。

---

# 最終判定

本輪已確實修正 `perplexity` 的硬編碼 700 問題，該舊疑慮不再成立。主要數學手算、單位、shape、mask reduction、one-hot CE 與章稿形式均已達標。

但批次 KL 解答仍在已知 $P>0,Q=0$ 時先執行除以零，與本章自己的安全 mask 原則直接衝突；CE 的無限值域、可數空間證明範圍與未證明的 Fisher 結果也尚未完整交代。以上均可局部修正，不需重寫全章，但修正前仍不宜批准。

VERDICT: REVISE