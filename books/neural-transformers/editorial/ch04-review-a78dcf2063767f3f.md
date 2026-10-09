## 獨立審稿結論

本次提交的關鍵程式與前一輪仍相同，未修正 `perplexity` 的 700 閾值，也未修正批次 KL 在支撐集違規位置的除零運算。因此仍有可確定重現的數值語義錯誤，不能批准。

先自我推翻兩項不應繼續阻擋的疑慮：

1. KL 非負性在可數空間下確實可用擴展期望版本的 Jensen 證明；不能說命題本身錯。問題只在稿件是否要補充該版本的適用說明，或把已展示的初等證明限於有限空間。
2. 有限 NLL 求和可能 overflow 的擔憂，在所有 NLL 非負且能實際配置的 NumPy 陣列中，通常已對應極大的平均 NLL，其 perplexity 本來就會超出 `float64`。這不是本輪的主要拒稿理由。真正確定的錯誤是把 $700<\overline L\leq\ln(\text{float64 max})$ 的仍可表示結果提前改成 `inf`。

我沒有執行程式；以下為逐式重算與逐行分析。

---

# 一、手算與定義核對

## 1. 例題 4.1

正確類別機率為 $0.9,0.9,0.6$：

$$
-\ln0.9\approx0.1053605,
\qquad
-\ln0.6\approx0.5108256.
$$

因此

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

換算為 bits：

$$
0.2405155/\ln2
\approx0.34699\text{ bits}.
$$

稿中 $0.72155$、$0.24052$、$0.3470$ 正確。

## 2. KL 習題

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.1\log_2(0.5)+0.2\log_2(2/3)+0.7\log_2(1.4)\\
&\approx-0.1-0.116993+0.339799\\
&\approx0.122806.
\end{aligned}
$$

稿中 $0.12281$ bits 正確。

## 3. 困惑度例題

有效 token NLL 為 $2.0,0.5,0.5$：

$$
\overline L=\frac{3}{3}=1,
\qquad PP=e\approx2.71828.
$$

錯誤的 batch PP 平均為

$$
\frac{e^2+e^{0.5}}2\approx4.51889.
$$

稿中答案正確，且足以反駁「先算每批 PP 再做算術平均」。

## 4. 整合習題

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

稿中答案正確。

---

# 二、已通過且不應再以舊理由拒稿的部分

## 1. 似然條件

**原句：**

> `若 $x_1,\dots,x_N$ 在給定參數 $\theta$ 下為獨立同分布（i.i.d.）……則聯合似然函數為……`

乘積似然所需的 i.i.d. 條件已列出。

## 2. 自回歸鏈式分解

**原句：**

> `此分解來自機率鏈式法則，不要求各 token 無條件獨立，也不額外假設固定階數的馬可夫性`

此敘述正確。一般 decoder-only 模型可條件於完整可見前綴，不必是一階馬可夫模型。

## 3. 序列 NLL 的符號

**原句：**

> `$$\text{Total NLL}(\theta)=-\sum_{t=1}^T\ln p_\theta(x_t|x_{<t})$$`

負號正確。

## 4. 總 NLL 與平均 NLL

**原句：**

> `在有效集合預先固定且 $N_{\text{valid}}>0$ 時……具有相同的最優參數`

條件完整。稿件也正確指出分母若依賴 $\theta$，等價性可能失效。

## 5. shape 與 reduction

目前：

- `cross_entropy`：`y_pred` 是 `(N,C)`，回傳 `(N,)`；
- `kl_divergence`：輸入 `(C,)`，回傳純量；
- 批次 KL：輸入 `(N,C)`，沿最後類別軸求和，回傳 `(N,)`；
- `perplexity`：只選取 `valid_mask=True` 的 token，總 NLL 除以有效 token 數一次。

沒有發現跨 batch 錯誤聚合、錯誤 broadcast 或平均兩次。

## 6. 章稿契約

正文 3134 字已超過最低 3000 字；已有三個手算、一個證明、自足 NumPy 程式、正常／邊界／故障測試，以及手算／程式／反例／整合習題與完整答案。本章不是完整 Transformer 章，不需要加入 Transformer 模型、cache 或訓練 loop。

---

# 三、阻斷批准的確定錯誤：Perplexity 閾值

**原句：**

```python
if mean_nll > 700: # ln(float_max) approx 709
    return np.inf
```

## 原因

`float64` 最大有限值的自然對數約為 $709.78$。所以當

$$
700<\text{mean NLL}\leq709.78
$$

時，`np.exp(mean_nll)` 仍可能是有限的 `float64`。例如 $e^{701}$ 雖然極大，仍小於 `float64` 最大有限值。現函數卻回傳 `np.inf`。

這不是無關緊要的近似，而是把「有限」錯報為「無限」。本章又明確宣稱嚴格區分數學上的無限與浮點表示範圍，因此這個錯誤直接違反章內契約。

## 最小修法

```python
limit = np.log(np.finfo(np.float64).max)
if mean_nll > limit:
    return np.inf
return float(np.exp(mean_nll))
```

或：

```python
with np.errstate(over="ignore"):
    return float(np.exp(mean_nll))
```

後者應說明：回傳 `inf` 代表結果超出 `float64` 可表示範圍，不一定代表數學上的 perplexity 是無限。

## 必要邊界測試

目前測試只覆蓋 $e^1$，抓不到此錯誤。應增加：

```python
pp_701 = perplexity(
    np.array([701.0]),
    np.array([True])
)
assert np.isfinite(pp_701)
```

再以超過 `np.log(np.finfo(np.float64).max)` 的值驗證回傳 `inf`。不得宣稱這些測試已通過，只能列為執行時的預期。

---

# 四、批次 KL 解答仍執行除以零

**原句：**

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
terms = np.zeros_like(p, dtype=np.float64)
terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
```

## 原因

`bad` 已經識別 $P>0,Q=0$，但計算 `terms` 時仍以全部 `p_support` 為 mask。測試的第二列包含：

$$
P_2=0.5,\qquad Q_2=0,
$$

所以程式仍會計算 $0.5/0$。之後把整列設成 `np.inf` 雖得到正確最終值，但中間會產生 divide-by-zero。這和正文推薦「先 mask，避免非法中間運算」的原則矛盾。

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

一維 `kl_divergence` 已在除法前檢查支撐集並返回 `inf`，因此一維版正確；問題只存在於程式習題的批次答案。

---

# 五、KL 證明的範圍須明示

**原句：**

> `本章定義限於有限或可數離散樣本空間 $\mathcal X$。`

以及：

> `命題 4.1：對於任意兩個概率分布 $P,Q$……`

現有證明直接對和式使用 Jensen。

## 核對結果

在有限 $\mathcal X$ 上，證明完整正確。在可數無限空間上，結論仍成立；令

$$
Z(x)=\frac{Q(x)}{P(x)}
$$

於 $P$ 的支撐集上，則

$$
\mathbb E_P[Z]=\sum_{P(x)>0}Q(x)\leq1.
$$

廣義 Jensen 可給出

$$
\mathbb E_P[-\ln Z]\geq-\ln\mathbb E_P[Z]\geq0,
$$

且左側允許是 $+\infty$。因此不能說原命題錯誤。

但稿中沒有說明使用廣義 Jensen、擴展期望或相應可積性處理；對初學章節而言，公式看起來只是有限加權和版本。

## 最小修法

二選一：

1. 把命題明確限於有限離散樣本空間，並說可數情形需廣義 Jensen 或截斷極限；
2. 保留可數情形，補一句上述 $Z$ 的期望有限且使用擴展期望版 Jensen。

習題 2 同樣應假設 $\mathcal X$ 有限，或假設相關熵有限，避免直接對 $\infty-\infty$ 作代數移項。

這項是證明完整性的修訂要求；真正直接造成錯誤輸出的阻斷項仍是上一節的 perplexity。

---

# 六、CE 與擴展值域仍應收斂

## 1. CE 的無限條件未明列

**原句：**

> `若 $P(x)>0$ 且 $Q(x)=0$，則 $D_{KL}(P\|Q)=+\infty$。`

此時也有：

$$
H(P,Q)=+\infty.
$$

**最小修法：**

改成：

> 若存在 $x$ 使 $P(x)>0,Q(x)=0$，則 $H(P,Q)=+\infty$，且 $D_{\mathrm{KL}}(P\parallel Q)=+\infty$。

## 2. 最終指標不取 $-\infty$

**原句：**

> `採用擴展實數系 $\mathbb R\cup\{+\infty,-\infty\}$`

合法離散概率下，熵、CE、KL 與 NLL 的最終值在 $[0,+\infty]$。$-\infty$ 只是 $\ln0$ 的中間值。

**最小修法：**

> 本章指標允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln0=-\infty$。

## 3. 零權重規則重複

**原句：**

> `$0\cdot\ln0$ 定義為 $0$。`
>
> `$P(x)\ln(0)$ 定義為 $0$（若 $P(x)=0$）。`

可合併為：

> 零權重項約定為 $0\ln a=0$，包括 $a=0$；若權重為正而概率為零，則相應損失項為 $+\infty$。

---

# 七、困惑度直覺仍過強

**原句：**

> `若 $PP=100$，表示模型在平均意義上的預測困難度相當於在 100 個候選中隨機猜測。`

一般情況下：

$$
PP
=
\left(
\prod_{t\in\mathrm{valid}}
p(x_t\mid x_{<t})
\right)^{-1/N_{\mathrm{valid}}},
$$

也就是正確 token 機率幾何平均的倒數。它不代表每一步真的在 100 個候選間均勻抽樣。

**最小修法：**

> $PP=100$ 表示正確 token 機率的幾何平均為 $1/100$；只有均勻候選的特殊情形，才等同於在 100 個等可能候選間猜測。

---

# 八、程式介面與測試

## 1. `safe_log_prob` 只適合作為內部函數

**原句：**

```python
def safe_log_prob(p):
    if np.any(p < 0):
```

`cross_entropy` 內部傳入的 `p` 已驗證且為浮點 NumPy 陣列，因此主流程正常。但直接呼叫 `safe_log_prob([0.5,0.5])` 時，Python list 未經 `_as_float_array`。

**最小修法：**

要麼函數內先轉換並檢查 finite，要麼改名 `_safe_log_prob`，明示只接收已驗證陣列。

## 2. 測試覆蓋

現有測試已包含正常、邊界與故障類，不能說沒有測試。但建議新增：

- Python list labels；
- soft labels；
- NaN 概率；
- 空有效 mask；
- label batch 長度錯誤；
- 701 nats perplexity；
- 批次 KL 支撐集違規的安全 mask。

**原句：**

> `執行時應輸出 "All expected tests passed."`

這是預期，不是作者宣稱已執行。稿件沒有虛構測試成功。

---

# 九、校準、養殖與來源

## 1. ECE

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱方式與樣本量影響。

**最小修法：**

改為「可搭配可靠度圖、ECE 等檢查，但 ECE 不能單獨證明模型已校準」。

## 2. Fisher 公式

**原句：**

> `$$D_{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^T I(θ)\delta$$`

應把 `I(θ)` 改為 `I(\theta)`。這是未證明進階結果，還需列出正則條件或可定位來源；否則標為延伸直覺。

## 3. 養殖案例

**原句：**

> `這是一個高風險信號，觸發人工審查。`

高 NLL 不等於實際現場風險。

**最小修法：**

改成「高 NLL 的待查信號，可提交人工審查」。稿件已說明合成示例、非實際安全閾值，也沒有控制外部設備，這部分邊界基本合格。

## 4. 來源聲明

**原句：**

> `本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。`

是否執行程式與是否查閱文獻無關。提供的來源紀錄沒有 Cover、Goodfellow、Jurafsky–Martin 的逐章核對資訊。

**最小修法：**

改成：

> 以下為延伸參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。

---

# 最終判定

本章數學主線、手算、nats／bits、one-hot CE、token 加權 perplexity、PAD mask、shape 與大部分故障策略均正確，篇幅及章稿形式也已合格。當前不需要重寫，只需局部修正。

但 `mean_nll > 700` 會把仍可表示的有限 perplexity 錯報為 `inf`；批次 KL 又在已知支撐集違規時實際執行除以零。前者是確定的輸出錯誤，後者違反本章自己的安全邊界契約。在這兩項修正前仍不能批准。

VERDICT: REVISE