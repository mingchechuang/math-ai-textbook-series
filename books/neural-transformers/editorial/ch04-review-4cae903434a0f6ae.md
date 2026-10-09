## 獨立審稿結論

本輪已修正上一版最重要的程式問題：批次 KL 現在使用

```python
safe = p_support & (q > 0)
```

只在 $P>0,Q>0$ 的位置計算對數比，不再於支撐集違規位置先執行除以零。上一輪對批次 KL 的拒稿理由已失效，不能繼續沿用。

重新核對後，主要數學、shape、mask、reduction、零機率程式行為、困惑度及習題答案均已達標。剩餘問題已集中在章稿的定義域與未證明進階結果：

1. CE 為 $+\infty$ 的支撐集條件仍未明列，與本章核心「交叉熵的域」要求不完全相符；
2. KL 證明同時聲稱有限與可數空間，但沒有說明可數情形使用擴展期望版 Jensen；
3. Fisher 局部展開屬未證明進階結果，既未列具體條件，也沒有可定位且已核對的來源；
4. 參考來源註記把「是否執行外部程式」和「是否實際查閱來源」混為一談。

其中前三項是明確章稿契約，不只是文字風格。故仍建議一次局部修訂後再批准。我沒有執行程式，以下為手算與靜態審查。

---

# 一、重新計算與已通過部分

## 1. 分類 NLL

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

轉為 bits：

$$
0.2405155/\ln2\approx0.34699\text{ bits}.
$$

稿中數值正確。

## 2. KL 習題

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.1\log_2(0.5)
 +0.2\log_2(2/3)
 +0.7\log_2(1.4)\\
&\approx0.122806\text{ bits}.
\end{aligned}
$$

稿中 $0.12281$ bits 正確。

## 3. 困惑度

有效 NLL 為 $2.0,0.5,0.5$：

$$
\overline L=1,
\qquad
PP=e\approx2.71828.
$$

錯誤 batch PP 平均：

$$
\frac{e^2+e^{0.5}}2\approx4.51889.
$$

例題正確。

## 4. 整合題

$$
\text{Total NLL}=3.5,
\quad
\text{Mean NLL}=3.5/3\approx1.16667,
$$

$$
\text{bits/token}\approx1.68315,
\qquad
PP\approx3.21127.
$$

答案正確。

---

# 二、本輪批次 KL 修正通過

**原句：**

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
terms = np.zeros_like(p, dtype=np.float64)
safe = p_support & (q > 0)
terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
out = np.sum(terms, axis=-1)
out[bad] = np.inf
```

這次邏輯正確：

- $P=0,Q=0$：不參與計算；
- $P=0,Q>0$：不參與計算；
- $P>0,Q>0$：計算有限 KL 項；
- $P>0,Q=0$：該列由 `bad` 設成 `inf`；
- reduction 沿最後類別軸；
- 回傳 shape 為 `(N,)`；
- 一筆違規不污染其他 batch；
- 不先執行除以零。

因此批次 KL 的 shape、mask 與 reduction 現在通過。

---

# 三、主程式核對

## 1. `cross_entropy`

輸入：

- index label：`(N,)`；
- one-hot／soft label：`(N,C)`；
- 預測概率：`(N,C)`。

輸出 `(N,)`。類別軸沿 `axis=-1` 求和，batch 軸保留。

零機率處理：

```python
mask = y_onehot > 0
ce_terms[mask] = y_onehot[mask] * log_q[mask]
```

避免 $0\cdot(-\infty)$。若 $P>0,Q=0$，該樣本被設為 `inf`，且只影響違規樣本。這部分正確。

## 2. 一維 KL

先檢查：

```python
if np.any(p_support & (q_model == 0)):
    return np.inf
```

再計算正支撐項，因此不會先除以零。正確。

## 3. Perplexity

輸入 NLL 與 mask 都要求一維且 shape 相同；mask 必須是布林 dtype；空有效集合拒絕；有效 NLL 要求非負；有效位置出現 `+inf` 時回傳 `inf`，NaN 或 `-inf` 則拒絕。

overflow 邊界：

```python
if mean_nll > np.log(np.finfo(np.float64).max):
    return np.inf
```

與輸入轉成 `float64` 的契約一致。先前硬編碼 700 的問題已修正。

## 4. `safe_log_prob`

仍有一個小型介面問題：

**原句：**

```python
def safe_log_prob(p):
    if np.any(p < 0):
```

它在 `cross_entropy` 中只接收已轉換、已驗證的陣列，因此主流程沒有錯。但若讀者直接呼叫 `safe_log_prob([0.5,0.5])`，list 沒有經 `_as_float_array`；若傳 NaN，也沒有自行拒絕。

**最小修法：**

若它是內部函數，改名 `_safe_log_prob` 並在 docstring 明說只接受已驗證 `float64` 陣列；若是公開函數，就在入口呼叫 `_as_float_array` 並檢查 finite。此項單獨不足以拒稿，但宜一併修正。

---

# 四、交叉熵的定義域仍未完整陳述

**原句：**

> `若 $P(x)>0$ 且 $Q(x)=0$，則 $D_{KL}(P\|Q)=+\infty$。`

## 原因

相同條件也直接使：

$$
-P(x)\ln Q(x)=+\infty,
$$

因而：

$$
H(P,Q)=+\infty.
$$

程式已正確實作 CE 的 `inf`，例題也展現 KL 的支撐集違規，但定義區只明列 KL，沒有明列 CE。章綱核心明確要求「交叉熵、KL 及熵的域」，因此應補全。

## 最小修法

把支撐集規則第二條改成：

> 若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則該交叉熵項為 $+\infty$，故 $H(P,Q)=+\infty$，且 $D_{\mathrm{KL}}(P\parallel Q)=+\infty$。

這只需修改一句，不必重寫定義段。

## 擴展值域措辭

**原句：**

> `我們採用擴展實數系 $\mathbb R\cup\{+\infty,-\infty\}$。`

合法離散概率下，本章熵、CE、KL、NLL 的最終值屬於：

$$
[0,+\infty].
$$

$-\infty$ 只作為 $\ln0$ 的中間值。

**最小修法：**

> 本章的熵、CE、KL 與 NLL 允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln0=-\infty$。

## 零權重規則

**原句：**

> `$0\cdot\ln0$ 定義為 $0$。`
>
> `$P(x)\ln(0)$ 定義為 $0$（若 $P(x)=0$）。`

兩句重複，第二句容易被誤讀。

**最小修法：**

> 零權重項約定為 $0\ln a=0$，包括 $a=0$；若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$。

---

# 五、KL 非負性證明的範圍

**原句：**

> `本章定義限於有限或可數離散樣本空間 $\mathcal X$。`

命題接著對「任意兩個概率分布」直接使用 Jensen。

## 重新核對

命題結論沒有錯。在可數空間令：

$$
Z(x)=\frac{Q(x)}{P(x)}
$$

於 $P$ 支撐上，則：

$$
\mathbb E_P[Z]
=\sum_{P(x)>0}Q(x)
\leq1.
$$

擴展期望版 Jensen 給出：

$$
\mathbb E_P[-\ln Z]
\geq-\ln\mathbb E_P[Z]
\geq0.
$$

左側允許為 $+\infty$。因此不能以「可數情形不成立」拒稿。

問題在於稿件沒有說明使用擴展 Jensen，也沒有指出左側可能是無限；目前證明更像有限和版本。依本卷「完整證明」契約，應補一句或縮小命題範圍。

## 最小修法

二選一：

1. 命題只聲稱有限離散空間，並說可數情形需擴展 Jensen 或截斷極限；
2. 保留可數空間，補充 $\mathbb E_P[Z]\leq1$ 有限、左側允許 $+\infty$，使用擴展期望版 Jensen。

習題 2 也應明示 $\mathcal X$ 有限或相關熵有限，否則 $H(P,Q)-H(P)$ 可能形式上涉及 $\infty-\infty$。

---

# 六、測試契約

現有 `run_expected_tests()` 確實包含：

- 正常 CE；
- index／one-hot 一致性；
- 零機率 CE；
- 相同分布 KL；
- 支撐集違規 KL；
- token 加權 perplexity；
- 負概率故障；
- 非布林 mask 故障。

因此不能說缺少正常、邊界或故障測試。

仍應增加以下關鍵測試：

1. NaN 概率拒絕；
2. 空有效 mask 拒絕；
3. soft label；
4. Python list index labels；
5. label batch 長度錯誤；
6. $701$ nats perplexity 仍為有限；
7. 超過 `float64` 指數上限時為 `inf`；
8. 批次 KL 安全 mask。

程式題要求「形狀、正常值、支撐集違規與拒絕跨 batch 各寫一個小型 assert」。目前 shape、正常值與違規結果都有；「禁止跨 batch 聚合」主要由 `out.shape==(2,)` 證明。措辭上可再說明 shape assert 同時驗證沒有回傳純量。

**原句：**

> `執行時應輸出 "All expected tests passed."`

這是預期，不是虛構執行紀錄。稿件沒有聲稱測試實際通過。

---

# 七、Fisher 局部展開仍不符合進階定理契約

**原句：**

> `在適當正則條件及小 $\delta$ 下：`
>
> `$$D_{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^TI(θ)\delta$$`

## 原因

這是未證明的進階結果。本卷規約要求未證明進階定理標明引用與條件。目前「適當正則條件」過於籠統，至少涉及：

- 參數化可微；
- 局部支撐適當或固定；
- 可交換微分與積分／期望；
- Fisher 資訊存在；
- $\delta$ 足夠小；
- 餘項是 $o(\|\delta\|^2)$，而不只是未定義的近似符號。

公式也混入 Unicode：

> `I(θ)`

應改成 `I(\theta)`。

## 最小修法

若保留，改為可定位且有限制的敘述：

$$
D_{\mathrm{KL}}(p_\theta\parallel p_{\theta+\delta})
=\frac12\delta^\top I(\theta)\delta
+o(\|\delta\|^2),
$$

並列出基本條件與已核對來源。若沒有查閱來源，最小修法是刪除公式，只保留「KL 的局部二階項與 Fisher 資訊相關」且標成未證明延伸直覺。

---

# 八、校準與養殖能力界線

## 1. ECE

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱邊界、分箱數量與樣本量影響，不能單獨證明校準。

**最小修法：**

> 可搭配可靠度圖、ECE 等檢查，但 ECE 受分箱策略影響，不能單獨證明模型已校準。

## 2. 養殖風險

**原句：**

> `這是一個高風險信號，觸發人工審查。`

高 NLL 是模型低預測機率，不等於真實養殖風險；也可能源於模型失配、感測異常或分布偏移。

**最小修法：**

改成：

> 這是一個高 NLL 的待查信號，可提交人工審查。

稿件已正確說明門檻用驗證集制定、不能查看測試集後調整，且案例為合成、非實際安全閾值。沒有控制設備或真實操作建議。

---

# 九、來源與虛構能力核對

**原句：**

> `本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。`

是否執行程式與是否查閱文獻無關。題目附帶的查閱紀錄沒有 Cover、Goodfellow、Jurafsky–Martin 的逐章或頁碼核對，也沒有 Fisher 展開的定位來源。

**最小修法：**

> 以下為延伸參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。

若作者確實查閱，則列出版本與章節。不能用「未執行外部程式」代替文獻查證聲明。

稿件沒有虛構以下能力：

- 沒有說程式已執行；
- 沒有說測試已通過；
- 沒有說模型已訓練；
- 沒有捏造設備、耗時、收斂率或 benchmark；
- 沒有下載模型或資料；
- 沒有宣稱低 NLL 證明推理能力或安全性。

---

# 十、章稿範圍

正文 3156 字，符合至少 3000 字。已有命題證明、三個手算、自足 NumPy 程式、正常／邊界／故障測試、四類習題與答案。

本章不是完整 Transformer 章，因此不要求完整模型、訓練 loop、KV cache、position 或 attention mask。logits 版穩定 CE 與梯度也已排在下一章，不能因本章未推導 logits 梯度而拒稿。

---

# 最終判定

本輪已修復上一版的批次 KL 除零問題；主程式、數值例題、shape、mask、perplexity 與習題解答基本正確。剩餘程式問題只屬介面與測試覆蓋。

但本章核心要求明確核對熵、CE 與 KL 的域，現稿仍未直接列出 $P>0,Q=0$ 時 CE 為 $+\infty$；可數空間證明也未明示擴展 Jensen；Fisher 局部展開則缺少具體條件與已核對來源。這三項可用少量文字修正，但在完成前仍不宜批准。

VERDICT: REVISE