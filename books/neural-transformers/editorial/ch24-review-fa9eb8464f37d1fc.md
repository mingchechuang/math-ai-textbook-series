# 第24章獨立審稿

## 一、重算結果

本次未執行程式，也不宣稱測試已通過；以下依公式與 NumPy 語義靜態核對。

### 1. PPL 例題

第一個位置：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(e^{-3}+e^{-2}+e^{-1}+1)
\approx3.44019.
$$

第二、三位置的 NLL 分別為：

$$
\ln(1+3e^{-110}),\qquad
\ln(1+3e^{-101}),
$$

兩者都是極小正數，而不是精確零。第四位置為 $\ln4\approx1.38629$。因此：

$$
\mathrm{MeanNLL}\approx\frac{3.44019+1.38629}{4}
\approx1.20662,
$$

$$
\mathrm{PPL}\approx e^{1.20662}\approx3.3422.
$$

正文與測試的數值正確。

### 2. ECE 例題

第一箱貢獻：

$$
\frac3{10}\left|0-\frac{0.7}{3}\right|=0.07.
$$

第二箱貢獻：

$$
\frac7{10}\left|\frac37-\frac57\right|=0.20.
$$

故總 ECE 為 $0.27$，正確。

### 3. validation 閾值

validation 兩筆資料的 confidence 約為 $0.8808$、$0.7311$，correctness 為 $[1,0]$：

- $\tau=0$：接受兩筆，coverage $1$、risk $1/2$；
- $\tau=.8$：只接受第一筆，coverage $1/2$、risk $0$；
- $\tau=1$：不接受任何資料，risk 未定義。

因此在 target coverage $0.5$ 下，預期選到 `.8`。固定 `.8` 套到 OOD 時，只留下高信心錯誤 alert，符合章中「risk 不必隨閾值下降」的反例。

---

## 二、數學定義仍有阻擋缺口

### 1. 母體零機率箱 $q_b=0$ 仍未處理

**逐字原句：**

> 「定義母體固定分箱 ECE 為：
> $$\text{ECE}_{\text{true, binned}} = \sum_{b=1}^{B} q_b \left| E[A - S \mid S \in I_b] \right|$$」

以及：

> 「令 $\delta_b = E[A - S \mid S \in I_b]$」

**原因：**

若 $q_b=P(S\in I_b)=0$，則條件事件 $S\in I_b$ 的機率為零，通常不能直接定義

$$
E[A-S\mid S\in I_b].
$$

目前證明處理的是樣本空箱 $N_b=0$，但沒有處理母體零機率箱 $q_b=0$。兩者不是同一件事。將未定義的條件期望乘以零仍不能自動成為合法定義。

**最小修法：**

把母體固定分箱 ECE 改為：

$$
\mathrm{ECE}_{\mathrm{true,binned}}
=\sum_{b:q_b>0}q_b|\delta_b|,
\qquad
\delta_b=E[A-S\mid S\in I_b].
$$

再分情況證明：

- 若 $q_b=0$，則 $N_b=0$ 幾乎處處，所以 $Y_b=0$ 幾乎處處，因而 $E[Y_b]=0$；
- 若 $q_b>0$，才使用條件分布及 Jensen 不等式。

這是完整證明的定義域問題，不是風格偏好。

---

### 2. $Y_b$ 仍應正式作分段定義

**逐字原句：**

> 「令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\frac{N_b}{N}|X_b|$。」

**原因：**

這段文字已修正先前直接在空箱使用 $X_b$ 的主要問題，但 $X_b=\mathrm{Acc}_b-\bar p_b$ 本身只在 $N_b>0$ 時有定義。完整證明最好直接以不引用空箱 $X_b$ 的分段式定義 $Y_b$。

**最小修法：**

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|\mathrm{Acc}_b-\bar p_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

之後只在條件 $N_b=n>0$ 下引入 $X_b$。如此整個樣本空間上的 $Y_b$ 才明確有定義。

---

### 3. Selective Risk 公式仍未與程式的 reduction 形式統一

**逐字原句：**

$$
R(\tau) =
\frac{\sum_{i: s(x_i) \ge \tau} \mathbb{I}(\hat{y}_i \neq y_i)}
{\sum_{i: s(x_i) \ge \tau} \mathbb{I}(s(x_i) \ge \tau)}.
$$

**原因：**

求和索引已限制 $s(x_i)\ge\tau$，所以分母中的 indicator 對每個求和項都是一。公式數值沒有錯，但它重複限制索引，且不如程式的布林接受 mask 清楚。

**最小修法：**

定義：

$$
a_i(\tau)=\mathbb I(s(x_i)\ge\tau),
$$

然後寫：

$$
C(\tau)=\frac1N\sum_{i=1}^Na_i(\tau),
$$

$$
R(\tau)=
\frac{\sum_{i=1}^Na_i(\tau)\mathbb I(\hat y_i\ne y_i)}
{\sum_{i=1}^Na_i(\tau)}.
$$

若分母為零，risk 未定義。這也與程式的：

```python
accepted = confidence >= tau
```

完全一致。

---

## 三、程式與故障測試

### 4. 新增 `OverflowError`，但沒有任何對應測試

**逐字原句：**

```python
if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
    raise OverflowError("PPL not representable; record mean NLL instead.")
```

**原因：**

這個政策本身合理：Mean NLL 可有限，但其指數可能超出 `float64` 表示範圍。然而現有 helper 只捕捉 `ValueError`：

```python
def expect_value_error(fn, *args):
```

現有邊界測試：

```python
z = np.array([[[1000., 0.]]])
compute_global_ppl(z, np.array([[0]]), ...)
```

真標籤是最高 logit 類別，所以 NLL 接近零，完全不會觸發 PPL overflow。該測試只能證明減最大值的 log-softmax 預期保持有限，不能測新增的 overflow 分支。

**最小修法：**

增加：

```python
def expect_error(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError("Expected exception")
```

再測：

```python
expect_error(
    OverflowError,
    compute_global_ppl,
    np.array([[[1000., 0.]]]),
    np.array([[1]]),
    np.array([[True]])
)
```

此例的真類別位於低 logit，Mean NLL 約為 $1000$，預期觸發 `OverflowError`。

---

### 5. `compute_classwise_ece` 已補空維度，但仍沒有測試

**逐字原句：**

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

**原因：**

此修正方向正確，並解決 `np.mean([])` 靜默產生 NaN 的問題。但習題解答仍只提供函式，沒有正常、邊界或故障測試。依章稿契約，程式題要有完整解答；新增分支不能完全未被檢查。

`compute_classwise_ece` 的 shape/reduction 本身合理：

- `probs`：$(N,C)$；
- `labels`：$(N,)$；
- 類別 $c$ 的 confidence：$(N,)$；
- 類別 $c$ 的二元事件：$(N,)$；
- macro ECE：沿 $C$ 個 classwise ECE 作平均。

但至少要附下列預期檢查：

1. 合法二類輸入；
2. $N=0$；
3. $C=0$；
4. row sum 不等於一；
5. label 越界；
6. NaN probability。

**最小修法：**

在習題答案後加入小型測試，並明示未執行。例如：

```python
eces, macro = compute_classwise_ece(
    [[.8, .2], [.3, .7]], [0, 1], [0, .5, 1])
assert len(eces) == 2
assert np.isclose(macro, np.mean(eces))
```

再用例外 helper 覆蓋上述故障案例。

---

### 6. validation 選擇證據仍未存入回傳結果

**逐字原句：**

```python
validation = evaluate([r for r in records if r[2] == "valid"])
tau = select_threshold(...)
...
return result
```

**原因：**

閾值確實只由 validation 選擇，沒有 test 洩漏；這一點正確。但是 `result` 中只有 ID/OOD row 的 `selected_tau`，沒有：

- validation curve；
- coverage target；
- 候選 thresholds；
- validation selected point。

因此呼叫端無法只依回傳結果稽核 `.8` 的選擇依據。

**最小修法：**

加入：

```python
validation["curve"] = risk_coverage(
    validation["confidence"],
    validation["correctness"],
    [0, .8, 1]
)
result["validation", "all"] = validation
result["selection"] = {
    "thresholds": [0, .8, 1],
    "target_coverage": .5,
    "selected_tau": tau,
}
```

這只是保存 validation 證據，不會使用 test 調參。

---

### 7. `selected_point` 已計算，測試卻完全不核對

**逐字原句：**

```python
row["selected_point"] = risk_coverage(
    row["confidence"], row["correctness"], [tau])[0]
```

但正常測試只寫：

```python
assert whole["selected_tau"] == .8
```

**原因：**

這只核對閾值數字，沒有核對 accepted count、errors、coverage 與 risk。若 `selected_point` 的欄位錯置或 mask 錯誤，現有測試仍可能通過。

**最小修法：**

至少加入：

```python
id_point = result["id", "all"]["selected_point"]
ood_point = result["ood", "all"]["selected_point"]

assert id_point[1] > 0
assert id_point[2] == 0
assert ood_point[1] > 0
assert ood_point[2] == ood_point[1]
assert ood_point[4] > id_point[4]
```

依目前資料，ID 的高信心接受事件正確，而 OOD 的高信心接受事件錯誤。這才實際測到本章 OOD selective-risk lab 的核心結論。

---

### 8. `check_sources` 仍不能拒絕重複來源時間記錄

**逐字原句：**

```python
if source in owners and owners[source] != split:
    raise ValueError("Source crosses splits.")
```

**原因：**

同來源跨 split 已正確拒絕，這是主要洩漏防線。但同一 `(source,time)` 可以重複出現，造成同一事件被重複納入 PPL、ECE 與 risk。未知 split/group 也會被靜默接受。

**最小修法：**

加入：

```python
valid_splits = {"valid", "id", "ood"}
valid_groups = {"routine", "alert"}
seen = set()
```

逐筆檢查：

```python
if split not in valid_splits or group not in valid_groups:
    raise ValueError("Invalid split or group.")
key = (source, time)
if key in seen:
    raise ValueError("Duplicate source-time record.")
seen.add(key)
```

並補一個重複 `(source,time)` 故障案例。

---

### 9. `compute_stable_log_softmax` 的公開契約與實際接受 shape 不一致

**逐字原句：**

```python
Input: logits (N, L, V)
Output: log_probs (N, L, V)
```

函式本身卻沒有 `ndim == 3` 檢查。

**原因：**

它實際上沿最後軸計算，可以接受一維、二維或更高維張量；但 docstring 宣稱只接受三維。單獨呼叫時，純 Python list 也沒有先明確轉為 ndarray。

**最小修法：**

二選一：

- 在函式入口 `np.asarray` 並要求三維；
- 或把 docstring 改成「接受至少一維 array-like，沿最後軸計算」。

因 `compute_global_ppl` 已自行保證 $(N,L,V)$，較簡單的做法是讓此底層函式正式接受任意至少一維張量，並拒絕零維與末軸大小零。

---

## 四、OOD 報告範圍仍需修正

### 10. 表格仍以 token 數搭配 ECE，沒有 calibration event count

**逐字原句：**

| Total Tokens | 1000 | 1000 |
| ECE (5 bins) | 0.05 | 0.20 |

**原因：**

PPL 的分母是 valid token count；ECE 的分母是 calibration event count；selective risk 的條件分母是 accepted decision count。三者不應默認相同。後文雖說決策數要另列，但表格沒有提供 ECE 事件數。

**最小修法：**

把表格拆成或新增：

- Valid token count；
- Calibration event count；
- Decision count；
- Accepted count at $\tau$。

若手設案例中每個有效 token 恰好就是一個 calibration event，也須逐字說明，不能只列 `Total Tokens`。

---

### 11. PPL 仍被過度解讀為語義泛化失敗

**逐字原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

高 PPL 直接支持的是模型對該群真實 token 給出較低概率，或模型與該群 token 分布匹配較差。它可能來自詞彙、模板、序列長度、tokenizer、上下文或標註規則差異，不能只由 PPL 判定語義泛化能力。

**最小修法：**

改為：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer、上下文與標註規則。」

---

### 12. 小結對 OOD 的概括也稍強

**逐字原句：**

> 「OOD 評估揭示模型在分布偏移下的脆弱性」

**原因：**

本章只有一組極小的確定性資料和一張手設表格，沒有統計推論，不能概括證明模型普遍脆弱。

**最小修法：**

改為：

> 「OOD 評估量化模型在指定合成偏移下的表現差異；分群分析可協助定位待進一步驗證的失敗模式。」

---

## 五、來源與能力聲明

### 13. 引用末註仍不忠實反映核對狀態

**逐字原句：**

> 「以上引用為標準學術參考，具體版本與細節請查閱原始論文。」

**原因：**

提供的來源備註明確指出：

- N1 只核對摘要；
- N6 尚未逐條核對；
- Guo et al. 尚待核對；
- Niculescu-Mizil & Caruana 尚待核對。

把它們統稱為「標準學術參考」容易造成已核對其內容的印象。

**最小修法：**

末註改為：

> 「N1 僅核對摘要；N6、Guo et al. 與 Niculescu-Mizil & Caruana 尚未逐條核對，目前只列作延伸閱讀，不作已查證證據。本章未使用未公開實測資料。」

---

## 六、整體判定

本稿已符合下列核心要求：

- token 加權 PPL；
- logits 版穩定 NLL；
- 空有效 token 拒絕；
- 固定分箱 ECE；
- ID/OOD 與分群結果；
- validation 選閾值、test 不調參；
- risk 非單調反例；
- token confidence 不外推為整段回答安全性；
- 未虛構執行或訓練結果；
- 正文達最低中文字數。

但目前 ECE 命題仍在 $q_b=0$ 時使用未定義條件期望，完整證明尚有定義域缺口；新增的 classwise ECE 與 PPL overflow 分支沒有相應測試；validation 選擇證據和固定閾值 ID/OOD risk 沒有被整合測試核對。OOD 表格分母、資料重複防線、PPL 語義解讀及引用狀態也尚未修正。因此本輪仍不能核准。

VERDICT: REVISE