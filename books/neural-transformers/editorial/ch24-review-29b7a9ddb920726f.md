# 獨立審稿意見

## 一、先行重算

本次未執行任何程式，只依公式及 NumPy 語義檢查。

### 1. PPL 例題

第一個位置：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(1+e^{-1}+e^{-2}+e^{-3})
\approx3.44019.
$$

第二、三個位置的 NLL 分別為：

$$
\ln(1+3e^{-110}),\qquad \ln(1+3e^{-101}),
$$

皆為極小正數。加上均勻 logits 的 $\ln4$：

$$
\mathrm{MeanNLL}\approx1.20662,
\qquad
\mathrm{PPL}\approx3.3422.
$$

手算及 `test_normal` 的容差合理。

### 2. ECE 例題

兩箱貢獻分別為：

$$
0.3\left|0-\frac{0.7}{3}\right|=0.07,
$$

$$
0.7\left|\frac37-\frac57\right|=0.20.
$$

總 ECE 為 $0.27$，正確。

### 3. 確定性 ID/OOD 資料

validation 的兩筆 confidence 約為 $0.8808$ 與 $0.7311$，正誤為 $[1,0]$。候選閾值 $[0,.8,1]$、目標 coverage $0.5$ 時：

- $\tau=0$：coverage $1$、risk $0.5$；
- $\tau=.8$：coverage $0.5$、risk $0$；
- $\tau=1$：接受數零、risk 未定義。

因此預期選到 $\tau=.8$。OOD 在該閾值下只接受高信心但錯誤的 alert 事件，能正確展示 OOD 下 selective risk 可能惡化。整體 OOD PPL 也預期高於 ID PPL。

---

## 二、仍須修正的實質問題

### 1. `compute_classwise_ece` 對空樣本與空類別未定義

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 `probs.shape == (N, 0)`，迴圈不執行，`np.mean([])` 產生 NaN，而不是明確拒絕。若 `probs.shape == (0, C)`，則會延遲到內層 `compute_ece` 才失敗。這是習題要求的完整程式解答，不能留下未定義空維度。

此外，主程式的正常、邊界及故障測試沒有任何一個呼叫 `compute_classwise_ece`。因此程式題雖有答案，仍未滿足完整解答及測試契約。

**最小修法：**

加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並至少測試：

1. 合法二類概率；
2. $N=0$；
3. $C=0$；
4. row sum 不為一；
5. label 越界；
6. 非有限概率。

這是目前最直接的阻擋項。

---

### 2. Selective Risk 的數學公式仍未與程式一致

**原句：**

$$
R(\tau)=
\frac{\sum_{i:s(x_i)\ge\tau}\mathbb I(\hat y_i\ne y_i)}
{\sum_{i:s(x_i)\ge\tau}\mathbb I(s(x_i)\ge\tau)}.
$$

**原因：**

求和索引已限制 $s(x_i)\ge\tau$，因此分母中的 indicator 對每一項都等於一。程式使用 `accepted` mask 對所有樣本 reduction，反而更清楚。公式數值沒有錯，但不利於核對 shape、accepted count 與空接受集。

**最小修法：**

先定義：

$$
a_i(\tau)=\mathbb I(s(x_i)\ge\tau),
$$

再寫：

$$
C(\tau)=\frac1N\sum_i a_i(\tau),
$$

$$
R(\tau)=
\frac{\sum_i a_i(\tau)\mathbb I(\hat y_i\ne y_i)}
{\sum_i a_i(\tau)}.
$$

分母為零時 risk 未定義。

---

### 3. ECE 理論段與小結仍不一致

**原句：**

理論段說：

> 「ECE 僅作為相對指標，用於比較同條件下的不同模型」

小結則正確地說：

> 「它可描述單一模型並輔助比較」

**原因：**

ECE 不只是相對指標，它也可描述單一模型的分箱校準差距。另一方面，即使使用相同資料與 bins，不同模型的信心分布和箱占比不同，ECE 排序也不必穩健。

**最小修法：**

將理論段統一改為：

> 「ECE 是依賴資料、樣本量及分箱規則的描述性估計量，可在相同評估程序下輔助比較，但不能單獨作為絕對品質或安全保證。」

---

### 4. 空箱證明已修正方向，但最好明列 piecewise 公式

**原句：**

> 「令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\frac{N_b}{N}|X_b|$。」

**原因：**

這段散文定義已足以修補先前未定義問題，後續推導也正確。不過本卷要求完整證明，將它寫成正式 piecewise 公式更自足，並避免讀者誤以為 $X_b$ 在空箱仍有值。

**最小修法：**

加入：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|X_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

這是小修，不需改變後續推導。

---

### 5. `compute_global_ppl` 的輸入型別契約與其他函式不一致

**原句：**

```python
if logits.ndim != 3:
```

而 `compute_ece` 與 `risk_coverage` 都先使用 `np.asarray`。

**原因：**

如果呼叫端傳入巢狀 list，ECE 與 Risk–Coverage 可進入明確驗證，PPL 卻會因 list 沒有 `.ndim` 而拋出 `AttributeError`。要嘛所有函式只接受 ndarray，要嘛都接受 array-like，現在兩者不一致。

**最小修法：**

入口加入：

```python
logits = np.asarray(logits)
labels = np.asarray(labels)
valid_mask = np.asarray(valid_mask)
```

轉換後仍保留布林 mask dtype 檢查，不能把整數 mask 靜默轉為布林。

---

### 6. 全 logits 有限的政策沒有寫入函式契約

**原句：**

```python
if not np.all(np.isfinite(logits)):
    raise ValueError(...)
```

**原因：**

這表示即使 `valid_mask=False`，padding 位置也不得含 NaN/Inf。這是合理而保守的政策，但 docstring 沒有說明；讀者可能誤認 loss mask 會豁免無效位置的非有限值。

**最小修法：**

在 `compute_stable_log_softmax` 及 `compute_global_ppl` 的 docstring 明列：

> 「所有 logits，包括 mask=False 的位置，均須有限；mask 只控制 loss reduction。」

---

### 7. PPL overflow 仍用 `print` 作為底層介面

**原句：**

```python
if not np.isfinite(perplexity):
    print(f"Warning: PPL overflow. Mean NLL is {mean_nll}.")
```

**原因：**

底層數值函式直接列印有副作用，而且 `np.exp` 可能先產生 runtime warning。Mean NLL 本身仍可能有效，應讓呼叫端透過回傳值或例外處理，而不是解析標準輸出。

目前的 `[1000,0]` 測試只驗證穩定 softmax，不會產生 Mean NLL $1000$；因此也沒有真正測到 PPL overflow 策略。

**最小修法：**

可選擇：

- 回傳 `inf` 且不列印；
- 指數化前檢查範圍並拋 `OverflowError`；
- 回傳明確 overflow 狀態。

另用真類別位於低 logit 的例子測試大 Mean NLL。

---

### 8. validation 選擇證據沒有被回傳保存

**原句：**

```python
validation = evaluate(...)
tau = select_threshold(...)
...
return result
```

**原因：**

閾值確實只由 validation 選出，沒有 test 洩漏，這一點正確。但回傳值中沒有 validation curve、target coverage 或候選 thresholds，因此無法從結果物件稽核 `.8` 是如何被選出的。每個 test row 只有 `selected_tau`，缺少選擇證據。

**最小修法：**

增加：

```python
result["validation", "all"] = validation
result["selection"] = {
    "thresholds": [0, .8, 1],
    "target_coverage": .5,
    "selected_tau": tau,
}
```

這不會讓 test 參與調參，只是保存可追溯狀態。

---

### 9. 正常測試沒有核對固定閾值下的 ID/OOD risk

**原句：**

```python
assert whole["selected_tau"] == .8
```

**原因：**

這只核對兩個 split 記錄了同一閾值，沒有驗證 `selected_point` 中的 accepted count、errors、coverage、risk。新增資料最重要的整合證據正是：相同 $\tau$ 在 ID 與 OOD 的結果不同。

**最小修法：**

加入對 `selected_point` 五個欄位的預期檢查。例如至少核對 OOD 在 $.8$ 下接受的樣本為錯誤，以及 ID 對應接受樣本的錯誤數。若不希望把具體數字寫死，也應核對 OOD risk 高於 ID risk。

---

### 10. source 契約只檢查跨 split，沒有檢查重複時間與非法 split

**原句：**

```python
def check_sources(records):
    ...
    if source in owners and owners[source] != split:
        raise ValueError(...)
```

**原因：**

它已正確允許同來源多個時間點，並拒絕跨 split，這比前版正確。但若同一 `(source,time)` 被重複列入，資料會被重複計數；未知 split/group 也會被靜默保留。由於本章強調 group/time/生成規則，最小 schema 驗證仍不足。

**最小修法：**

加入：

- split allowlist；
- group allowlist；
- `(source,time)` 唯一性；
- time 有限或為合法整數。

再補重複時間故障測試。

---

### 11. 手設表格仍混合不同評估分母

**原句：**

表格列出：

> 「Total Tokens | 1000 | 1000」

同時列出 ECE。

**原因：**

PPL 分母是有效 token；ECE 分母是 calibration event；selective risk 的條件分母是 accepted decision。表格只有 token 數，不能支持 ECE 的樣本規模。

**最小修法：**

新增：

- Valid token count；
- Calibration event count；
- Decision count；
- Accepted count at $\tau$。

若不想增加欄位，可把表格拆成 token-level 與 decision-level 兩張。

---

### 12. OOD 段落仍過度推論成因

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

高 PPL 直接表示模型給告警群真實 token 較低概率。原因也可能是詞彙、模板、序列長度、tokenizer、標註或合成規則差異，不能只由 PPL 判定語義泛化能力。

**最小修法：**

改成：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer 與標註規則。」

---

### 13. 整合題解答仍不完整，且對 OOD 閾值作一般化推論

**原句：**

> 「因此，為了達到相同的 Risk，OOD 集可能需要更高的 $\tau$」

**原因：**

章內已用反例證明 risk 不隨 $\tau$ 單調。OOD 可能需要更高閾值，也可能沒有任何閾值能達到目標 risk。答案沒有利用新增的 validation/test 程式流程，也沒有說明不可達時的報告策略。

**最小修法：**

完整答案應列出：

1. validation 上掃描 thresholds；
2. 在 coverage 約束下選 $\tau^\star$；
3. 固定 $\tau^\star$；
4. 套到 ID-test、OOD-test；
5. 回報 decision、accepted、coverage、risk；
6. validation 無可行候選時報告不可達；
7. 不在 OOD-test 重選閾值；
8. OOD 信心排序失效時，提高 $\tau$ 可能變差。

---

### 14. `risk_coverage` 的 thresholds 契約不清楚

**原句：**

```python
thresholds = np.asarray(thresholds, dtype=float)
```

**原因：**

函式允許未排序或重複 thresholds。數值本身仍正確，但若稱為 R-Curve，輸出順序可能不是依閾值或 coverage 排列，重複值也會產生重複點。

**最小修法：**

二選一：

- 要求 thresholds 嚴格遞增，否則拒絕；
- 或文件明說保留輸入順序，呼叫端自行排序及去重。

並加入非遞增 threshold 的邊界或故障測試。

---

### 15. 故障測試只檢查例外型別，不檢查故障來源

**原句：**

```python
def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
```

**原因：**

任何意外的 `ValueError` 都會被視為預期成功。例如未來函式在更早的錯誤位置失敗，測試仍通過。這不代表必須逐字匹配完整訊息，但至少可檢查關鍵子字串。

**最小修法：**

讓 helper 接收預期訊息片段：

```python
expect_value_error(fn, "No valid tokens", ...)
```

並檢查 `str(exc)`。同時統一正文的 `"Label out of bounds."` 與程式實際 `"Label out of bounds for valid tokens."`。

---

### 16. 引用核對狀態仍不一致

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻說：

> 「以上引用為標準學術參考」

**原因：**

待核對來源不能被概括為已確認依據。N6 依來源備註也未逐條核對，N1 只取得摘要頁。

**最小修法：**

末註改為：

> 「N1 僅核對摘要；N6 及新增校準文獻尚待逐條核對，目前只列作延伸閱讀，不作已查證證據。」

---

## 三、shape、broadcast、mask 與 reduction 核對

目前主要 shape 正確：

- logits：$(N,L,V)$；
- labels、valid mask：$(N,L)$；
- gather：$(N,L)$；
- log-softmax 沿 $V$ 軸；
- NLL 對有效 token 求和後除一次；
- confidence/correctness：$(N,)$；
- threshold 逐值計算。

`safe_labels` 能防止 padding 的 $-1$ 被 NumPy 當成最後一類；`np.where` 也正確避免無效位置進入 reduction。整體 split 的 ECE 是重新聚合原始事件，不是平均群組 ECE；整體 PPL 也由總 NLL／總 token 數推得，均符合契約。

本章不涉及反向傳播，無梯度錯誤。本章也不必重複完整 Transformer 或訓練 loop；目前評估 loop 的範圍已基本足夠。

---

## 四、可保留內容

以下部分目前正確：

1. token 加權 PPL；
2. 兩個逐步手算；
3. 穩定 log-softmax；
4. mask、shape、label 範圍及空 token 拒絕；
5. 文件邊界不跨接；
6. ECE 的固定分箱 Jensen 推導；
7. 空箱貢獻變數 $Y_b$ 的概念；
8. coverage 與 selective risk；
9. 空接受集回傳 NaN；
10. validation 選閾值；
11. 同一閾值套用 ID/OOD；
12. ID/OOD 分群與整體聚合；
13. 不平均群組 PPL/ECE；
14. source 跨 split 故障測試；
15. risk 非單調反例；
16. 正常、邊界與故障測試；
17. 手設表格與程式資料分開；
18. 未虛構執行、訓練或硬體能力。

---

## 五、結論

現稿的核心數學及主要評估程式已接近完成，ID/OOD 整體聚合、validation 閾值及來源洩漏檢查亦已補上。現在最主要的阻擋項是程式題 `compute_classwise_ece` 仍對 $N=0$、$C=0$ 未定義且完全沒有測試，因而不符合完整程式解答要求。另須同步修正文中的 risk 公式、PPL API/overflow、validation 稽核輸出、表格分母、OOD 因果解讀、整合題不可達情況與引用核對狀態。這些修正雖已較局部，但仍涉及故障測試和完整解答契約，故本輪仍需退修。

VERDICT: REVISE