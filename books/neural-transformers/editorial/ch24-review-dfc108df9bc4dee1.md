# 獨立審稿意見

## 一、重新核算與靜態檢查

本次未執行程式，只依數學與 NumPy 語義檢查。

### 1. 例題數值

PPL 例題：

$$
\operatorname{LSE}([0,1,2,3])\approx3.44019,
$$

另外兩個高信心位置的 NLL 是極小正數，故：

$$
\mathrm{MeanNLL}\approx1.20662,\qquad
\mathrm{PPL}\approx3.3422.
$$

ECE 例題兩箱貢獻為 $0.07$ 與 $0.20$，總計 $0.27$。兩項手算均正確。

### 2. 新增整體 ID/OOD 聚合

程式現在對每個 split 產生 `routine`、`alert`、`all`，且 `all` 是重新串接事件計算，不是平均群組 ECE。這個 reduction 方向正確。

依資料重算：

- ID：兩筆 routine 與一筆 alert 都預測正確；
- OOD：兩筆 routine 正確，一筆 alert 錯誤；
- validation 在 $\tau=.8$ 時只接受正確的 `val-a`，coverage $0.5$、risk $0$，因此 selector 預期選出 $.8$；
- 同一固定閾值套到 OOD 時，只接受高信心但錯誤的 alert，展示 OOD 下拒答可能失效。

整體 OOD PPL 也確實高於 ID PPL，且不等於兩個群組 PPL 的算術平均。新增的整體聚合、validation 流程及 source 跨 split 故障測試均有實質改善。

---

## 二、仍須修正的問題

### 1. Selective Risk 的正文公式仍未和程式統一

**原句：**

$$
R(\tau)=
\frac{\sum_{i:s(x_i)\ge\tau}\mathbb I(\hat y_i\ne y_i)}
{\sum_{i:s(x_i)\ge\tau}\mathbb I(s(x_i)\ge\tau)}.
$$

**原因：**

求和範圍已限制 $s(x_i)\ge\tau$，分母中的 indicator 對每一項都等於一。程式的 `accepted` mask 反而更清楚。雖然數值結果沒有錯，但教材的 shape/reduction 契約應讓公式和實作一致。

**最小修法：**

定義：

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

分母為零時仍按現有規則定義為未定義。

---

### 2. `compute_global_ppl` 的輸入介面與其他函式不一致

**原句：**

```python
if logits.ndim != 3:
```

而 `compute_ece`、`risk_coverage` 都先使用 `np.asarray`。

**原因：**

若使用者傳入巢狀 Python list，ECE 與 Risk–Coverage 可以進入契約化驗證，PPL 卻會因 list 沒有 `.ndim` 而出現 `AttributeError`。程式稱為自足評估介面時，接受的輸入型別應一致。

**最小修法：**

在入口加入：

```python
logits = np.asarray(logits)
labels = np.asarray(labels)
valid_mask = np.asarray(valid_mask)
```

仍須保留 `valid_mask.dtype == np.bool_`，不能自動將整數 mask 轉成布林後掩蓋錯誤。

---

### 3. logits 的全張量有限政策沒有寫入 docstring

**原句：**

```python
if not np.all(np.isfinite(logits)):
    raise ValueError(...)
```

**原因：**

這表示即使 `valid_mask=False`，該 padding 位置的 logits 也不得為 NaN/Inf。此策略保守且合理，但目前文件只說 mask 決定有效 token，容易讓讀者以為無效位置可以含非有限值。

**最小修法：**

在 `compute_stable_log_softmax` 與 `compute_global_ppl` 的 docstring 明列：

> 「所有 logits，包括 mask=False 的位置，都必須有限；mask 只控制 loss reduction。」

---

### 4. PPL overflow 仍以 `print` 作為底層 API 行為

**原句：**

```python
if not np.isfinite(perplexity):
    print(f"Warning: PPL overflow. Mean NLL is {mean_nll}.")
```

**原因：**

底層數值函式直接列印會造成不可控制的副作用，且 `np.exp` 可能先發出 runtime warning。Mean NLL 即使很大仍是可用結果，應由明確回傳契約處理。

**最小修法：**

選擇一種策略並記錄：

- 允許回傳 `inf`，但不列印；
- 或在指數化前檢查 dtype 可表示範圍並拋 `OverflowError`；
- 或回傳 `(ppl, mean_nll, overflowed)`。

另加入 overflow 故障或邊界測試。目前 `1000` logits 測的是穩定 softmax，不是 Mean NLL $1000$ 所造成的 PPL overflow。

---

### 5. `compute_classwise_ece` 仍會在空類別軸回傳 NaN

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 `probs.shape == (N,0)`，迴圈不執行，`np.mean([])` 產生 NaN；若 `probs.shape == (0,C)`，錯誤要到內層 `compute_ece` 才出現。程式題要求完整解答，這兩個非法輸入應在函式入口明確拒絕。

**最小修法：**

加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並補：

- 正常 classwise ECE；
- $N=0$；
- $C=0$；
- row sum 非一；
- label 越界。

目前主測試程式完全沒有呼叫習題答案中的 `compute_classwise_ece`。

---

### 6. `check_sources` 尚未驗證 time 或 group 欄位

**原句：**

```python
for source, time, split, group, logits, label in records:
    if source in owners and owners[source] != split:
        raise ValueError(...)
```

**原因：**

它已正確拒絕同一 source 跨 split，這解決主要洩漏問題；但沒有檢查：

- time 是否有限或可排序；
- 同 source 是否有重複 time；
- split/group 是否在 allowlist；
- label/logits shape 是否一致。

對目前四元示例不一定需要完整 schema，但章稿聲稱保留時間與分組契約，至少應拒絕未知 split 及重複 `(source,time)`，否則重複資料可能被重複計數。

**最小修法：**

加入 split/group allowlist，以及 `(source,time)` 唯一性檢查。新增重複時間故障測試。

---

### 7. `synthetic_eval` 未將 validation 結果放入回傳值

**原句：**

```python
validation = evaluate(...)
tau = select_threshold(...)
...
return result
```

**原因：**

validation 確實只用於選閾值，這點正確；但回傳結果沒有記錄 validation 的 event count、curve、target coverage 或選擇依據。讀者只能看到各 test split 的 `selected_tau`，無法稽核它如何從 validation 得到。

**最小修法：**

回傳：

```python
result["validation", "all"] = validation
result["selection"] = {
    "target_coverage": .5,
    "thresholds": ...,
    "selected_tau": tau,
}
```

這不是讓 test 參與調參，而是保存可追溯的選擇證據。

---

### 8. 對 ID/OOD 的固定閾值測試不夠完整

**原句：**

```python
assert whole["selected_tau"] == .8
```

**原因：**

這只驗證兩個 split 記錄相同閾值，沒有核對 `selected_point` 的 accepted count、coverage 與 risk。此資料其實很適合展示：

- ID 在 $\tau=.8$ 接受高信心正確事件；
- OOD 在 $\tau=.8$ 接受高信心錯誤事件；
- 因此相同拒答規則可能在 OOD 失效。

**最小修法：**

增加預期 assertion，明確核對 tuple 中的 count、errors、coverage、risk。由於未執行，正文仍只稱為預期。

---

### 9. ECE 理論段的限制敘述仍稱其「僅作為相對指標」

**原句：**

> 「因此，ECE 僅作為相對指標，用於比較同條件下的不同模型」

小結已改成較正確的：

> 「它可描述單一模型並輔助比較」

**原因：**

理論段與小結仍互相不一致。ECE 可以描述單一模型的分箱校準差距；比較也只是輔助，不能保證排序穩定。

**最小修法：**

把理論段同步改成小結的表述：

> 「ECE 是依賴資料、樣本量與分箱規則的描述性估計量，可在相同程序下輔助比較，但不能單獨作為品質或安全保證。」

---

### 10. ECE 證明雖已修正空箱概念，但最好給出 $Y_b$ 的明確公式

**原句：**

> 「令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\frac{N_b}{N}|X_b|$。」

**原因：**

此文字在數學上已足以消除未定義問題，主要推導也正確；但本卷要求完整證明，piecewise 公式會比散文定義更清晰，也可避免讀者誤以為 $X_b$ 在空箱仍需定義。

**最小修法：**

加入：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|X_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

這屬小修，不再是推導方向錯誤。

---

### 11. OOD 段落仍把高 PPL 直接歸因於異常泛化能力

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

高 PPL 直接證明的是模型給該群真實 token 較低概率。原因也可能是詞彙、模板、序列長度、tokenizer、標註或資料生成規則不同，不能只由 PPL 得出語義泛化原因。

**最小修法：**

改成：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer 與標註規則。」

---

### 12. 表格仍只給 token 數，沒有 ECE 事件數

**原句：**

| 指標 | ID 集 | OOD 集 |
|---|---|---|
| Total Tokens | 1000 | 1000 |
| ECE | 0.05 | 0.20 |

**原因：**

PPL 的分母是有效 token；ECE 的分母是 calibration event；Risk 的條件分母是 accepted decision。三者不可由同一個 `Total Tokens` 代表。後文已說 decision count 另列，但手設表格仍未提供 calibration event count。

**最小修法：**

新增：

- Valid token count；
- Calibration event count；
- Decision count；
- Accepted count at $\tau$。

若只是手設 PPL 表，也可刪除 ECE 與拒答數字，避免混合不同評估單位。

---

### 13. 整合題解答仍未利用新增的 validation/test 流程

**原句：**

> 「繪製 R-Curve：x 軸為 Coverage，y 軸為 Risk。」

> 「OOD 集可能需要更高的 $\tau$」

**原因：**

章內程式現在已有 validation 選 $\tau$、固定套到 ID/OOD 的流程，但習題解答仍停留在概念敘述。它也沒有說明 OOD 不一定能靠提高 $\tau$ 達到目標 risk，而章內反例已證明 risk 不單調。

**最小修法：**

完整答案應寫：

1. 在 validation 掃描 thresholds；
2. 依 coverage 約束選 $\tau^\star$；
3. 固定 $\tau^\star$；
4. 套到 ID-test 與 OOD-test；
5. 報告 event、accepted、coverage、risk；
6. 若 validation 無可行候選則報告不可達；
7. 不在 OOD-test 重調；
8. 若 OOD 信心排序失效，提高 $\tau$ 可能無效或變差。

---

### 14. `risk_coverage` 的綜合錯誤訊息不利故障定位

**原句：**

```python
raise ValueError("Invalid decision data or thresholds.")
```

**原因：**

shape、空輸入、NaN、超出範圍、非二元 correctness、非法 threshold 都共用一個訊息。功能上能拒絕，但測試難以驗證具體故障原因。

**最小修法：**

拆成數個順序檢查並使用不同訊息；至少加入 NaN confidence、非二元 correctness、空 thresholds 的故障測試。

---

### 15. `select_threshold` 沒有檢查 duplicate thresholds，但可保留或明說

**原句：**

```python
rows = risk_coverage(... thresholds)
```

**原因：**

重複 threshold 會產生重複候選；目前不影響正確性，但可能使輸出曲線含重複點。這不是阻擋問題，只需定義政策。

**最小修法：**

可要求 thresholds 嚴格遞增，或明說保留輸入順序及重複值。若要求曲線用於繪圖，建議拒絕非遞增 thresholds。

---

### 16. PPL 故障訊息和文字預期不完全相同

**原句：**

測試說：

> 「拋出 `ValueError: Label out of bounds.`」

實際程式是：

```python
raise ValueError("Label out of bounds for valid tokens.")
```

**原因：**

測試程式目前只檢查例外類型，所以不會失敗；但正文聲稱的預期訊息不精確。

**最小修法：**

統一文字與程式訊息，或正文只寫「預期拋出 `ValueError`」。

---

### 17. 引用核對狀態仍有不當概括

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻說：

> 「以上引用為標準學術參考」

**原因：**

待核對來源不能被概括成已確認的依據。提供的來源說明亦指出 N6 未逐條核對，N1 只取得摘要。

**最小修法：**

末註改成：

> 「N1 僅核對摘要；N6 及新增校準文獻尚待逐條核對，目前只列作延伸閱讀，不作已查證證據。」

---

## 三、shape、broadcast、mask 與 reduction

目前 PPL 的核心 shape 正確：

- logits：$(N,L,V)$；
- labels／mask：$(N,L)$；
- gather 結果：$(N,L)$；
- log-softmax：沿最後的 $V$ 軸；
- NLL：有效 token 求總和後除一次。

`safe_labels` 能避免 padding 的 $-1$ 被 NumPy 當作最後一類；`np.where` 也避免以乘法處理 mask 時讓無效位置污染 reduction。

`risk_coverage` 中 confidence 與 correctness 都是 $(N,)$，threshold 逐一迭代，沒有錯誤 broadcast。`evaluate(part)` 將二類 logits 變為 $(N,1,2)$，labels 與 mask 為 $(N,1)$，亦符合 PPL 函式契約。

本章沒有反向傳播，無梯度需核對；這符合評估章範圍。本章也不需要重複完整 Transformer 或訓練 loop，但 classwise ECE 解答、端到端 validation/test 稽核輸出及失敗測試仍須完整。

---

## 四、可保留內容

以下內容目前可保留：

1. token 加權 NLL 與 PPL；
2. 穩定 log-softmax；
3. 極小正 NLL 的精確／近似區分；
4. 不跨文件重跑模型；
5. 固定分箱 ECE 的 Jensen 推導；
6. 空箱貢獻的正確處理方向；
7. coverage 與 selective risk；
8. 空接受集 risk 為 NaN；
9. risk 非單調反例；
10. validation 選固定閾值；
11. ID/OOD 分群及整體重新聚合；
12. 不平均 group PPL/ECE；
13. source 跨 split 故障檢查；
14. 正常、邊界與故障測試；
15. 手設表格與程式資料明確區分；
16. 不以 token confidence 保證回答安全；
17. 未聲稱已執行程式；
18. 正文已達最低字數。

---

## 五、結論

現稿已大致完成本章的核心數學與實作：PPL、ECE、Risk–Coverage、validation 閾值、ID/OOD/group 聚合及故障測試均已有可讀程式。剩餘問題比前版少，但仍有一個實質阻擋項：程式題的 `compute_classwise_ece` 對空樣本／空類別未定義且完全沒有測試，不符合「程式題完整解答」與正常／邊界／故障測試契約。另有正文 selective-risk 公式、PPL API、overflow、validation 稽核輸出、表格分母、整合題答案、OOD 因果解讀與未核對引用等需同步修正。完成這些最小修法後才適合核准。

VERDICT: REVISE