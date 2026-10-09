# 獨立審稿意見

## 一、重算與總體判斷

本次只審當前章稿，未執行任何程式。

### 1. 困惑度

對第一個 token：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(1+e^{-1}+e^{-2}+e^{-3})
\approx3.44019.
$$

第二、三個 token 的精確 NLL 分別是：

$$
\ln(1+3e^{-110}),\qquad \ln(1+3e^{-101}),
$$

均嚴格大於零，但在四位小數下近似為零。故：

$$
\overline{\mathrm{NLL}}\approx1.20662,
\qquad
\mathrm{PPL}\approx3.3422.
$$

現稿此處已修正正確。

### 2. ECE

兩個箱的貢獻分別為：

$$
\frac3{10}\left|0-\frac{0.7}{3}\right|=0.07,
$$

$$
\frac7{10}\left|\frac37-\frac57\right|=0.20.
$$

故 ECE 為 $0.27$，手算正確。

### 3. 分群 PPL

$$
\frac{800(1.5)+200(5.5)}{1000}=2.3,
$$

$$
e^{2.3}\approx9.974.
$$

表格與分群算術一致。章稿也已正確說明不能平均群組 PPL。

現稿的主要數學已顯著改善，但指定 lab、自足測試及 Risk–Coverage 實作仍缺失，且 ECE 命題措辭與形式化仍未完全一致，尚不能核准。

---

## 二、必須修正的問題

### 1. 指定的 ID/OOD lab 沒有程式實作

**原句：**

> 「OOD 評估：必須比較保留集（ID）與合成偏移集（OOD）的分群指標」

實作部分卻只有：

```python
compute_stable_log_softmax
compute_global_ppl
compute_ece
```

**原因：**

沒有任何程式資料或 loop 支援：

- ID/OOD split；
- routine/alert group；
- group、time、source ID、seed；
- 按 split/group 聚合有效 token；
- ID/OOD 的 PPL 對照；
- 分群 ECE；
- calibration event count；
- decision count；
- Risk–Coverage；
- 資料來源重疊檢查。

章稿已誠實把表格改稱「手設情境，非執行結果」，這避免了虛構執行，但也表示 lab 尚未完成。手設表格不能替代「自足 CPU 程式」。

**最小修法：**

補一個很小的確定性 NumPy 資料集即可，不一定需要完整文字生成器。每筆至少保存：

- `source_id`；
- `time`；
- `split`；
- `group`；
- logits、label、valid mask；
- confidence、correctness。

然後提供按 split/group 評估的完整 loop。若使用常數陣列，不需宣稱隨機 seed；若使用亂數，則提供 seed 與完整生成規則。

---

### 2. 「合成資料」與「手設假設數值」仍有衝突

**原句：**

> 「下表是手設假設數值，並非由 seed 或程式生成的日誌評估。」

稍後又說：

> 「本節所有數據均為合成資料。」

**原因：**

手設的聚合指標不是由合成樣本計算出的「合成資料結果」。它只能稱為假設或示意數值。現在雖沒有聲稱已執行，但證據層級仍不一致。

**最小修法：**

在補生成器之前，改為：

> 「本節所有數值均為手設示意，並非真實資料、實測結果或已執行的合成實驗。」

若之後補齊生成器，才可稱為合成資料的預期評估結果。

---

### 3. Risk–Coverage 仍只有公式，沒有程式

**原句：**

> 「實際應在驗證集檢查完整 Risk–Coverage 曲線」

**原因：**

這是本章核心契約之一，但章內沒有任何可計算 R-Curve 的函式。缺少：

- confidence shape $(N,)$；
- correctness shape $(N,)$；
- threshold 陣列；
- accepted count；
- error count；
- coverage；
- risk；
- 接受集為空時的未定義值；
- coverage 約束；
- validation 上的 $\tau^\star$；
- tie-breaking。

所以案例中的 Coverage 40%、Risk 10% 不能由程式或手算資料重建。

**最小修法：**

加入 `compute_risk_coverage(confidence, correctness, thresholds)`。若接受數為零，risk 應回傳 `np.nan` 或明確未定義標記。再加入 `select_threshold`，只輸入 validation 指標，從 coverage 達標且 risk 有定義的候選中取最小 risk。

---

### 4. 正常、邊界、故障測試仍只是自然語言

**原句：**

> 「輸入：見例題 4.1 和 4.2 的數據。」

> 「預期行為：拋出……」

**原因：**

章內沒有建立測試陣列、沒有呼叫函式、沒有 assertion、沒有 `try/except`。本卷要求「自足 CPU 程式、正常／邊界／故障測試」，測試規格不能代替測試程式。

目前至少缺少：

1. 正常 PPL；
2. 正常 ECE；
3. 全 False mask；
4. label $=V$；
5. NaN logits；
6. 空 ECE；
7. 非遞增 bins；
8. $p=0,0.5,1$ 的端點分箱；
9. classwise ECE；
10. 空接受集 risk；
11. risk 非單調；
12. ID/OOD 分群聚合。

**最小修法：**

補 `test_normal()`、`test_boundaries()`、`test_failures()`，用 `assert np.isclose` 及明確例外檢查；在 `if __name__ == "__main__":` 呼叫。沒有執行紀錄時只能稱為「預期 assertion 不觸發」。

---

### 5. ECE 命題標題和小結仍錯稱「正偏差」

**原句：**

> 「命題 3.2（有限樣本 ECE 的正偏差性）」

> 「有限樣本下存在正偏差」

但章首已正確寫成：

> 「期望不低於母體固定分箱 ECE，不保證嚴格大於。」

**原因：**

證明得到：

$$
E[\widehat{\mathrm{ECE}}]\ge\mathrm{ECE}_{\mathrm{true,binned}},
$$

只保證非負向上偏差，不保證嚴格正偏差。章首、命題標題及小結彼此不一致。

**最小修法：**

統一改為：

> 「有限樣本固定分箱 ECE 的非負向上偏差」

小結也改成「其期望不低於母體固定分箱 ECE」。

---

### 6. ECE 證明的空箱變數仍未完整定義

**原句：**

> 「定義空箱的貢獻為零；非空箱的貢獻為 $\frac{N_b}{N}|X_b|$。」

後面仍直接寫：

$$
E\left[\frac{N_b}{N}|X_b|\right].
$$

**原因：**

$X_b=\mathrm{Acc}_b-\bar p_b$ 在 $N_b=0$ 時沒有定義，因此乘積也尚未被定義。文字說明意圖清楚，但正式證明應先建立在所有樣本結果上都有值的隨機變數。

此外：

> 「從條件分布 $P(X\mid S\in I_b)$ 中抽出」

其中 $X$ 與已定義的 $X_b$ 容易混淆；真正需要的是 $(A,S)$ 的條件分布。

**最小修法：**

定義：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|\mathrm{Acc}_b-\bar p_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

然後證明 $E[Y_b]\ge q_b|\delta_b|$。把條件抽樣明寫為 $(A_i,S_i)\mid S_i\in I_b$。

---

### 7. 「ECE 僅作為相對指標」仍不精確

**原句：**

> 「ECE 僅作為相對指標，用於比較同條件下的不同模型」

**原因：**

ECE 也能描述單一模型的分箱校準差距，不只是一個相對指標。另一方面，即使資料與 bins 相同，不同模型的信心分布和箱占比不同，ECE 排序仍不一定穩健。

**最小修法：**

改為：

> 「ECE 是依賴資料、樣本量與分箱規則的描述性估計量，可在相同評估程序下輔助比較，但不能單獨作為絕對品質或安全保證。」

---

### 8. $\tau\to1$ 時 coverage 不一定趨近零

**原句：**

> 「$\tau\to1$：$C\to0$」

**原因：**

接受規則是 $s(x)\ge\tau$。若存在信心恰為 $1$ 的樣本，則：

$$
\lim_{\tau\uparrow1}C(\tau)=P(S=1),
$$

不一定為零。

**最小修法：**

改成：

> 「coverage 隨 $\tau$ 提高而不增；若不存在信心等於 1 的樣本，$\tau\to1$ 時 coverage 才趨近零。」

---

### 9. Selective Risk 公式可簡化以避免重複條件

**原句：**

$$
R(\tau)=
\frac{\sum_{i:s_i\ge\tau}\mathbb I(\hat y_i\ne y_i)}
{\sum_{i:s_i\ge\tau}\mathbb I(s_i\ge\tau)}.
$$

**原因：**

求和範圍已限制 $s_i\ge\tau$，所以分母每項的 indicator 必為一。這不是數值錯誤，但掩蓋了 accepted count，也不便對應程式。

**最小修法：**

先定義：

$$
a_i(\tau)=\mathbb I(s_i\ge\tau),
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

---

### 10. OOD 的 PPL 解讀仍推論過強

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

高 PPL 直接表示模型對該群真實 token 給出較低概率。原因也可能是詞彙、模板、序列長度、tokenizer、標註方式或合成規則差異，不能只由 PPL 鎖定為語義泛化失敗。

**最小修法：**

改成：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer 與標註規則。」

---

### 11. PPL、ECE、Risk 的分母仍未在表格中分開

**原句：**

表格列出：

> 「Total Tokens | 1000 | 1000」

並在同一表中列 ECE。

**原因：**

PPL 的分母是有效 token 數；ECE 的分母是校準事件數；Risk 的分母是接受決策數。三者不必相同。拒答段已承認決策數需另行報告，但 ECE event count 仍缺失。

**最小修法：**

表格分開列出：

- valid token count；
- calibration event count；
- decision count；
- accepted count at $\tau$。

---

### 12. `compute_classwise_ece` 的空 shape 尚未處理

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 $C=0$，`eces` 為空並產生 NaN；若 $N=0$，錯誤延遲到內層函式才出現。

**最小修法：**

加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並補對應故障測試。

---

### 13. `compute_global_ppl` 與 `compute_ece` 的 array-like 政策不一致

**原句：**

`compute_ece` 先做：

```python
probs = np.asarray(probs)
```

但 `compute_global_ppl` 直接使用：

```python
if logits.ndim != 3:
```

**原因：**

若傳入巢狀 list，ECE 可以處理，PPL 則會因沒有 `.ndim` 而發生非契約化錯誤。函式應一致地只接受 ndarray，或一致地轉換 array-like。

**最小修法：**

在 PPL 函式入口將 logits、labels、mask 轉為 `np.asarray`；或者 docstring 明確限制只能輸入 ndarray。轉換後仍須檢查 mask dtype，不能把任意整數 mask 靜默當布林。

---

### 14. 全 logits 有限的政策應寫進 docstring

**原句：**

```python
if not np.all(np.isfinite(logits)):
    raise ValueError(...)
```

**原因：**

這表示即使 `valid_mask=False`，padding 位置也不得含 NaN/Inf。此為合理的嚴格策略，但目前介面沒有明說。`np.where` 只控制 reduction，不會豁免無效位置的非有限值。

**最小修法：**

在 docstring 明列：

> 「整個 logits 張量，包括無效 token 位置，都必須有限。」

---

### 15. PPL overflow 以 `print` 回報不適合作為函式契約

**原句：**

```python
if not np.isfinite(perplexity):
    print(f"Warning: PPL overflow...")
```

**原因：**

底層數值函式直接輸出文字有副作用，而且 `np.exp` 可能先產生 runtime warning。Mean NLL 仍可能完全有效，應由 API 明確處理 PPL 不可表示的情況。

**最小修法：**

選一種策略：

- 回傳 `inf` 且不列印；
- 指數化前檢查範圍並拋出 `OverflowError`；
- 回傳結構化 overflow 狀態。

---

### 16. 整合題解答仍不完整

**原句：**

> 「繪製 R-Curve：x 軸為 Coverage，y 軸為 Risk。」

**原因：**

題目要求設計 ID/OOD 實驗，但答案沒有明確指定：

- validation 與 test；
- confidence、correctness；
- threshold 候選；
- 空接受集；
- coverage 約束；
- 非單調失敗；
- 如何固定 $\tau$；
- 目標 risk 不可達時如何報告。

「OOD 可能需要更高 $\tau$」也不是一般保證；若 OOD 信心排序失效，提高閾值可能無效。

**最小修法：**

補完整步驟，並明示若 validation 上不存在符合 risk/coverage 的候選，就報告不可達，而不是在 OOD-test 上重新調閾值。

---

### 17. 資料切分文字已正確，但缺少可驗證的故障測試

**原句：**

> 「先按來源文件與時間切分，再於各集合內建立窗口」

**原因：**

方向正確，但沒有資料欄位或 assertion 證明同一來源不跨 split，也沒有重疊窗口洩漏測試。

**最小修法：**

在合成資料中保存 `source_id/group/time/seed`，加入檢查：

- 每個 `source_id` 只能屬於一個 split；
- 建窗後 split 不變；
- validation/test 不參與詞表或 bins 擬合。

---

### 18. 引用核對狀態仍互相矛盾

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻稱：

> 「以上引用為標準學術參考」

**原因：**

待核對文獻不能被概括為已確認的標準依據。來源備註也明確說 N6 尚未逐條核對，N1 只取得摘要頁。

**最小修法：**

末註改為：

> 「N1 僅核對摘要；N6 及新增校準文獻尚待逐條核對，目前僅列作延伸閱讀，不作已查證證據。」

---

## 三、shape、broadcast、mask 與 reduction 核對

PPL 主程式的核心 shape 正確：

- logits：$(N,L,V)$；
- labels：$(N,L)$；
- valid mask：$(N,L)$；
- `batch_idx`：$(N,1)$；
- `time_idx`：$(1,L)$；
- `safe_labels`：$(N,L)$。

進階索引會 broadcast 成 $(N,L)$；log-softmax 沿最後的詞表軸；有效 token NLL 先求總和，再只除一次有效 token 數，均正確。`safe_labels` 也避免 padding label 的負索引。

`np.where(valid_mask, -correct_log_probs, 0.0)` 的 reduction 策略正確。但因為上游先檢查整個 logits 張量有限，mask 並不允許 padding 位置含 NaN；此政策須明說。

本章是評估章，不必重複完整 Transformer 模型或訓練 loop；但指定的合成 OOD 評估、分群 loop、R-Curve 和故障測試不能省略。

---

## 四、可保留內容

以下內容已正確：

1. token 加權 PPL；
2. 不平均樣本 PPL；
3. tokenizer 可比性限制；
4. 穩定 log-softmax；
5. shape、dtype、label 範圍檢查；
6. 空有效 token 拒絕；
7. 極小正 NLL 的精確／近似區分；
8. 不跨文件重新前向；
9. 固定分箱 ECE 的主要 Jensen 推導；
10. coverage 是回答率；
11. 空接受集 risk 未定義；
12. validation 選 $\tau$；
13. token confidence 不等於回答安全；
14. risk 不必單調；
15. OOD 分群算術一致；
16. 手設數值已明示不是執行結果；
17. 正文已達最低字數；
18. 沒有虛構訓練、設備或 benchmark。

---

## 五、結論

現稿的 PPL 數學、ECE 手算、shape、mask 與資料洩漏概念已大致正確。然而，章稿明定的 ID/OOD 分群 lab、Risk–Coverage 實作及可運行正常／邊界／故障測試仍缺失；假設表格不能代替自足程式。ECE 命題標題和小結仍誤稱嚴格正偏差，空箱隨機變數未完全形式化，$\tau\to1$ 的 coverage 敘述不普遍成立，classwise ECE 空 shape、overflow、評估分母和引用核對狀態也仍需修正。這些屬概率、程式自足性、資料洩漏與證據範圍問題，尚不能通過。

VERDICT: REVISE