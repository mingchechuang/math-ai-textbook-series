# 獨立審稿意見

## 一、重算結果

本次未執行任何程式，只依當前章稿手算與檢查程式語義。

### 1. PPL

第一個 token：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(1+e^{-1}+e^{-2}+e^{-3})
\approx3.44019.
$$

第二、三個 token 的 NLL 分別為：

$$
\ln(1+3e^{-110}),\qquad \ln(1+3e^{-101}),
$$

皆嚴格大於零，但在四位小數下可近似為零。因此：

$$
\overline{\mathrm{NLL}}\approx1.20662,
\qquad
\mathrm{PPL}\approx3.3422.
$$

本輪已正確區分精確正值與近似零。

### 2. ECE

$[0,0.5)$ 箱貢獻為：

$$
\frac3{10}\left|0-\frac{0.7}{3}\right|=0.07.
$$

$[0.5,1]$ 箱貢獻為：

$$
\frac7{10}\left|\frac37-\frac57\right|=0.20.
$$

總 ECE 為 $0.27$，正確。

### 3. OOD 分群

$$
\frac{800(1.5)+200(5.5)}{1000}=2.3,
\qquad
e^{2.3}\approx9.974.
$$

表格與分群算術目前一致。命題 3.1 也已正確澄清只是重新排列既有 loss 項，不是跨文件重跑模型。

---

## 二、仍然阻擋通過的問題

### 1. 指定 lab 仍未由程式實作

**原句：**

> 「必須比較保留集（ID）與合成偏移集（OOD）的分群指標」

實作部分卻只有：

```python
compute_stable_log_softmax
compute_global_ppl
compute_ece
```

**原因：**

目前沒有任何程式建立或評估：

- ID 與 OOD split；
- routine 與 alert group；
- group/time/source 等資料欄位；
- ID/OOD 各自的 logits、labels、valid mask；
- 分群有效 token 數；
- 分群 ECE 的樣本數；
- 分群或 split 評估 loop；
- Risk–Coverage 曲線。

正文已誠實改稱表格是「手設情境，非執行結果」，這避免了虛構執行；但也等於承認指定的合成偏移 lab 尚未實作。卷規要求自足 CPU 程式，不是只有手設結果表。

**最小修法：**

補一個小型、確定性的 NumPy 評估資料生成器或直接提供完整常數陣列。資料至少包含：

- `split`：ID／OOD；
- `group`：routine／alert；
- `logits`：$(N,L,V)$；
- `labels`：$(N,L)$；
- `valid_mask`：$(N,L)$；
- 單步分類的 confidence 與 correctness。

再提供按 split/group 聚合 PPL、ECE、coverage、risk 的 loop。沒有執行紀錄時，輸出只能標為預期。

---

### 2. 「本節所有數據均為合成資料」仍與「手設假設數值」不一致

**原句：**

> 「下表是手設假設數值，並非由 seed 或程式生成的日誌評估。」

後面又說：

> 「本節所有數據均為合成資料。」

**原因：**

手設的指標數字不是由合成資料評估得到的數據。它們最多是「虛構的示意數值」或「假設情境」。若稱為合成資料，通常表示存在可定位的合成樣本及生成規則，現稿沒有。

**最小修法：**

在補生成器之前，把後句改為：

> 「本節所有數值均為手設示意，不是真實資料、實測結果或已執行的合成實驗。」

若補齊生成器，才可稱為合成資料的預期評估。

---

### 3. Risk–Coverage 是核心主題，卻沒有程式

**原句：**

> 「實際應在驗證集檢查完整 Risk–Coverage 曲線」

**原因：**

章稿正確指出單一閾值點不能證明提高閾值降低 risk，但沒有提供執行該要求的函式。缺少：

- confidence/correctness shape 檢查；
- threshold shape 與有限值檢查；
- accepted count；
- coverage；
- error count；
- selective risk；
- 空接受集的未定義策略；
- coverage 約束下選 $\tau$；
- tie-breaking；
- ID/OOD 曲線比較。

這也使整合題的解答無法自足。

**最小修法：**

新增 NumPy `compute_risk_coverage(confidence, correctness, thresholds)`。對接受數為零的點令 risk 為 `np.nan`，並保留 accepted count。另新增只以 validation 指標選閾值的函式，test 函式不得參與選擇。

---

### 4. 測試仍只是文字規格，不是可運行測試

**原句：**

> 「輸入：見例題 4.1 和 4.2 的數據。」

> 「預期行為：拋出……」

**原因：**

程式沒有建立例題的 NumPy 陣列、沒有呼叫函式、沒有 assertion，也沒有捕獲預期例外。依卷規，「正常／邊界／故障測試」需要自足程式，不只是測試清單。

至少缺少：

- 正常 PPL 與 ECE assertion；
- 全 False mask；
- label 越界；
- NaN logits；
- 空 ECE；
- 非遞增 bins；
- $p=0,0.5,1$ 的端點分箱；
- classwise ECE；
- risk 空接受集；
- risk 非單調反例；
- ID/OOD 分群聚合。

**最小修法：**

加入 `test_normal()`、`test_boundaries()`、`test_failures()`，並在 `if __name__ == "__main__":` 呼叫。沒有執行紀錄時只說「預期 assertion 不觸發」。

---

### 5. ECE 命題標題及小結仍使用「正偏差」

**原句：**

> 「命題 3.2（有限樣本 ECE 的正偏差性）」

> 「有限樣本下存在正偏差」

但章首已正確改為：

> 「期望不低於母體固定分箱 ECE，不保證嚴格大於。」

**原因：**

證明得到的是：

$$
E[\widehat{\mathrm{ECE}}]\ge\mathrm{ECE}_{\mathrm{true,binned}},
$$

只能保證非負向上偏差，不保證嚴格為正。章首與命題標題、小結現在互相不一致。

**最小修法：**

統一改成：

> 「有限樣本固定分箱 ECE 的非負向上偏差」

小結也改為「期望不低於母體固定分箱 ECE」。

---

### 6. ECE 證明的空箱隨機變數仍未形式化

**原句：**

> 「定義空箱的貢獻為零；非空箱的貢獻為 $\frac{N_b}{N}|X_b|$。」

但後面仍寫：

$$
E\left[\frac{N_b}{N}|X_b|\right].
$$

**原因：**

$X_b$ 在 $N_b=0$ 時沒有定義，因而乘積本身也未被完整定義。文字說明意圖清楚，但完整證明應先定義一個在所有結果上都有值的箱貢獻隨機變數。

此外：

> 「從條件分布 $P(X\mid S\in I_b)$ 中抽出」

又使用 $X$ 指原始樣本，而前面 $X_b$ 已代表校準差，符號容易混淆。

**最小修法：**

定義：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|\mathrm{Acc}_b-\bar p_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

之後證明 $E[Y_b]\ge q_b|\delta_b|$。把條件分布寫成 $(A,S)\mid S\in I_b$，不要重用 $X$。

---

### 7. 「ECE 僅作為相對指標」仍不準確

**原句：**

> 「ECE 僅作為相對指標，用於比較同條件下的不同模型」

**原因：**

ECE 也可描述一個模型自身的分箱校準差距，不只是相對指標。另一方面，即使使用相同資料和 bins，不同模型的信心分布及箱占比不同，ECE 排序也未必穩定。因此「用於比較」只能是輔助用途。

**最小修法：**

改成：

> 「ECE 是依賴資料、樣本量與分箱規則的描述性估計量；可在相同評估程序下輔助比較，但不能單獨作為絕對品質或安全保證。」

---

### 8. $\tau\to1$ 時 coverage 不一定趨近零

**原句：**

> 「$\tau\to1$：$C\to0$」

**原因：**

接受條件是 $s(x)\ge\tau$。若存在 $s(x)=1$ 的樣本，$\tau$ 從下方趨近 1 時仍會接受它們。因此：

$$
\lim_{\tau\uparrow1}C(\tau)=P(S=1),
$$

未必為零。

**最小修法：**

改為：

> 「coverage 隨 $\tau$ 提高而不增；若不存在信心恰為 1 的樣本，$\tau\to1$ 時 coverage 才趨近零。」

---

### 9. Selective Risk 的分母寫法冗餘且易誤讀

**原句：**

$$
R(\tau)=
\frac{\sum_{i:s(x_i)\ge\tau}\mathbb I(\hat y_i\ne y_i)}
{\sum_{i:s(x_i)\ge\tau}\mathbb I(s(x_i)\ge\tau)}.
$$

**原因：**

求和索引已限制在 $s(x_i)\ge\tau$，分母中的 indicator 對每一項都等於一。公式數值沒錯，但容易掩蓋 accepted count 的定義。

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

這也直接對應程式 shape 與 reduction。

---

### 10. OOD 段落仍從 PPL 過度推論語義原因

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

較高 PPL 直接表示模型對該群真實 token 給出較低概率。成因也可能是模板、詞彙、長度、tokenizer、標註或合成規則不同，不能單由 PPL 判定為語義異常泛化失敗。

**最小修法：**

改成：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，須進一步控制詞彙、模板、長度、tokenizer 與標註規則。」

---

### 11. ECE 與 PPL 的分母仍未在表格中完全分開

**原句：**

表格只列：

> 「Total Tokens | 1000 | 1000」

同時列出 ECE。

**原因：**

PPL 的分母是有效 token 數；top-label ECE 的分母是校準預測事件數；selective risk 的分母是決策數或接受數。三者不一定相同。章稿在拒答段已提醒決策數另報，但 ECE 的樣本數仍未列出。

**最小修法：**

表格分別增加：

- `Valid token count`；
- `Calibration event count`；
- `Decision count`；
- `Accepted count at tau`。

不能用 token 數代表其他分母。

---

### 12. `compute_classwise_ece` 仍未處理 $N=0$ 或 $C=0$

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 shape 為 $(N,0)$，`eces` 為空，`np.mean([])` 產生 NaN；若 shape 為 $(0,C)$，會延遲到 `compute_ece` 才失敗。函式入口應明確定義非法 shape。

**最小修法：**

加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並提供兩個故障測試。

---

### 13. 核心 PPL 函式的輸入轉換與有限值政策應更明確

**原句：**

```python
if logits.ndim != 3:
```

**原因：**

`compute_ece` 會先 `np.asarray`，但 `compute_global_ppl` 不會。如果傳入合法的巢狀 Python list，會在 `.ndim` 失敗，而不是收到章稿描述的 shape 驗證。若函式契約只接受 `np.ndarray`，應明示；若要比照 ECE 接受 array-like，應先轉換。

另外 `compute_stable_log_softmax` 檢查整個 logits 張量有限，表示 padding 位置也不能含 NaN/Inf。這是合理策略，但 docstring 沒有說明。

**最小修法：**

函式入口統一 `np.asarray`，或明示只接受 ndarray；並在 docstring 寫明所有 logits，包括 mask=False 的位置，都必須有限。

---

### 14. PPL overflow 以 `print` 回報不適合核心函式

**原句：**

```python
if not np.isfinite(perplexity):
    print(f"Warning: PPL overflow...")
```

**原因：**

核心數值函式直接列印有不可控副作用，且 `np.exp` 可能已先發出 runtime warning。Mean NLL 仍可有效，因此應提供明確、可由呼叫端處理的政策。

**最小修法：**

可回傳 `inf` 而不列印，並在文件中說明；或於指數化前檢查 dtype 範圍並拋出結構化錯誤。不要在底層函式直接 `print`。

---

### 15. 整合題解答仍未構成完整實驗

**原句：**

> 「繪製 R-Curve：x 軸為 Coverage，y 軸為 Risk。」

**原因：**

題目要求設計實驗，但解答沒有指定：

- ID/OOD 的來源切分；
- validation/test 角色；
- confidence 與 correctness；
- threshold 候選集合；
- 空接受集；
- coverage 約束；
- 非單調失敗；
- 固定 $\tau$ 後的 test 報告。

它也沒有對應程式，所以不符合「完整解答」。

**最小修法：**

補上述步驟，尤其明示只在 validation 選 $\tau$，再把同一固定閾值套到 ID-test 與 OOD-test；若 OOD 曲線在目標 coverage 下無法達到風險要求，應報告不可達，而不是自行再用 OOD-test 調閾值。

---

### 16. 資料切分說明已改善，但生成契約仍不完整

**原句：**

> 「實際評估應先按來源文件與時間切分，再於各集合內建立窗口」

此方向正確，但尚未提供：

- source/group/time 的資料欄位；
- train/validation/ID-test/OOD-test 的生成規則；
- 同文件不可跨集合的檢查；
- 重疊窗口去洩漏測試。

**最小修法：**

在合成資料函式中保留 `source_id`、`group`、`time`、`seed`，並加入 assertion：任何 `source_id` 只能出現在一個 split；建窗後不得改變 split。

---

### 17. 引用狀態仍自相矛盾

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻說：

> 「以上引用為標準學術參考」

**原因：**

待核對來源不能被概括為已確認依據。提供的來源備註亦明示 N6 尚未逐條核對，N1 只取得摘要頁。

**最小修法：**

末註改為：

> 「N1 僅核對摘要；N6 及新增校準文獻尚待逐條核對，目前僅列作延伸閱讀，不作已查證證據。」

---

## 三、shape、broadcast、mask 與 reduction

目前 PPL 核心 shape 正確：

- logits：$(N,L,V)$；
- labels：$(N,L)$；
- mask：$(N,L)$；
- `batch_idx`：$(N,1)$；
- `time_idx`：$(1,L)$；
- `safe_labels`：$(N,L)$。

進階索引 broadcast 為 $(N,L)$；log-softmax 沿最後的 $V$ 軸；NLL 對有效 token 求和後只除一次有效 token 數，均正確。`safe_labels` 也避免 padding 的負索引問題。

`np.where(valid_mask, -correct_log_probs, 0.0)` 的 reduction 策略正確，但由於函式事先拒絕全張量中的非有限 logits，mask 不會豁免 padding 位置的 NaN。這項政策應在介面中明說。

本章是評估章，不必重複完整 Transformer 模型及訓練 loop；然而指定的 OOD 評估資料、分群 loop、Risk–Coverage 及故障測試不能省略。

---

## 四、可保留內容

以下內容已修正到可保留：

1. token 加權 PPL；
2. 不平均 per-sample PPL；
3. tokenizer 可比性限制；
4. 穩定 log-softmax；
5. shape、dtype、label 範圍與空 token 檢查；
6. PPL 手算中的極小正 NLL；
7. 命題 3.1 不跨文件重新前向；
8. 固定分箱 ECE 的主要 Jensen 推導；
9. coverage 為回答率；
10. 空接受集 risk 未定義；
11. validation 選 $\tau$；
12. token confidence 不等於回答安全；
13. risk 不必單調；
14. OOD 表格與分群算術一致；
15. 手設數值已明示不是執行結果；
16. 正文已達 3000 字下限；
17. 沒有虛構訓練、硬體或 benchmark。

---

## 五、結論

現稿的核心數學與 PPL 程式已大致正確，且已修正跨文件串接、極小 NLL、手設數據證據範圍等問題。但指定的 ID/OOD 分群 lab、Risk–Coverage 函式及可運行測試仍完全缺失；ECE 命題標題與小結仍錯稱嚴格正偏差；空箱證明尚未形式化；$\tau\to1$ 的 coverage 敘述不普遍成立；classwise ECE、overflow、評估分母及引用核對狀態也仍需修正。這些是章稿契約及概率／程式正確性問題，不能核准。

VERDICT: REVISE