# 獨立審稿意見

## 一、先行重算

我未執行程式，以下均由公式與程式語義重算。

### 1. 例題 4.1

對 $z=[0,1,2,3]$：

$$
\operatorname{LSE}(z)
=3+\ln(e^{-3}+e^{-2}+e^{-1}+1)
\approx3.44019.
$$

第二個位置的精確 NLL 不是零，而是

$$
\ln(1+3e^{-110})>0.
$$

第三個位置則是

$$
\ln(1+3e^{-101})>0.
$$

兩者在目前小數精度下都可近似為零。因此：

$$
\mathrm{MeanNLL}
\approx\frac{3.44019+\ln4}{4}
\approx1.20662,
$$

$$
\mathrm{PPL}\approx3.3422.
$$

章稿最終 PPL 數值正確，惟中間把近似零寫成精確零。

### 2. 例題 4.2

低箱 $[0,0.5)$ 有三筆，平均信心 $0.7/3$、準確率 $0$；高箱 $[0.5,1]$ 有七筆，平均信心 $5/7$、準確率 $3/7$。故：

$$
\mathrm{ECE}
=\frac3{10}\frac{0.7}{3}
+\frac7{10}\frac27
=0.07+0.20=0.27.
$$

手算正確。

### 3. OOD 分群表

若常規組有 800 tokens、Mean NLL $1.5$，告警組有 200 tokens、Mean NLL $5.5$，則：

$$
\mathrm{MeanNLL}_{\mathrm{OOD}}
=\frac{800(1.5)+200(5.5)}{1000}
=2.3,
$$

$$
\mathrm{PPL}_{\mathrm{OOD}}=e^{2.3}\approx9.974.
$$

這與同節表格中的 Mean NLL $2.5$、PPL $12.182$ 不相容。

---

## 二、阻擋通過的問題

### 1. 同一 OOD 集的整體數值互相矛盾

**原句：**

> 「Mean NLL | 1.0 | 2.5」

> 「PPL | 2.718 | 12.182」

稍後又寫：

> 「整體 Mean NLL = $(800 \times 1.5 + 200 \times 5.5) / 1000 = 2.3$。」

> 「整體 PPL = $\exp(2.3) \approx 9.97$。」

**原因：**

兩個群組若完整分割表格中的同一個 1000-token OOD 集，整體 Mean NLL 只能是群組 Mean NLL 的 token 加權平均，即 $2.3$，不能同時是 $2.5$。對應 PPL 也不能同時為 $9.97$ 與 $12.182$。

**最小修法：**

將表格的 OOD Mean NLL／PPL 改為 $2.3$／$9.974$；或重新指定群組 NLL，使其加權結果為 $2.5$。若本意是兩個不同假設情境，須分開命名，不得都稱為同一 OOD 集。

---

### 2. 指定的 OOD lab 仍未完成

**原句：**

> 「資料生成：使用固定 Seed=42 生成合成日誌。」

**原因：**

程式中完全沒有 seed 42、隨機生成器、日誌生成規則、ID/OOD 標記或 group 欄位。正文提供的是手設表格，不是由章內程式可重現的合成評估。指定 lab 要求：

- 保留集與合成偏移集比較；
- 空有效 token 拒絕；
- 分群結果。

目前只有第二項在函式中落實；第一與第三僅有無法重現的假設數字。固定 seed 不能替代生成規則。

**最小修法：**

補一個自足 NumPy 合成資料函式，輸出至少包含：

- `split`：ID／OOD；
- `group`：routine／alert；
- `logits`：$(N,L,V)$；
- `labels`：$(N,L)$；
- `valid_mask`：$(N,L)$；
- 單步校準所需的 confidence 與 correctness。

再按 split 與 group 呼叫相同評估函式，回報總 NLL、有效 token 數、PPL、ECE、coverage 和 risk。沒有執行紀錄時只能標為預期輸出。

---

### 3. 假設數字被混寫成合成資料結果

**原句：**

> 「評估指標比較（假設性情境，非執行結果）」

後面又寫：

> 「本節所有數據均為合成資料。」

以及：

> 「資料生成：使用固定 Seed=42 生成合成日誌。」

**原因：**

「假設性情境」表示數字是手設示例；「使用 seed 生成」則表示數字可由明確生成程序得到。現稿沒有生成器，因此只能支持前者，不能支持後者。這雖未直接聲稱已執行，但仍虛構了不存在於章稿中的生成能力與可重現性。

**最小修法：**

在生成器補齊前，統一改成「手設假設數值，未由 seed 或程式生成」。若補齊生成器，則表格數值必須能由提供的資料與公式重算。

---

### 4. 選擇性拒答被無條件宣稱能降低風險

**原句：**

> 「此機制能有效降低『已回答樣本』中的風險」

但後文又寫：

> 「提高 $\tau$……導致 Risk 上升。」

**原因：**

後面的反例已正確說明 risk 不必隨 $\tau$ 單調。若高信心錯誤比低信心正確更多，提高閾值會使 risk 增加。因此前面的無條件承諾與後面的反例自相矛盾。

**最小修法：**

改成：

> 「若信心分數能有效排序正誤風險，拒答可能降低已回答樣本的 risk；是否成立必須由驗證集 R-Curve 檢查。」

---

### 5. OOD 段落沒有證據支持「提高閾值有效降低風險」

**原句：**

> 「在 OOD 集上，設定 $\tau=0.8$。Coverage: 40%，Risk: 10%。」

> 「這表明在 OOD 情境下，提高拒答閾值能有效降低錯誤答案的風險」

**原因：**

只給 $\tau=0.8$ 的一個點，沒有較低閾值下的 risk，不能推論「提高」閾值造成 risk 下降。Coverage 40% 與 Risk 10% 也沒有 confidence／correctness 資料可供重算，而且本節已承認是非執行的假設情境。

**最小修法：**

提供至少兩個閾值的可手算資料，或刪除因果結論，改為：

> 「此假設點只描述 $\tau=0.8$ 下的 coverage 與 risk；是否優於其他閾值須由完整 R-Curve 判定。」

---

### 6. Risk–Coverage 沒有任何程式實作

**原句：**

> 「繪製 $R(\tau)$ 對 $C(\tau)$ 的曲線。」

> 「$\tau^\star \in \arg\min\cdots$」

**原因：**

本章核心包括選擇性拒答，但自足程式只實作 PPL 與 ECE。沒有：

- confidence/correctness shape 驗證；
- threshold 掃描；
- accepted count；
- coverage；
- 空接受集回傳未定義 risk；
- 驗證集 constrained argmin；
- tie-breaking；
- ID/OOD R-Curve 比較。

整合題也只給概念文字，不是完整實驗解答。

**最小修法：**

新增純 NumPy `risk_coverage(confidence, correctness, thresholds)`，要求兩者 shape $(N,)$，對每個 $\tau$ 回傳 accepted count、coverage、risk；接受數為零時 risk 回傳 `np.nan`。另新增驗證集選閾值函式，僅從 coverage 達標且 risk 有定義的候選中選擇。

---

### 7. 正常、邊界與故障測試仍只有敘述，並非自足程式

**原句：**

> 「輸入：見例題 4.1 和 4.2 的數據。」

> 「預期行為：拋出……」

**原因：**

程式塊沒有建立例題陣列、沒有呼叫函式、沒有 assertion，也沒有實際構造故障輸入。依本卷契約，「自足 CPU 程式、正常／邊界／故障測試」不是只列測試計畫。

缺少的實際測試包括：

- 正常 PPL；
- 正常 ECE；
- 全 False mask；
- label $=V$；
- NaN logits；
- 空 ECE；
- 非遞增 bins；
- 機率恰為 $0$、$0.5$、$1$；
- OOD 分群聚合；
- 空接受集 risk；
- risk 非單調反例；
- classwise ECE。

**最小修法：**

在同一程式中加入 `test_normal()`、`test_boundaries()`、`test_failures()`，用 `assert np.isclose(...)` 及明確的 `try/except` 驗證預期錯誤。最後用 `if __name__ == "__main__":` 呼叫。正文仍須寫「預期」，不能聲稱已通過。

---

### 8. 有限 logits 的 NLL 被寫成精確零

**原句：**

> 「$\text{LSE}=10+\ln(1)=10$。」

> 「$\text{NLL}_2=10-10=0$。」

**原因：**

對有限 logits，softmax 中其他類別概率皆為正，因此正確類別概率嚴格小於一，NLL 嚴格大於零。此例精確值為：

$$
\ln(1+3e^{-110}).
$$

它只是在顯示精度下近似零。

**最小修法：**

將等號改為：

$$
\mathrm{NLL}_2=\ln(1+3e^{-110})\approx0.
$$

Total NLL 也改成「約 $3.4402$」。

---

### 9. ECE 證明不應引入未定義的 $0\cdot\infty$

**原句：**

> 「若定義 $0 \cdot \infty = 0$ 或明確跳過」

**原因：**

$X_b$ 在 $N_b=0$ 時是未定義，不是無限大；而 $0\cdot\infty$ 也不應在此被任意定義。證明不需要這個約定。

**最小修法：**

直接定義箱貢獻：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|X_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

然後只對 $n=1,\ldots,N$ 條件求和。另刪除前一行對未定義 $X_b$ 寫出的 $E[|X_b|]$。

---

### 10. 「正偏差」比證明結論更強

**原句：**

> 「有限樣本下存在正偏差。」

證明所得是：

$$
E[\widehat{\mathrm{ECE}}]\ge\mathrm{ECE}_{\mathrm{true,binned}}.
$$

**原因：**

這只保證非負的向上偏差，不保證嚴格大於。某些退化或無變異情況可取等號。

**最小修法：**

將章首、命題標題與小結統一改為「非負的向上偏差」或「期望不低於固定分箱母體 ECE」。

---

### 11. 「ECE 僅作為相對指標」表述過度簡化

**原句：**

> 「ECE 僅作為相對指標，用於比較同條件下的不同模型。」

**原因：**

ECE 是分箱依賴的描述性估計量，既可描述單一模型，也可輔助模型比較；但即使使用相同 bins，不同模型的 confidence 分布與箱占比也不同，排序未必穩健。它不能直接變成安全保證，但也不是只能作相對指標。

**最小修法：**

改成：

> 「ECE 可在相同資料、分箱和估計程序下作描述及輔助比較，但其值依賴樣本與分箱，不應單獨作為絕對品質或安全保證。」

---

### 12. `compute_classwise_ece` 未拒絕空類別軸或空樣本

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 `probs.shape == (N, 0)`，迴圈不執行，`np.mean([])` 產生 NaN；若 $N=0$，錯誤要等到內層 `compute_ece` 才出現。函式契約應在入口拒絕。

**最小修法：**

新增：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並加入兩個故障測試。

---

### 13. 全 logits 有限的政策未在介面說明

**原句：**

```python
if not np.all(np.isfinite(logits)):
    raise ValueError(...)
```

**原因：**

此程式要求 mask=False 的 padding 位置 logits 也必須有限。這是合理的嚴格策略，但應明確說明：loss mask 只控制 reduction，不會豁免 padding 位置的 NaN/Inf。否則讀者可能誤以為 `np.where` 可以讓無效位置包含非有限值；實際上它們早已被拒絕。

**最小修法：**

在 docstring 增加一句：「整個 logits 張量均須有限，包括無效 token 位置。」

---

### 14. PPL 的 OOD 原因解讀過強

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

較高 PPL 只直接表示模型給該組真實 token 較低概率。成因可能包括模板、詞彙、tokenizer、長度、標註或合成規則差異，不能只由 PPL 判定為語義異常泛化失敗。

**最小修法：**

改成：

> 「這與告警群上的分布匹配較差一致；原因仍須以控制 tokenizer、模板、長度與標註規則的實驗區分。」

---

### 15. 資料切分及洩漏契約未進入案例或程式

**原句：**

> 「ID 集：合成水溫 25°C……OOD 集：合成水溫 20°C……」

**原因：**

沒有 `source_id`、文件、時間或窗口建立順序。若同一文件的重疊窗口跨 ID/OOD 或 validation/test，評估會洩漏。也沒有說 bins、score 形式及 $\tau$ 是在哪個 split 選定。

**最小修法：**

明列並在資料結構中保留 `source_id/group/time/seed`；先按來源或時間切分，再建窗口。詞表只能由訓練集擬合；bin 規則、sequence score 與 $\tau$ 只能用驗證集選；test 只用於最終報告。

---

### 16. 參考來源的核對狀態仍不一致

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻稱：

> 「以上引用為標準學術參考」

**原因：**

待核對來源不能用「標準」一詞替代書目與主張核驗。依來源備註，N6 也只是尚未逐條核對的延伸入口。N1 僅核對摘要，而且不直接支持本章 ECE 偏差與 selective risk 的所有主張。

**最小修法：**

末註改為：「N1 僅核對摘要；N6 及新增校準文獻尚待逐條核對，目前僅作延伸閱讀，不作已查證依據。」

---

## 三、shape、broadcast、mask 與 reduction 核對

PPL 主函式目前主要 shape 正確：

- logits：$(N,L,V)$；
- labels：$(N,L)$；
- mask：$(N,L)$；
- `batch_idx`：$(N,1)$；
- `time_idx`：$(1,L)$；
- `safe_labels`：$(N,L)$。

三個索引 broadcast 為 $(N,L)$，結果也是 $(N,L)$。詞表 reduction 沿最後軸，token reduction 對有效位置求總和後只除一次，均正確。`safe_labels` 配合 `np.where` 也避免 padding label 的負索引問題。

但本章同時混用兩種評估單位：

- PPL 是有效 token 級；
- top-label ECE 是預測樣本級；
- 回答級 selective risk 則需要獨立的回答單位。

OOD 表中的 `Total Tokens=1000` 不能自動成為 ECE 或 risk 的分母。程式與表格必須分別報告 token 數、校準樣本數及回答數。

本章不必重複完整 Transformer 與訓練 loop，因為本章核心是評估；但宣稱完成 OOD 實驗時，至少必須提供完整的合成評估資料生成與評估 loop。

---

## 四、可保留內容

以下內容已正確，可保留：

1. token 加權 NLL 與 PPL 定義；
2. tokenizer 不同時 PPL 不宜直接比較；
3. 穩定 log-softmax 的最後軸 reduction；
4. labels／mask shape 與 dtype 驗證；
5. 空有效 token 拒絕；
6. 固定分箱 ECE 對 $q_b$、$E[N_b]=Nq_b$ 的主要推導；
7. coverage 是回答率，空接受集 risk 未定義；
8. 驗證集 constrained argmin 的閾值選擇；
9. token confidence 不等於回答級安全；
10. OOD 下 risk 未必隨 $\tau$ 單調；
11. classwise ECE 使用 $\mathbb I(y_i=c)$；
12. 正文已達 3000 字下限；
13. 沒有聲稱已執行程式、已訓練模型或取得硬體 benchmark。

---

## 五、結論

現稿的 PPL、ECE 手算及大部分核心函式已接近正確，但尚未滿足章稿驗收。最主要的阻擋項是：同一 OOD 集的數值互斥、沒有實作所要求的 ID/OOD 合成與分群 lab、沒有 Risk–Coverage 程式、測試仍只是文字計畫，以及把手設數值混稱為 seed 生成的合成資料。另有有限 logits 的精確零、ECE 證明中的 $0\cdot\infty$、無條件降低 risk 的承諾、空 classwise ECE 與引用核對狀態等問題。這些均屬數學、概率、程式自足性與證據範圍問題，不能以文風修飾替代。

VERDICT: REVISE