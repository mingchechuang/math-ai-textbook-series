# 獨立審稿意見

## 一、先行重算

本次只核對當前章稿，未執行程式。

### 1. PPL 手算

對第一個 token 的 logits $[0,1,2,3]$：

$$
\operatorname{LSE}
=3+\ln(e^{-3}+e^{-2}+e^{-1}+1)
\approx3.44019.
$$

第二個 token 的精確 NLL 為：

$$
\ln(1+3e^{-110})>0,
$$

第三個 token 的精確 NLL 為：

$$
\ln(1+3e^{-101})>0.
$$

兩者在四位小數下可近似為零。因此：

$$
\overline{\mathrm{NLL}}
\approx\frac{3.44019+\ln4}{4}
\approx1.20662,
$$

$$
\mathrm{PPL}\approx e^{1.20662}\approx3.3422.
$$

章稿最終 PPL 正確，但把有限 logits 對應的極小正 NLL 寫成精確零。

### 2. ECE 手算

$[0,0.5)$ 箱有三筆，平均信心為 $0.7/3$，準確率為零，其 ECE 貢獻為 $0.07$。$[0.5,1]$ 箱有七筆，平均信心 $5/7$，準確率 $3/7$，貢獻為 $0.2$。總和：

$$
\mathrm{ECE}=0.27.
$$

此處正確。

### 3. OOD 分群

$$
\frac{800(1.5)+200(5.5)}{1000}=2.3,
$$

$$
e^{2.3}\approx9.974.
$$

目前表格與分群算術一致。

---

## 二、阻擋通過的問題

### 1. OOD lab 仍是手設表格，不是自足實驗

**原句：**

> 「資料生成：使用固定 Seed=42 生成合成日誌。」

**原因：**

章內程式沒有 seed 42、亂數生成器、日誌生成函式、ID/OOD split、group、time 或 source ID。表格雖標示「假設性情境，非執行結果」，卻又聲稱使用 seed 生成，兩種證據層級不一致。

本章指定 lab 要求「保留集與合成偏移集比較、空有效 token 拒絕、分群結果」。目前只有空 token 拒絕進入程式；ID/OOD 與分群只存在於無法重現的表格。

**最小修法：**

二選一：

1. 刪除 seed 生成聲稱，將所有數字明確稱為「手設假設數值」；
2. 補完整 NumPy 生成器，輸出 split、group、time、logits、labels、valid mask、confidence 與 correctness，再由同一評估程式產生整體及分群指標。

若補生成器，輸出仍只能稱為預期結果，不能稱已執行。

---

### 2. 沒有 Risk–Coverage 實作

**原句：**

> 「繪製 $R(\tau)$ 對 $C(\tau)$ 的曲線。」

> 「$\tau^\star\in\arg\min_{\tau:C_{\mathrm{val}}(\tau)\ge C_{\mathrm{target}}}R_{\mathrm{val}}(\tau)$」

**原因：**

本章核心包含選擇性拒答，但程式只有 PPL 與 ECE。沒有 threshold 掃描、coverage、accepted count、selective risk、空接受集處理或驗證集閾值選擇。後文的 Coverage 40% 與 Risk 10% 因此無法重算。

**最小修法：**

補一個函式，輸入 shape $(N,)$ 的 confidence、二元 correctness 與 thresholds，逐閾值輸出：

- accepted count；
- coverage；
- error count；
- risk。

接受數為零時 risk 應回傳 `np.nan` 或明確的未定義標記，不得填零。另補驗證集選 $\tau$ 的函式，排除 coverage 不達標及 risk 未定義的候選。

---

### 3. 一個閾值點不能證明提高閾值降低風險

**原句：**

> 「在 OOD 集上，設定 $\tau=0.8$。Coverage: 40%，Risk: 10%。」

> 「這表明在 OOD 情境下，提高拒答閾值能有效降低錯誤答案的風險」

**原因：**

章稿沒有提供較低閾值的 risk，因而不能判定「提高」造成下降。章稿前面還正確給出 risk 可能隨 $\tau$ 上升的反例，與此結論衝突。

**最小修法：**

改成：

> 「此假設點只描述 $\tau=0.8$ 下的 coverage 與 risk；不能單獨證明提高閾值會降低 risk，須檢查完整 R-Curve。」

或提供至少兩個可由具體 confidence/correctness 手算的閾值點。

---

### 4. 正常、邊界與故障測試仍未寫成程式

**原句：**

> 「輸入：見例題 4.1 和 4.2 的數據。」

> 「預期行為：拋出……」

**原因：**

這些是測試計畫，不是自足測試。程式沒有建立例題陣列、呼叫函式、assert 數值或捕獲預期例外。本卷要求自足 CPU 程式及正常／邊界／故障測試，不是只要求文字描述。

缺少的實際測試至少包括：

- 正常 PPL；
- 正常 ECE；
- 全 False mask；
- label $=V$；
- NaN logits；
- 空 ECE；
- 非遞增 bins；
- $p=0,0.5,1$ 的分箱；
- classwise ECE；
- risk 空接受集；
- risk 非單調反例；
- ID/OOD 分群聚合。

**最小修法：**

在程式末尾加入測試函式及 `if __name__ == "__main__":`，使用 `assert np.isclose` 和明確的預期例外檢查。文字只能寫「預期 assertion 不觸發」，不可宣稱已通過。

---

### 5. 有限 logits 的 NLL 仍誤寫為精確零

**原句：**

> 「$\text{LSE}=10+\ln(1)=10$。」

> 「$\text{NLL}_2=10-10=0$。」

**原因：**

前一步只是 $\sum e^{z-m}\approx1$，不能在下一步改成精確等號。有限 logits 下其他類別概率皆為正，所以真類別概率嚴格小於一，NLL 嚴格大於零。

**最小修法：**

改成：

$$
\mathrm{LSE}=10+\ln(1+3e^{-110}),
$$

$$
\mathrm{NLL}_2=\ln(1+3e^{-110})\approx0.
$$

Total NLL 也應使用約等號。

---

### 6. 命題 3.1 的「串接長序列」可能改變條件分布

**原句：**

> 「若將所有樣本的有效 token 串接為一個長序列，並忽略樣本邊界」

**原因：**

如果真的把 token 序列串接後重新前向，後一樣本會以先前文件為上下文，NLL 通常改變，也會造成文件邊界洩漏。證明實際上只是在串接已經於原始上下文中計算出的 loss 項。

**最小修法：**

改成：

> 「將各樣本在原有文件邊界及上下文下已計算出的有效 token NLL 串接成一個損失列表。」

並明示不重新串接 token、不跨文件建立 attention。

---

### 7. 「正偏差」比證明所得更強

**原句：**

> 「有限樣本下存在正偏差。」

> 「有限樣本 ECE 的正偏差性」

但證明只得到：

$$
E[\widehat{\mathrm{ECE}}]\ge\mathrm{ECE}_{\mathrm{true,binned}}.
$$

**原因：**

這是非負向上偏差，不保證嚴格為正；某些退化分布可以取等號。

**最小修法：**

章首、命題標題及小結統一改為「非負向上偏差」或「期望不低於固定分箱母體 ECE」。若要主張嚴格正偏差，須補嚴格 Jensen 的成立條件。

---

### 8. ECE 證明仍有符號及空箱形式問題

**原句：**

> 「樣本 $i\in D_b$ 是從條件分布 $P(X\mid S\in I_b)$ 中抽出的。」

**原因：**

前文已用 $X_b$ 表示箱內差，這裡的 $X$ 又像輸入變數，符號含混。需要的其實是 $(A,S)$ 的條件分布。

另外章稿說空箱貢獻為零，但仍寫：

$$
E\left[\frac{N_b}{N}|X_b|\right],
$$

而 $X_b$ 在空箱未定義。直覺可理解，形式上仍應定義完整隨機變數。

**最小修法：**

定義：

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|\mathrm{Acc}_b-\bar p_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

之後證明 $E[Y_b]\ge q_b|\delta_b|$。將條件抽樣寫成 $(A_i,S_i)\mid S_i\in I_b$。

---

### 9. ECE 被過度簡化為「僅作相對指標」

**原句：**

> 「ECE 僅作為相對指標，用於比較同條件下的不同模型」

**原因：**

ECE 也可描述單一模型的分箱校準差距；但即使 bins 相同，不同模型的信心分布與箱占比不同，模型排序也未必穩健。因此它既不只是相對指標，也不能被當成可靠排名或安全保證。

**最小修法：**

改為：

> 「ECE 是依賴資料、樣本量與分箱規則的描述性估計量，可在相同評估程序下輔助比較，但不可單獨作為絕對品質或安全保證。」

---

### 10. $\tau\to1$ 時 coverage 不一定趨近零

**原句：**

> 「$\tau\to1$：$C\to0$」

**原因：**

接受規則是 $s(x)\ge\tau$。若部分樣本有 $s(x)=1$，當 $\tau$ 從下方趨近 1 時仍會接受這些樣本，因此 coverage 趨近於 $P(S=1)$，未必是零。

**最小修法：**

改成：

> 「coverage 隨 $\tau$ 提高而不增；若沒有信心恰為 1 的樣本，$\tau\to1$ 時 coverage 才趨近零。」

---

### 11. OOD PPL 的語義結論過強

**原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

較高 PPL 只直接表示模型對該群真實 token 指派較低概率。原因也可能是詞彙、模板、序列長度、tokenizer、標註或生成規則差異。不能只由 PPL 鎖定為語義泛化失敗。

**最小修法：**

改為：

> 「這表示模型與告警群的 token 分布匹配較差；原因須再控制詞彙、模板、長度、tokenizer 與標註規則。」

---

### 12. `compute_classwise_ece` 未拒絕空樣本或空類別

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

若 shape 為 $(N,0)$，`eces` 為空，`np.mean([])` 得到 NaN；若為 $(0,C)$，錯誤延遲到內層函式才出現。

**最小修法：**

加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

並加入對應故障測試。

---

### 13. logits 的全張量有限政策未寫進契約

**原句：**

```python
if not np.all(np.isfinite(logits)):
    raise ValueError(...)
```

**原因：**

這代表即使 `valid_mask=False`，padding 位置也不能含 NaN/Inf。這是合理的嚴格策略，但 docstring 沒有說明。`np.where` 只控制 reduction，並不豁免無效位置的非有限值。

**最小修法：**

在 docstring 明寫：「所有 logits，包括 mask=False 的位置，皆須有限。」

---

### 14. PPL overflow 用 `print` 不適合作為核心 API

**原句：**

```python
if not np.isfinite(perplexity):
    print(f"Warning: PPL overflow...")
```

**原因：**

底層數值函式直接輸出文字有副作用，且 `np.exp` 可能已先發出 runtime warning。Mean NLL 仍可能是有效值，應由介面明確決定如何處理 PPL overflow。

**最小修法：**

選擇並記錄一種策略：

- 回傳 `inf` 並不列印；
- 或在指數化前拒絕超出 dtype 可表示範圍；
- 或回傳結構化的 overflow 標記。

---

### 15. 整合題解答不完整

**原句：**

> 「繪製 R-Curve：x 軸為 Coverage，y 軸為 Risk。」

**原因：**

題目要求設計實驗，但解答沒有資料切分、confidence/correctness、threshold 候選、空接受集、coverage 約束、非單調失敗情況或最終 test 報告程序。這不符合「完整解答」。

**最小修法：**

補逐步實驗契約，並與新增的 Risk–Coverage 程式一一對應。尤其必須說明 $\tau$ 在 validation 選定，ID/OOD test 只套用固定 $\tau$ 報告。

---

### 16. 資料洩漏防線未落實到案例

**原句：**

> 「ID 集：合成水溫 25°C……OOD 集：合成水溫 20°C……」

**原因：**

沒有說明是否先按文件、來源或時間切分，再建立窗口。若同一原始日誌的重疊窗口跨集合，會造成洩漏。也未說明 tokenizer、bins、confidence 規則及 $\tau$ 的擬合範圍。

**最小修法：**

案例中明列：

1. 先按 `source_id/time` 切 train、validation、ID-test、OOD-test；
2. 再各自建立窗口；
3. tokenizer 只由 train 擬合；
4. bins、score 與 $\tau$ 只由 validation 決定；
5. test 不參與任何調整。

---

### 17. 引用核對狀態仍不一致

**原句：**

> 「Guo et al.……（待核對）」

> 「Niculescu-Mizil & Caruana……（待核對）」

末註卻說：

> 「以上引用為標準學術參考」

**原因：**

待核對來源不能以「標準」概括成已確認依據。來源備註也明說 N6 尚未逐條核對，N1 僅取得摘要頁。

**最小修法：**

末註改為：

> 「N1 僅核對摘要；N6 與新增校準文獻尚待逐條核對，目前僅列作延伸閱讀，不作已查證證據。」

---

## 三、shape、broadcast、reduction 與 mask 核對

PPL 主程式的主要 shape 正確：

- logits：$(N,L,V)$；
- labels：$(N,L)$；
- valid mask：$(N,L)$；
- `batch_idx`：$(N,1)$；
- `time_idx`：$(1,L)$；
- `safe_labels`：$(N,L)$。

三個索引會 broadcast 成 $(N,L)$；log-softmax 沿最後的詞表軸 reduction；所有有效 token 的 NLL 先求和，再只除一次有效 token 數，這些均正確。`safe_labels` 也避免 padding label 的負索引問題。

但報表須分開三種分母：

- PPL 的有效 token 數；
- ECE 的預測事件數；
- selective risk 的決策／回答數。

目前 OOD 表只列 token 數，無法支撐 ECE 與 risk 的樣本規模。

本章是評估章，不必重複完整 Transformer 或訓練 loop；但既然聲稱完成 OOD 評估 lab，就必須提供自足的合成評估資料、分群 loop 與故障測試。

---

## 四、可保留內容

以下內容可保留：

1. token 加權 PPL 定義；
2. 不平均 per-sample PPL；
3. tokenizer 不同時不可直接比較 PPL；
4. 穩定 log-softmax；
5. labels、mask、shape 與有效 label 檢查；
6. 空有效 token 拒絕；
7. 固定分箱 ECE 的主要 Jensen 推導；
8. coverage 定義為回答比例；
9. 空接受集 risk 未定義；
10. $\tau$ 只在驗證集選擇；
11. token confidence 不等於回答安全；
12. risk 不必隨 $\tau$ 單調；
13. OOD 分群算術目前一致；
14. 正文已達 3000 字下限；
15. 章稿明示程式未執行，沒有虛構訓練、硬體或 benchmark。

---

## 五、結論

現稿的 PPL、ECE 手算與核心 shape 已大致正確，固定分箱 ECE 證明也接近完整。然而，本章指定的 OOD lab、Risk–Coverage 程式和可運行測試仍缺失；seed 生成聲稱沒有對應生成器；單一閾值點被用來推論提高閾值降低 risk；有限 logits 的 NLL 仍被寫成精確零；命題 3.1 的串接措辭可能導致跨文件上下文洩漏。另有 ECE「正偏差」措辭、空 classwise shape、資料切分與引用核對狀態等問題。這些均屬數學、概率、資料洩漏、自足程式及證據範圍問題，仍需修訂。

VERDICT: REVISE