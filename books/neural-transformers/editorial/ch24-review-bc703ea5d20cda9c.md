# 獨立審稿意見

## 一、先行重算與整體判斷

本輪已修正上一版多項核心錯誤，包括 logits／log-probabilities 的混用、ECE 手算箱邊界、Risk–Coverage 定義，以及穩定 log-softmax 的方向。例題 4.1 與 4.2 的主要數值目前正確：

### 困惑度重算

第一個有效 token：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(e^{-3}+e^{-2}+e^{-1}+1)
\approx3.44019.
$$

均勻 logits 的 NLL：

$$
\operatorname{LSE}([0,0,0,0])=\ln4\approx1.38629.
$$

因此：

$$
\mathrm{MeanNLL}
=\frac{3.44019+1.38629}{4}
\approx1.20662,
$$

$$
\mathrm{PPL}=e^{1.20662}\approx3.3422.
$$

章稿的 `3.342` 可接受。

### ECE 重算

依 $[0,0.5)$ 與 $[0.5,1]$ 分箱：

- 低箱信心總和 $0.7$、數量 $3$、正確數 $0$；
- 高箱信心總和 $5.0$、數量 $7$、正確數 $3$。

所以：

$$
\mathrm{ECE}
=\frac3{10}\left|\frac03-\frac{0.7}3\right|
+\frac7{10}\left|\frac37-\frac57\right|
=0.07+0.20=0.27.
$$

此部分正確。

不過現稿仍有會使正常程式必然失敗的 dtype 判斷錯誤、未完成的 ECE 證明、錯誤的閾值選擇敘述、虛構或未標明為假設的 OOD 數值，以及未達最低正文長度等阻擋問題，尚不能通過。

---

## 二、必須修正的問題

### 1. `labels.dtype` 判斷永遠為真，正常輸入必然被拒絕

**原句：**

```python
if labels.dtype != np.int64 or labels.dtype != np.int32:
    raise ValueError("Labels must be integers.")
```

**原因：**

任何 dtype 都不可能同時等於 `np.int64` 與 `np.int32`。若為 `int64`，右側 `labels.dtype != np.int32` 為真；若為 `int32`，左側為真。因此整個 `or` 條件永遠為真，`compute_global_ppl` 對所有輸入都會拋出錯誤，正常測試不可能得到章稿宣稱的 PPL。

**最小修法：**

可寫成：

```python
if not np.issubdtype(labels.dtype, np.integer):
    raise ValueError("Labels must be integers.")
```

或者把 `or` 改為 `and`，但 `np.issubdtype` 更能涵蓋其他合法整數 dtype。

這是直接阻擋程式運作的重大錯誤。

---

### 2. shape 驗證沒有確認 `(N,L)` 與 logits 前兩軸一致

**原句：**

```python
if logits.ndim != 3 or labels.shape != valid_mask.shape:
    raise ValueError("Shape mismatch.")
N, L, V = logits.shape
```

**原因：**

这里只確認 `labels` 與 `valid_mask` 彼此同形，沒有確認兩者等於 `(N,L)`。例如 logits shape 是 `(2,3,4)`，labels 與 mask 都是 `(2,2)`，前置檢查會通過，直到進階索引才以難以理解的 broadcast/index 錯誤失敗。

另外沒有檢查：

- `labels.ndim == 2`；
- `valid_mask.dtype` 是否為布林；
- $V>0$；
- logits 是否為數值陣列。

**最小修法：**

在取得 `N,L,V` 後明確檢查：

```python
if labels.shape != (N, L) or valid_mask.shape != (N, L):
    raise ValueError(...)
if valid_mask.dtype != np.bool_:
    raise ValueError(...)
if V <= 0:
    raise ValueError(...)
```

---

### 3. ECE 函式仍缺少空輸入及輸入契約驗證

**原句：**

```python
N = len(probs)
...
ece += (n_b / N) * diff
```

**原因：**

若 `probs` 與 `correctness` 都為空，程式不一定觸發除零：每箱皆走 `n_b == 0` 分支，最後會靜默回傳 ECE `0.0`。空評估集的校準誤差未定義，不應被報成完美校準。

此外，函式沒有檢查：

- `probs`、`correctness` 是否為一維；
- `correctness` 是否只包含 $0$、$1$；
- `correctness` 是否含 NaN；
- `bin_edges` 是否一維、有限、嚴格遞增；
- 首尾是否覆蓋 $[0,1]$；
- 是否至少有兩個邊界；
- 所有樣本是否恰落入一個箱。

若 `probs` 是普通 Python list，表達式 `probs >= 0` 也不能按預期工作。

**最小修法：**

函式開頭先用 `np.asarray`，拒絕 $N=0$，驗證所有 shape、有限值、二元 correctness 和嚴格遞增的完整邊界。迴圈後核對各箱 `count` 總和等於 $N$。

---

### 4. ECE 正偏差命題的證明仍把隨機量當常數

**原句：**

> 「因此，
> $$E\left[\frac{N_b}{N}|X_b|\right]\geq\frac{N_b}{N}|\mu_b|$$」

> 「對所有箱求和：
> $$E[\mathrm{ECE}_{\mathrm{empirical}}]\geq\sum_b\frac{N_b}{N}|\mu_b|=\mathrm{ECE}_{\mathrm{true,binned}}$$」

**原因：**

左側已取期望，但右側仍保留隨機的 $N_b$，所以這不是合法的確定量不等式。母體固定分箱 ECE 應以箱機率

$$
q_b=P(S\in I_b)
$$

作權重，而不是樣本中的隨機比例 $N_b/N$。

應先條件於 $N_b=n>0$：

$$
E[|X_b|\mid N_b=n]
\ge
|E[X_b\mid N_b=n]|.
$$

對固定 bins 與 i.i.d. 樣本，須再證明

$$
E[X_b\mid N_b=n]
=
E[A-S\mid S\in I_b]
=:\delta_b.
$$

因此：

$$
E\left[\frac{N_b}{N}|X_b|\right]
\ge
E\left[\frac{N_b}{N}\right]|\delta_b|
=q_b|\delta_b|.
$$

最後求和才得到母體固定分箱 ECE。

章稿目前的「$\mu_b=E[X_b\mid N_b>0]$」也不足以直接替換每個 $E[X_b\mid N_b]$。證明後段又說「若 $N_b$ 與 $X_b$ 相關，不等式方向可能改變」，這會反過來削弱自己剛聲稱完成的命題。固定 bins 下本來就應透過條件期望處理這種依賴，而不是把它列成未解決限制。

**最小修法：**

依上述四步改寫證明，明確定義母體目標：

$$
\mathrm{ECE}_{\mathcal B}
=\sum_b q_b
\left|
E[A-S\mid S\in I_b]
\right|.
$$

自適應分箱另列為不適用條件即可。現稿雖比上一版進步，仍未滿足「完整證明」要求。

---

### 5. 閾值選擇目標寫成「最小 $\tau$」是錯的

**原句：**

> 「在驗證集上尋找滿足 $C(\tau)\ge C_{\text{target}}$ 的最小 $\tau$，以最小化 $R(\tau)$。」

**原因：**

滿足最低 coverage 約束的「最小 $\tau$」通常會接受更多樣本，甚至 $\tau=0$ 就滿足約束；它並不因此最小化 risk。即使信心排序良好，為降低 risk 往往會在仍符合 coverage 下提高 $\tau$。而一般情況下 empirical risk 也未必隨 $\tau$ 單調。

**最小修法：**

改成：

$$
\tau^\star
\in
\arg\min_{\tau:C_{\mathrm{val}}(\tau)\ge C_{\mathrm{target}}}
R_{\mathrm{val}}(\tau).
$$

若多個閾值同分，再事先規定 tie-breaking。不可稱為「最小 $\tau$」。

---

### 6. 單 token 的 $p_{\max}$ 仍被直接套用到語言模型整段回答

**原句：**

> 「選擇性拒答則是利用模型的不確定性（如 $1-p_{\max}$）作為閾值」

以及：

> 「定義模型對輸入 $x$ 的信心分數為 $s(x)=\max_vP(y=v|x)$。」

**原因：**

這適合單步分類或單一 next-token 決策，卻不能自動代表生成回答的事實正確性。語言模型每個位置都有不同的 token 分布；首 token 的 $p_{\max}$、整段平均 entropy、最小 token 信心與答案級正確機率不是同一概念。若本章用回答級 Risk，必須定義回答級 score 與回答正誤標籤。

**最小修法：**

明示本章 R-Curve 程式與公式先限定於「單步分類／下一 token 正確性」。若延伸到整段回答，只列出需另行定義並在驗證集校準的 sequence-level score，不把 token confidence 稱為安全或事實正確信心。

---

### 7. OOD 表格和拒答數字沒有生成器或推導，構成無執行證據的結果聲稱

**原句：**

> 「使用固定 Seed 生成合成日誌。」

> 「Total Tokens 1000、Mean NLL 1.0／2.5、ECE 0.05／0.20」

> 「發現『異常告警』群的 PPL 飆升至 50，而『常規操作』僅為 8。」

> 「Coverage: 40%，Risk: 10%。」

**原因：**

章內沒有 seed 值、合成資料生成函式、logits／labels／mask、group 比例、分群 token 數或 R-Curve 計算程式。這些數字無法由前面的程式重現，也沒有被標為純假設示例。「發現」尤其暗示已執行實驗，違反無執行紀錄只能寫預期的契約。

分群 PPL 也不能只給 50 與 8 而不給兩群 token 數；整體 PPL 是按 token 加權的平均 NLL 再指數化，不能由群組 PPL 做普通平均。

**最小修法：**

二選一：

1. 將全表明確標為「假設性情境，不是執行結果」，刪除「發現」；
2. 更符合 lab 要求的做法是補一個自足 NumPy 合成資料生成器，固定 seed，產生 ID/OOD、group、logits、labels、mask，並由同一函式計算整體及分群結果。輸出只能寫「預期」。

目前未完成指定的「保留集與合成偏移集比較、分群結果」實驗契約。

---

### 8. 測試章節只有敘述，沒有可運行的測試呼叫

**原句：**

> 「預期輸出：PPL 3.342、ECE 0.27」

> 「驗證：與手算結果一致。」

**原因：**

程式區沒有建立例題資料、沒有 `if __name__ == "__main__"`、沒有 assertion，也沒有呼叫邊界及故障案例。更重要的是，現有 dtype bug 會使所謂正常測試必然失敗，因此「與手算結果一致」不能成立。

**最小修法：**

補完整的正常、空 token、邊界概率、越界 label、NaN logits、非法 bins、空 ECE 資料測試呼叫，並寫成預期 assertion。文字改為「若實作正確，預期與手算一致」，不可寫成已驗證。

---

### 9. PPL 的「均勻猜測等價」仍需加條件

**原句：**

> 「$PPL=V$：模型表現等同於在整個詞表上均勻隨機猜測。」

**原因：**

均勻分布確實會得到 PPL $V$，但反命題不成立：某個非均勻模型也可能在特定評估集上具有平均 NLL $\ln V$，因而得到同樣 PPL。相同 PPL 也不表示準確率、概率分布或個別 token 行為相同。

**最小修法：**

改為：「若每一步都在 $V$ 個 token 上均勻分配概率，則 PPL 為 $V$；觀察到 PPL $=V$ 不足以證明模型逐步均勻猜測。」

---

### 10. `compute_stable_log_softmax` 的零和檢查沒有實際作用，政策也需一致

**原句：**

```python
if np.any(sum_exp == 0):
    raise ValueError(...)
```

**原因：**

對每列全為有限 logits 的輸入，減去最大值後至少有一項為零，故至少有一個 `exp(0)=1`；`sum_exp` 不可能為零。這不是錯誤，但屬不可達防線。另一方面，函式拒絕所有位置的非有限 logits，包括 mask 為 False 的 padding 位置；這可以是合理的嚴格政策，但此時後文所說用 `np.where` 防止無效位置 NaN 傳播並非真正適用，因為 NaN 早已被整體拒絕。

**最小修法：**

明示採「所有 logits，包括 padding 位置，皆必須有限」的嚴格契約，並刪除誤導說明；或只在有效位置計算，但那需要更細緻的介面。不可同時暗示兩種政策。

---

### 11. Classwise ECE 解答仍不是完整、健壯的程式答案

**原句：**

```python
N, C = probs.shape
...
macro_ece = np.mean(eces)
```

**原因：**

這段依賴外部已定義的 `compute_ece`，在章內同一程式脈絡尚可，但沒有驗證：

- `probs` 是否為 $(N,C)$；
- $C>0$、$N>0$；
- labels shape 是否為 $(N,)$；
- labels 是否為合法整數；
- 每列概率是否有限、非負、總和近似 $1$。

題目要求程式題完整解答，而本卷要求正常／邊界／故障測試；目前沒有 classwise ECE 的任何測試。

**最小修法：**

補輸入驗證及至少一個兩類手算對照、空資料拒絕、非法概率列、label 越界測試。

---

### 12. 「OOD 需要更高閾值」不是一般保證

**原句：**

> 「因此，為了達到相同的 Risk，OOD 集可能需要更高的 $\tau$。」

**原因：**

「可能」本身尚可，但前面的因果解釋容易使讀者理解為一般規律。如果 OOD 上的信心排序失效，錯誤樣本反而比正確樣本更高信心，提高 $\tau$ 可能不降低 risk，甚至升高 risk。選擇性拒答有效與否必須由驗證／校準資料上的 risk-coverage 關係支持。

**最小修法：**

補一個反例：高信心樣本全錯、低信心樣本較正確時，提高 $\tau$ 不會改善 risk。明示不能預設單調性。

---

### 13. 參考資料仍有未核對來源與過度概括聲明

**原句：**

> 「以上引用為標準學術參考」

同時 Guo、Niculescu-Mizil 標為「待核對」，N6 依來源備註也未逐條核對。

**原因：**

未核對的作者、題名與主張不能以「標準學術參考」概括背書。N1 與本章的 ECE 偏差、校準及 selective risk 命題也沒有直接建立來源關係。現稿沒有虛構已讀全文，但書目狀態仍不夠一致。

**最小修法：**

將註記改成：「N1 僅核對摘要；N6 與新增校準文獻尚待逐條核對，不以其作為本章已驗證證據。」未核對前不要賦予「原始提出」或標準定義來源地位。

---

### 14. 正文長度未達硬性下限

自動量測為 2729 字，而卷規明定正文 3000 字為下限，且公式、程式、英文、參考來源不充字數。即使技術錯誤全部修正，本項仍單獨構成退修理由。

**最小修法：**

不要以重複敘述補字數；應補足目前確實缺失的內容：

- 完整固定分箱 ECE 證明；
- ID/OOD 合成資料契約與分群計算；
- risk-coverage 的 NumPy 函式；
- 非單調 confidence 排序反例；
- 完整測試與 classwise ECE 解答；
- tokenizer、文件邊界及 token 計數對 PPL 可比性的限制。

---

## 三、資料洩漏與評估範圍

現稿正確保留了「測試集不能選 $\tau$」的原則，但還應明列：若 bin 數、bin 邊界、temperature scaling、拒答 score 形式或分群方式是根據資料選擇，它們也屬模型選擇，應在訓練／驗證範圍完成。ID 與 OOD 的文件或來源必須先切分，再建立重疊窗口；否則相同日誌片段可能同時進入不同集合。合成資料也應保留 seed、group、time 與生成規則，不能只寫「固定 Seed」而不給值。

此外，PPL、ECE 與回答正確性應在同一評估單位上說清楚：PPL 是 token 級；目前 ECE 示例是分類預測級；回答級 risk 又是另一單位。三者不能在沒有聚合規則時直接串接。

---

## 四、可保留的改進

本輪以下修正是正確的：

1. coverage 已改成回答比例；
2. 空接受集的 selective risk 已明確為未定義；
3. logits 與 NLL 公式已一致；
4. 例題 LSE 手算已修正；
5. ECE 的 $0.5$ 與 $1.0$ 邊界已統一；
6. padding label 使用 `safe_labels`，並改用 `np.where`；
7. 習題 3 已改成比較樣本平均 mean NLL，而非錯誤的雙重指數；
8. classwise ECE 的 outcome 已正確寫成 $\mathbb I(y_i=c)$；
9. 安全聲明已明確指出 PPL 不能證明故障識別或操作安全；
10. 沒有聲稱下載模型、使用 GPU 或執行外部工具。

這些進步不足以抵消上述程式必敗、證明不完整與虛構結果格式問題。

---

## 五、結論

本輪已解決上一版大部分概念混淆，但仍有四個主要阻擋項：第一，dtype 條件使核心 PPL 函式對所有 labels 必然失敗；第二，ECE 偏差命題仍未正確處理隨機箱計數；第三，閾值最佳化被錯寫成選「最小 $\tau$」；第四，OOD 與分群數值沒有生成程式或明確的假設標記，不能當成評估結果。另有空 ECE 資料靜默回傳零、測試未自足、回答級 confidence 未定義，以及正文不足 3000 字等問題。均屬正確性與章稿契約問題，並非風格偏好。

VERDICT: REVISE