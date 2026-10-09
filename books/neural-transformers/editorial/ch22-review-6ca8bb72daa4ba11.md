## 審稿結論

本輪已補足正文、裸 `assert` 及 top-p 同分／組合篩選測試等前次問題。作者明確聲明未執行程式，沒有虛構執行紀錄、來源查證或模型能力。仍有一處實質錯誤須修正：

1. **範例模型的最後 EOS 列不符合程式預期，會多生成一個 EOS。**

   原句：「在查表模型中，greedy 的預期序列是：`<BOS> 水溫 正常 <EOS>`」

   查表模型中，`正常` 列的最大 logit 指向 EOS；但 `generate` 在新生成 EOS 後停止，預期序列確實是 `[BOS, 1, 2, EOS]`。不過，另一處宣稱 prompt 末端已有 EOS 時會立即停止，並由測試涵蓋；此例也符合。重新核對後，未發現這裡有錯。

2. **零溫度政策與 `greedy=False` 的實際抽樣語義不一致。**

   原句：「`$T_{\\mathrm{temp}}=0$：公式未定義；本章 API 明確轉為 greedy one-hot。`」

   `next_token_distribution` 在零溫度回傳 one-hot，非 greedy 路徑再交給 `rng.choice` 抽樣；這在結果上必然選中唯一非零項，分布語義等價於 greedy。生成器若 `greedy=True` 則直接 `argmax`，也同樣一致。這不是實質錯誤，不應阻擋。

3. **習題解答中的遮罩 helper 未驗證 `allowed` 的數值型別，可能讓非布林輸入以非預期方式被拒絕或接受。**

   原句：「`allowed = np.asarray(allowed)`」及「`if allowed.dtype != np.bool_ or allowed.shape != logits.shape:`」

   以 `np.asarray` 後嚴格比較 dtype，整數 0/1 向量會被拒絕，這是合理且明確的布林契約；object 型別也會拒絕。此處沒有正確性缺陷，僅需在文字說明 `allowed` 必須是 NumPy 布林向量。可選的最小修法：補上該契約說明，不必阻擋。

4. **全文唯一應阻擋的問題：數學命題對「所有有限 logits」的敘述忽略浮點運算溢位。**

   原句：「若 $z_i,z_j\\in\\mathbb{R}$ 且 $T_{\\mathrm{temp}}>0$，則……$p_i(T_{\\mathrm{temp}})>p_j(T_{\\mathrm{temp}})$」

   在實數數學中證明正確；但程式實作以 float64 計算 `logits / temperature`。有限 logits 除以極小正溫度可能溢位成無限值，`stable_softmax` 會拒絕；因此「有限 logits、正溫度」不保證本 API 能產生可比較的機率。文中穩定 softmax只處理減最大值，不能修復除法階段的溢位。

   最小修法：將命題明確限定為實數運算，並在實作契約中說明「若縮放 logits 發生非有限值，拒絕輸入」；或先以等價且穩定的縮放方式避免中間溢位，並補上極小溫度故障測試。不要把實數命題改成浮點保證。

字數欄為 4538，但其中是否排除了程式、公式及英文，仍不能單憑欄位確認；本輪正文已明顯補長。程式中的排序政策、重新正規化、EOS 與 seed 驗證、測試入口及預期結果聲明整體自洽。建議修正浮點溢位的契約後通過。

VERDICT: REVISE