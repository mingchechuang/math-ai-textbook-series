<<<PATCH 26>>>
<<<OLD>>>
若約定硬標籤項也用同一個溫度 $T$，第二項改為 $\alpha\,(q_{soft}-y)/N$。兩種約定在梯度數值上差一個除以 $T$ 的因子；實作時必須明寫採用哪一種，並在測試中固定。中心差分核對做法與主程式相同，只把 `distill_loss_params` 換成含 $\alpha$ 的版本。
<<<NEW>>>
若約定硬標籤項也用同一個溫度 $T$，即該項為 $\alpha\,\mathrm{CE}(y,\mathrm{softmax}(z_s/T))$，則先對縮放後 logits $z_s/T$ 微分得 $q_{soft}-y$，再對原學生 logits 乘鏈式法則的 $1/T$，按 $N$ 筆樣本平均後第二項應為 $\alpha\,(q_{soft}-y)/(TN)$。兩種約定在梯度數值上差一個除以 $T$ 的因子；實作時必須明寫採用哪一種，並在測試中固定。中心差分核對做法與主程式相同，只把 `distill_loss_params` 換成含 $\alpha$ 的版本。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
`np.where(p > 0, p * (logp - logq), 0.0)` 這一行是數值安全點：當某個教師機率在浮點下下溢為嚴格 $0$ 時，$p_k\log p_k$ 的極限值是 $0$，但 `0.0 * -inf` 在 IEEE 754 下是 NaN。以 `np.where` 先判斷再乘，才能保證 `kl_per_sample` 不會出現 NaN。既然 `p>0` 條件下 `logp` 有限、`logq` 也有限（`softmax_temp` 已保證），乘積仍可正常計算。
<<<NEW>>>
`np.where(p > 0, p * (logp - logq), 0.0)` 這一行用來讓零機率項的輸出為零：當某個教師機率在浮點下下溢為嚴格 $0$ 時，$p_k\log p_k$ 的極限值是 $0$，但 `0.0 * -inf` 在 IEEE 754 下會是 NaN。須注意 NumPy 的 `np.where` **不是**「先判斷再乘」：它會先完整計算兩個分支的運算式，再按遮罩選值；因此 `p=0` 位置的乘積仍會被求值，但該位置的 NaN 會被選為 `0.0` 而不出現在結果中。若連中間運算都不得產生 NaN，須改以有效索引取值再相乘，而非只依賴 `np.where`。此外，`softmax_temp` 只保證原始 logits 與 $T$ 有限，`z/T` 仍可能溢位；在 $p>0$ 且 `logp`、`logq` 皆為有限的情況下，乘積才可正常計算。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
- [N1] Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 （2026-10-06 取得摘要頁，未完整閱讀）。
- [N2] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, https://arxiv.org/abs/2106.09685 （2026-10-06 取得摘要頁，未完整閱讀）。
- [N3] PyTorch 2.14 `scaled_dot_product_attention` API 文件, https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html （已取得 2.14 API 全文；非本機版本或執行證據）。
<<<NEW>>>
- [N1] Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 （背景入口；未完整閱讀論文）。
- [N2] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, https://arxiv.org/abs/2106.09685 （背景入口；未完整閱讀論文）。
- [N3] PyTorch 2.14 `scaled_dot_product_attention` API 文件, https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html （可定位網址；非本機版本或執行證據，未逐條核對）。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
2. **重疊文件視窗**：故意將同一文件的視窗分到不同集合時，manifest驗證預期拒絕。
<<<NEW>>>
2. **重複文件跨split的manifest拒絕**：在manifest中把同一文件複製一列並改標為另一split時，manifest驗證預期拒絕。此測試只證明「同一文件在manifest被標兩種split」會被攔截，不等於已測到「實際視窗沒有跨集合」。要直接檢驗視窗來源，須另加一項核對：比較train、val、test的`pairs`內`doc_id`集合與manifest所允許的split，並故意把一個train視窗插入test `pairs`，預期核對失敗。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
工具回傳文件id及字元span。拒絕請求會在工具呼叫前中止；已發生的唯讀查詢則記入事件。模型文字、檢索文本與工具呼叫提議均不是權限授予。
<<<NEW>>>
工具回傳文件id及字元span。本基線的`audited_answer`只按`doc_id`比對，尚未獨立驗證版本、區間與原文切片，因此其輸出只能稱為「本次lookup回傳的命中片段」，不是具版本約束的逐字引用。要成為可核驗引用，須在回傳前另按`doc_id`、版本、$0\leq start<end\leq|text|$及`text[start:end] == span`覆核，未通過就拒答。拒絕請求會在工具呼叫前中止；已發生的唯讀查詢則記入事件。模型文字、檢索文本與工具呼叫提議均不是權限授予。
<<<END>>>