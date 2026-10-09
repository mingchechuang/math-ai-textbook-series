<<<PATCH 21>>>
<<<OLD>>>
測試 3：checkpoint 恢復。預期 `恢復後 dropout 遮罩一致` 為 `True`，前提是保存與恢復的是同一個 `bit_generator.state` 物件內容。
<<<NEW>>>
測試 3：checkpoint 的 RNG 狀態快照。程式將抽樣前的狀態值深拷貝保存，再還原至另一個 RNG；預期兩條路徑的下一個 dropout 遮罩一致。這只檢查 RNG 續接，不驗證參數、優化器或資料游標的完整訓練恢復。
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
本章證明了有效 token 加權的梯度累積等價性（命題 21.1）：只要損失對全體有效 token 只除一次，微批次梯度就必須以 $n_k/N$ 加權累積。未加權平均只在各微批次有效數相等時才正確。我們也證明了 checkpoint 恢復的充分條件（命題 21.2）：以步驟、參數、優化器狀態、RNG 狀態、資料游標組成的完整狀態，在確定性且不讀取額外資訊的前提下可重建後續軌跡。實作上，梯度累積必須與 optimizer step 對齊，最好是先累加總和、最後只除一次；AdamW 的解耦權重衰減與全域梯度裁切都只能在每次 optimizer step 施加一次。最後，本章區分了「數學等價」與「逐位一致」：前者靠證明，後者依賴硬體、函式庫、執行緒與 dtype 的固定，不能由有限 seed 實驗推論。所有數值測試結果在本章均以「預期」陳述，未經實際執行驗證。
<<<NEW>>>
本章證明了有效 token 加權的梯度累積等價性（命題 21.1）：微批次平均梯度按 $n_k/N$ 加權，或先累加總梯度再除以有效 token 總數一次。有效數不等時，未加權平均一般不等價，但特定梯度可能偶然相等。命題 21.2 的恢復結論則以確定性計算及完整狀態為前提：除步驟、參數、優化器與排程、RNG、資料游標外，還須固定或可重建資料內容與切分、批次順序及模型設定；累積中途存檔另須保存中間梯度與有效數。本章程式只示範部分欄位的記憶體內快照，不是完整可恢復的訓練 checkpoint。梯度累積須與 optimizer step 對齊；AdamW 的解耦權重衰減及全域梯度裁切均在每次 optimizer step 施加一次。數學等價不保證跨環境逐位一致；所有數值測試仍屬未執行的預期。
<<<END>>>
<<<PATCH 23>>>
<<<OLD>>>
邊界測試採 $B=T=1$，預期仍得到 $(1,1,11)$，而不是因索引或 squeeze 消掉 batch、time 軸。另一項因果邊界是只改最後一個詞元，預期更早位置的 logits 不變；最後位置的 logits則不要求不變。這項測試能抓到某些未遮罩的未來資訊洩漏，卻不能單獨證明整個實作正確。
<<<NEW>>>
邊界測試採 $B=T=1$，預期仍得到 $(1,1,11)$，而不是因索引或 squeeze 消掉 batch、time 軸。因果測試逐一改動位置 $j=1,\ldots,T-1$ 的詞元，檢查所有位置 $i<j$ 的 logits 不變，並對改動後的序列再比較完整與 cached 前向；改動最後一項只是其中一例。這些數值測試能抓到部分未來資訊洩漏，不能單獨代替因果性與 cache 等價性的證明。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
2.  **程式題**：
    修改 `compute_ece` 以支持多類別 ECE（Classwise ECE）。
    輸入：`probs` (N, C), `labels` (N,)。
    輸出：每個類別的 ECE 及其宏平均值。
    注意：類別 $c$ 的正確性定義為 $\mathbb{I}(y_i = c)$，而非僅限於 argmax 為 $c$ 的樣本。
<<<NEW>>>
2.  **程式題**：
    正文的 top-label ECE 以 $p_{\max}$ 為信心、以 argmax 是否命中為事件。另實作 **one-vs-rest classwise ECE**：輸入 `probs` (N, C)、`labels` (N,)，對每個類別 $c$ 將 $p_{i,c}$ 與事件 $\mathbb{I}(y_i=c)$ 分箱比較，輸出各類 ECE 及宏平均。兩種 ECE 定義不同，不得直接當成同一指標比較。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    eces, macro = compute_classwise_ece(
        [[.8, .2], [.3, .7]], [0, 1], [0, .5, 1])
    assert len(eces) == 2 and np.isclose(macro, np.mean(eces))
<<<NEW>>>
    eces, macro = compute_classwise_ece(
        [[.8, .2], [.3, .7]], [0, 1], [0, .5, 1])
    # 類別0：|1-.8|/2 + |0-.3|/2 = .25；
    # 類別1：|0-.2|/2 + |1-.7|/2 = .25。
    # 故逐類 ECE 均為 .25，宏平均亦為 .25（非 top-label ECE 的定義）。
    if len(eces) != 2 or not np.allclose(eces, [.25, .25]):
        raise AssertionError("逐類 ECE 與手算不符")
    if not np.isclose(macro, .25):
        raise AssertionError("宏平均 ECE 與手算不符")
<<<END>>>