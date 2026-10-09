<<<PATCH 19>>>
<<<OLD>>>
    try:
        _ = TinyDecoder(V, D, H, L, max_len=0)
        assert False
    except ValueError:
        pass
<<<NEW>>>
    try:
        _ = TinyDecoder(V, D, H, L, max_len=0)
    except ValueError:
        invalid_max_len_rejected = True
    else:
        invalid_max_len_rejected = False
    assert invalid_max_len_rejected, "max_len=0 was accepted"
<<<END>>>
<<<PATCH 23>>>
<<<OLD>>>
    # 故障：cache 長度與 start 不一致；跨 batch 重用亦拒絕
    try:
        model.step(ids[:, 2:3], cache, start=2)
    except ValueError:
        pass
    else:
        raise AssertionError("cache length/start mismatch was accepted")
    try:
        model.step(other[:, :1], cache, start=5)
    except ValueError:
        pass
    else:
        raise AssertionError("batch shape mismatch was accepted")
<<<NEW>>>
    # 故障：cache 長度與 start 不一致；跨 batch 重用亦拒絕
    try:
        model.step(ids[:, 2:3], cache, start=2)
    except ValueError:
        length_mismatch_rejected = True
    else:
        length_mismatch_rejected = False
    if not length_mismatch_rejected:
        raise AssertionError("cache length/start mismatch was accepted")
    try:
        model.step(other[:, :1], cache, start=5)
    except ValueError:
        batch_mismatch_rejected = True
    else:
        batch_mismatch_rejected = False
    if not batch_mismatch_rejected:
        raise AssertionError("batch shape mismatch was accepted")
<<<END>>>
<<<PATCH 23>>>
<<<OLD>>>
此程式中的 `pass` 只表示**測試已預期捕獲例外**，不是未完成的函式實作。`masked_attention` 的有限值檢查針對允許位置；被遮罩位置被明確置為 $-\infty$，不應把這個刻意使用的值誤判為輸入故障。
<<<NEW>>>
故障測試以布林旗標記錄是否捕獲預期例外；未拒絕非法 cache 時拋出 `AssertionError`。`masked_attention` 的有限值檢查針對允許位置；被遮罩位置被明確置為 $-\infty$，不應把這個刻意使用的值誤判為輸入故障。
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
測試 9：checkpoint 只存權重、不存 RNG 狀態。預期恢復後的第一個 dropout 遮罩與未中斷訓練不同，`np.array_equal` 回傳 `False`。這是命題 21.2 前提被違反的最小示範。
<<<NEW>>>
測試 9：若 checkpoint 只存權重、不存 RNG 狀態，便無法保證恢復後沿用未中斷訓練的抽樣軌跡；第一個 dropout 遮罩可能不同，也可能偶然相同。應核對保存與恢復的 RNG 狀態值，並比較後續抽樣序列；不能以單次遮罩相等或不等判定完整訓練是否可重建。本章程式未實作此故障測試。
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
checkpoint 在此的作用是讓長訓練能跨工作階段延續，並在事後稽核時能重建證據鏈：保存步驟、權重、優化器狀態、RNG 狀態、資料游標。有了這五項，才能主張「這次訓練的第 $t$ 步之後是可重建的」。
<<<NEW>>>
checkpoint 在此的作用是讓長訓練能跨工作階段延續，並在事後稽核時保留重建證據。步驟、權重、優化器狀態、RNG 狀態與資料游標只是所需狀態的一部分；還須固定或可重建資料內容、切分、批次順序、生成規則及模型與優化器設定，並滿足命題 21.2 的確定性條件，才能主張第 $t$ 步之後的軌跡可重建。
<<<END>>>