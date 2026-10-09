<<<PATCH 01>>>
<<<OLD>>>
    return {
        "status": "completed",
        "metrics": metrics,
        "W": best_W,
        "b": best_b,
<<<NEW>>>
    # 常數基線只依訓練標籤決定；平手選編號較小的類別。
    baseline_class = int(np.argmax(
        np.bincount(ytr, minlength=cfg.classes)
    ))
    baseline_accuracy = {
        name: float(np.mean(parts[name]["y"] == baseline_class))
        for name in ("train", "val", "test")
    }
    return {
        "status": "completed",
        "metrics": metrics,
        "baseline_class": baseline_class,
        "baseline_accuracy": baseline_accuracy,
        "W": best_W,
        "b": best_b,
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此實作只支援 `.npz` 與 JSON 兩檔均完整寫入後的 epoch 邊界中止；兩檔分別覆寫，寫入期間中斷可能留下不完整或不同輪次的檔案，不保證可恢復。
<<<END>>>