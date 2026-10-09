<<<PATCH 01>>>
<<<OLD>>>
    metrics = {
        name: evaluate(
            parts[name]["Xz"], parts[name]["y"], best_W, best_b
        )
        for name in ("train", "val", "test")
    }
    return {
        "status": "completed",
        "metrics": metrics,
        "W": best_W,
<<<NEW>>>
    metrics = {
        name: evaluate(
            parts[name]["Xz"], parts[name]["y"], best_W, best_b
        )
        for name in ("train", "val", "test")
    }
    # 多數類別只依訓練標籤決定；平手選編號較小者。
    baseline_class = int(np.argmax(np.bincount(
        ytr, minlength=cfg.classes
    )))
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
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。

這仍不保證跨 NumPy 版本逐位一致。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此例只保證兩個檔案都成功寫入後的 epoch 邊界中止可續訓；分別覆寫 `.npz` 與 JSON 並非原子提交，寫入期間中斷可能留下不完整或不同輪次的檔案，不保證可恢復。

模型選定後，程式另以訓練標籤的多數類別建立常數基線，分別計算三集合沿樣本軸平均的準確率，可與 `metrics` 中的模型準確率比較。驗證與測試標籤不參與決定基線類別；測試結果仍只在選模完成後查看。不預設模型一定優於基線。若需要抵抗寫入期間的故障，應先寫入並核驗完整的版本化檔案，再原子更新指向該版本的單一索引；本例沒有實作這項保證。

這仍不保證跨 NumPy 版本逐位一致。
<<<END>>>