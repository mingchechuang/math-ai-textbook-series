<<<PATCH 01>>>
<<<OLD>>>
    return {
        "status": "completed",
        "metrics": metrics,
        "W": best_W,
<<<NEW>>>
    # 基線類別只由訓練標籤決定；計數相同時選編號較小者。
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
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此實作只支援兩個檔案均完整寫入後的 epoch 邊界中止；`.npz` 與 JSON 分別覆寫，寫入期間中斷可能留下不完整或不同輪次的檔案，不保證可恢復。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
預期：測試先建立 checkpoint，再從同一前綴載入，不依賴其他測試留下的檔案。
<<<NEW>>>
預期：測試先建立 checkpoint，再從同一前綴載入，不依賴其他測試留下的檔案；它不涵蓋寫入期間中斷。

基線也可用以下未執行測試核對：

```python
cfg = Config(epochs=3)
result = train(cfg, checkpoint_prefix="baseline_test")
parts, _, _ = prepare(cfg)
chosen = int(np.argmax(np.bincount(
    parts["train"]["y"], minlength=cfg.classes
)))
assert result["baseline_class"] == chosen
for name in ("train", "val", "test"):
    expected = float(np.mean(parts[name]["y"] == chosen))
    assert np.isclose(result["baseline_accuracy"][name], expected)
```

預期：模型與基線的準確率均沿各集合的樣本軸平均，可逐集合比較；不預設模型必然較好。驗證與測試標籤只用於各自評估，不參與決定基線類別；測試結果不參與選模。
<<<END>>>