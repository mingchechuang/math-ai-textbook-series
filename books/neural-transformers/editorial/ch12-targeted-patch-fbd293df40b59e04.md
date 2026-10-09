<<<PATCH 01>>>
<<<OLD>>>
    return {
        "status": "completed",
        "metrics": metrics,
        "W": best_W,
<<<NEW>>>
    # 常數基線的預測類別只由訓練標籤決定；平手選編號較小者。
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
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此程式只支援兩個檔案均成功寫入後的 epoch 邊界中止：`.npz` 與 JSON 分開覆寫，寫入期間中斷可能留下不完整或不同輪次的檔案，不保證能恢復。若需抵抗此類故障，應先寫入並核驗完整的版本化檔案，再原子更新指向該版本的單一索引。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
### 正常測試：checkpoint 讀回
<<<NEW>>>
### 正常測試：訓練集多數類別基線

程式僅依訓練標籤選出多數類別，平手時選編號較小者；對每筆樣本固定預測該類別。基線和模型均以沿樣本軸求平均的準確率評估，不預設模型必然較好。以下僅為未執行的預期核對：

```python
cfg = Config(epochs=3)
result = train(cfg, checkpoint_prefix="baseline_test")
parts, _, _ = prepare(cfg)
chosen = int(np.argmax(
    np.bincount(parts["train"]["y"], minlength=cfg.classes)
))
assert result["baseline_class"] == chosen
for name in ("train", "val", "test"):
    expected = float(np.mean(parts[name]["y"] == chosen))
    assert np.isclose(result["baseline_accuracy"][name], expected)
```

預期：可並列比較 `metrics[name]["accuracy"]` 與 `baseline_accuracy[name]`。驗證及測試標籤只用於各自評估，不參與基線類別的決定；測試結果只在選模完成後查看。

### 正常測試：checkpoint 讀回
<<<END>>>