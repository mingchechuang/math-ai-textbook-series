<<<PATCH 01>>>
<<<OLD>>>
def checkpoint_paths(prefix):
<<<NEW>>>
def majority_class(y, classes):
    """只由訓練標籤決定常數預測；平手選較小類別。"""
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("訓練標籤必須是非空一維陣列")
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("訓練標籤超出類別範圍")
    return int(np.argmax(np.bincount(y, minlength=classes)))


def checkpoint_paths(prefix):
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    return {
        "status": "completed",
        "metrics": metrics,
        "W": best_W,
<<<NEW>>>
    baseline_class = majority_class(ytr, cfg.classes)
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
<<<PATCH 03>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此實作只支援兩個檔案均完成寫入後的 epoch 邊界中止；`.npz` 與 JSON 分開覆寫，若寫入期間中斷，檔案可能不完整或彼此不一致，不保證可恢復。若需抵抗這類故障，應將完整版本寫入暫存位置並核驗，再以原子方式發布完整版本。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
### 正常測試：checkpoint 讀回
<<<NEW>>>
### 正常測試：訓練集多數類別基線

只用訓練標籤選出預測類別；若計數相同，選編號較小者。對三集合均以與模型相同的準確率定義評估，但測試結果只在選模完成後查看，不用於改選模型。

```python
cfg = Config(epochs=3)
result = train(cfg, checkpoint_prefix="baseline_test")
parts, _, _ = prepare(cfg)
chosen = majority_class(parts["train"]["y"], cfg.classes)
assert result["baseline_class"] == chosen
for name in ("train", "val", "test"):
    expected = np.mean(parts[name]["y"] == chosen)
    assert np.isclose(result["baseline_accuracy"][name], expected)
```

預期：可將各集合的模型準確率與基線準確率並列；不預設模型必然較高。本測試未在寫作時執行。

### 正常測試：checkpoint 讀回
<<<END>>>