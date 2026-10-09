<<<PATCH 01>>>
<<<OLD>>>
        for name in ("train", "val", "test")
    }
    return {
        "status": "completed",
        "metrics": metrics,
<<<NEW>>>
        for name in ("train", "val", "test")
    }

    # 多數類別基線：只由訓練標籤決定，測試集不參與其選擇。
    counts = np.bincount(ytr, minlength=cfg.classes)
    majority_class = int(np.argmax(counts))
    baseline = {
        name: {
            "majority_class": majority_class,
            "accuracy": float(
                np.mean(parts[name]["y"] == majority_class)
            ),
        }
        for name in ("train", "val", "test")
    }

    return {
        "status": "completed",
        "baseline": baseline,
        "metrics": metrics,
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
def save_checkpoint(prefix, arrays, metadata):
    array_path, json_path = checkpoint_paths(prefix)
    np.savez(array_path, **arrays)
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
<<<NEW>>>
def save_checkpoint(prefix, arrays, metadata):
    """先寫暫存檔，兩份都完成後才原子替換正式檔。"""
    array_path, json_path = checkpoint_paths(prefix)
    tmp_array = array_path.with_name(array_path.name + ".tmp")
    tmp_json = json_path.with_name(json_path.name + ".tmp")

    np.savez(tmp_array, **arrays)
    tmp_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tmp_array.replace(array_path)
    tmp_json.replace(json_path)
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。

這仍不保證跨 NumPy 版本逐位一致。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。`save_checkpoint` 先寫入同目錄暫存檔，兩份都完成後才以原子替換更新 `.npz` 與 JSON；恢復點因此限於已完成整組寫入的 epoch 邊界，不涵蓋替換期間的程序強制終止。

這仍不保證跨 NumPy 版本逐位一致。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
### 正常測試：checkpoint 讀回

```python
prefix = "checkpoint_test"
<<<NEW>>>
### 正常測試：多數類別基線

```python
cfg = Config(epochs=3, patience=3)
result = train(cfg, checkpoint_prefix="baseline_test")

X, y, g, t = make_data(cfg)
parts = split_by_group(X, y, g, t)
counts = np.bincount(parts["train"]["y"], minlength=cfg.classes)
majority = int(np.argmax(counts))

for name in ("train", "val", "test"):
    expected = float(np.mean(parts[name]["y"] == majority))
    assert result["baseline"][name]["majority_class"] == majority
    assert abs(result["baseline"][name]["accuracy"] - expected) < 1e-12
```

預期：基線只由訓練標籤眾數決定；驗證集與測試集不參與其選擇，兩者以相同準確率定義與模型比較。

### 正常測試：checkpoint 讀回

```python
prefix = "checkpoint_test"
<<<END>>>