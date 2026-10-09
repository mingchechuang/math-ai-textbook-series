<<<PATCH 01>>>
<<<OLD>>>
6. 保存並恢復 epoch 邊界 checkpoint；
7. 在選模完成後才評估測試集。
<<<NEW>>>
6. 保存並恢復 epoch 邊界 checkpoint；
7. 在選模完成後才評估測試集；
8. 以訓練標籤計算多數類別基線，並與模型在驗證及測試集上以同一準確率定義比較。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    # 僅在訓練與選模完成後評估測試集。
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
        "b": best_b,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
    }
<<<NEW>>>
    # 僅在訓練與選模完成後評估測試集。
    metrics = {
        name: evaluate(
            parts[name]["Xz"], parts[name]["y"], best_W, best_b
        )
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
        "metrics": metrics,
        "baseline": baseline,
        "W": best_W,
        "b": best_b,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
    }
<<<END>>>
<<<PATCH 03>>>
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
    """先寫暫存檔，兩份都完成後再原子替換正式檔。"""
    array_path, json_path = checkpoint_paths(prefix)
    tmp_array = array_path.with_name(array_path.name + ".tmp")
    tmp_json = json_path.with_name(json_path.name + ".tmp")

    with open(tmp_array, "wb") as f:
        np.savez(f, **arrays)
    tmp_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 兩份暫存檔皆已完整寫入，才替換正式狀態。
    tmp_array.replace(array_path)
    tmp_json.replace(json_path)
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
這仍不保證跨 NumPy 版本逐位一致。更嚴格的實驗還需記錄 Python、NumPy、作業系統、CPU、執行緒設定及程式版本。`.npz` 與 JSON 也應放在受控目錄並配合完整性雜湊；`allow_pickle=False` 不等於檔案一定可信。
<<<NEW>>>
`save_checkpoint` 先寫入同目錄的暫存檔，兩份都完成後才以原子替換更新 `.npz` 與 JSON；因此恢復點是「已完成整組寫入的 epoch 邊界」，不涵蓋替換期間的程序強制終止。這仍不保證跨 NumPy 版本逐位一致。更嚴格的實驗還需記錄 Python、NumPy、作業系統、CPU、執行緒設定及程式版本。`.npz` 與 JSON 也應放在受控目錄並配合完整性雜湊；`allow_pickle=False` 不等於檔案一定可信。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
### 正常測試：checkpoint 讀回
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

預期：基線只由訓練標籤的眾數決定；驗證集與測試集不參與其選擇，兩者以相同準確率定義與模型比較。

### 正常測試：checkpoint 讀回
<<<END>>>