<<<PATCH 01>>>
<<<OLD>>>
def evaluate(X, y, W, b):
    loss, _, _ = loss_and_grad(X, y, W, b)
    pred = np.argmax(X @ W + b, axis=1)
    accuracy = np.mean(pred == y)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def checkpoint_paths(prefix):
<<<NEW>>>
def evaluate(X, y, W, b):
    loss, _, _ = loss_and_grad(X, y, W, b)
    pred = np.argmax(X @ W + b, axis=1)
    accuracy = np.mean(pred == y)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def majority_class(y, classes):
    """只從訓練標籤決定常數預測；平手時選較小類別。"""
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("訓練標籤必須是非空一維陣列")
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("訓練標籤超出類別範圍")
    return int(np.argmax(np.bincount(y, minlength=classes)))


def checkpoint_paths(prefix):
<<<END>>>

<<<PATCH 02>>>
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
<<<NEW>>>
    metrics = {
        name: evaluate(
            parts[name]["Xz"], parts[name]["y"], best_W, best_b
        )
        for name in ("train", "val", "test")
    }
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
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。

這仍不保證跨 NumPy 版本逐位一致。更嚴格的實驗還需記錄 Python、NumPy、作業系統、CPU、執行緒設定及程式版本。`.npz` 與 JSON 也應放在受控目錄並配合完整性雜湊；`allow_pickle=False` 不等於檔案一定可信。
<<<NEW>>>
`resume=True` 時，程式重建相同合成資料及切分，核對設定、群組和標準化統計，再恢復當前參數、最佳參數、早停計數與 RNG 狀態。因 checkpoint 在每個 epoch 結束後寫入，下一輪由 `epoch+1` 開始。此實作只支援 **兩個 checkpoint 檔案均完成寫入後** 的 epoch 邊界中止；`.npz` 與 JSON 分開覆寫，寫入期間若中斷，可能留下不一致或不完整的檔案，不保證可恢復。需要抵抗這類故障時，應先寫入新的暫存目錄並核驗兩個檔案，再以原子方式發布完整版本，而不是直接覆寫現有版本。

這仍不保證跨 NumPy 版本逐位一致。更嚴格的實驗還需記錄 Python、NumPy、作業系統、CPU、執行緒設定及程式版本。`.npz` 與 JSON 也應放在受控目錄並配合完整性雜湊；`allow_pickle=False` 不等於檔案一定可信。
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
預期：shape 成立，三集合群組互斥；訓練標準化後各特徵均值近零、母體標準差近一。驗證與測試資料不應被要求具有相同性質。

### 正常測試：checkpoint 讀回
<<<NEW>>>
預期：shape 成立，三集合群組互斥；訓練標準化後各特徵均值近零、母體標準差近一。驗證與測試資料不應被要求具有相同性質。

### 正常測試：訓練集多數類別基線

基線只根據訓練標籤選一個固定類別；若類別計數相同，選編號較小者。它對每筆樣本都預測該類別，以與模型相同的準確率定義分別評估三個集合。測試準確率只在選模完成後計算，不用來改選模型；基線也不保證低於模型。

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

預期：報告同時含模型與常數基線在各集合的準確率；不預設哪一方較高。驗證或測試標籤均不參與基線類別的決定。

### 正常測試：checkpoint 讀回
<<<END>>>