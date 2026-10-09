<<<PATCH 01>>>
<<<OLD>>>
        "metrics": metrics,
        "W": best_W,
<<<NEW>>>
        "metrics": metrics,
        "baseline_class": int(np.argmax(
            np.bincount(ytr, minlength=cfg.classes)
        )),
        "baseline_accuracy": {
            name: float(np.mean(parts[name]["y"] == np.argmax(
                np.bincount(ytr, minlength=cfg.classes)
            )))
            for name in ("train", "val", "test")
        },
        "W": best_W,
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
本章使用沒有動量的 SGD，因此沒有額外的動量張量。checkpoint 固定在完整 epoch 結束後寫入，恢復點也限定在 epoch 邊界；因此不必保存批次內游標。
<<<NEW>>>
本章使用沒有動量的 SGD，因此沒有額外的動量張量。checkpoint 固定在完整 epoch 結束後寫入，恢復點也限定在 epoch 邊界；因此不必保存批次內游標。本例的兩個檔案分別覆寫，只保證兩檔均完整寫入後的中止可按上述邊界恢復；若在寫入期間中斷，檔案可能不完整或分屬不同輪次，不保證可恢復。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
預期：測試先建立 checkpoint，再從同一前綴載入，不依賴其他測試留下的檔案。
<<<NEW>>>
預期：測試先建立 checkpoint，再從同一前綴載入，不依賴其他測試留下的檔案。

另須核對選模完成後的常數基線：預測類別由訓練標籤計數的最大值決定，平手時取較小編號；對驗證與測試集合各自計算預測正確的樣本比例，與 `metrics` 中相同集合的準確率比較。預期 `baseline_class` 等於訓練標籤的多數類別，`baseline_accuracy[name]` 等於該集合中屬於此類別的比例；不預設模型一定較好。驗證與測試標籤只用於各自評估，不決定基線類別，測試結果也不參與選模。此核對尚未執行。
<<<END>>>