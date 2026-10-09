<<<PATCH 09>>>
<<<OLD>>>
**故障測試設計**：另寫測試專用的錯誤梯度計算，把 `dZ1 = dH * (1 - self.H ** 2)` 故意改成 `dZ1 = dH`，不要修改正式模型。使用固定輸入與參數，使至少一個 hidden activation 明顯不接近零；將錯誤解析梯度與同一參數點的有限差分比較，並斷言至少一項尺度化誤差不小於容差。此測試預期能揭露漏掉 tanh 導數的故障；未執行前不宣稱斷言已通過，也不預先指定誤差必然「遠大於」某數值。
<<<NEW>>>
主程式已加入測試專用的錯誤梯度計算：刻意漏掉 tanh 導數，使用固定輸入與參數比較錯誤解析梯度及有限差分，並斷言尺度化誤差不小於容差。此為預期故障測試；未執行前不宣稱斷言已通過，也不預先指定誤差必然「遠大於」某數值。
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
    # Use different seeds for train and test to ensure they are distinct
<<<NEW>>>
    # Use separate random streams for independent synthetic draws
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
- **驗證**：使用歷史數據切分為 Train、Validation 和 Test。
<<<NEW>>>
- **切分原則**：若擴展到需要選擇設定的時間序列實驗，應先按時間切成 train、validation 與 test；本章程式只示範預先固定設定下的 train/test，未實作 validation。
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
5. **數據切分**：保留測試集以檢測過擬合，不使用測試集調參。
<<<NEW>>>
5. **數據切分**：本章預先固定設定，保留測試集作最終評估且不依測試結果調參；若要診斷過擬合或選擇設定，應另設 validation。
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
若 NumPy 計算結果與此接近（考慮浮點舍入），則實現正確。
<<<NEW>>>
若 NumPy 計算結果與此接近（考慮浮點捨入），則此案例未發現不一致；整體實作仍需由全部參數差分、方向導數、邊界及故障測試共同檢查。
<<<END>>>