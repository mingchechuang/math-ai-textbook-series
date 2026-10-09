<<<PATCH 09>>>
<<<OLD>>>
    def __init__(self, dim_in, dim_hid, seed=42):
        self.rng = np.random.default_rng(seed)
<<<NEW>>>
    def __init__(self, dim_in, dim_hid, seed=42):
        if any(isinstance(d, (bool, np.bool_)) or
               not isinstance(d, (int, np.integer)) or d <= 0
               for d in (dim_in, dim_hid)):
            raise ValueError("dim_in and dim_hid must be positive integers")
        self.rng = np.random.default_rng(seed)
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        
        return L
<<<NEW>>>
        old = (self.W1.copy(), self.b1.copy(),
               self.W2.copy(), self.b2.copy())
        with np.errstate(over="ignore", invalid="ignore"):
            updated = tuple(p - lr * g for p, g in zip(
                old, (self.dW1, self.db1, self.dW2, self.db2)))
        if not all(np.all(np.isfinite(p)) for p in updated):
            raise ValueError("Update would produce non-finite parameters")
        self.W1, self.b1, self.W2, self.b2 = updated
        return L
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    X_tiny, Y_tiny = X_train[:8], Y_train[:8]
    tiny.forward(X_tiny, Y_tiny)
    tiny_initial = tiny.loss()
    for _ in range(2000):
        tiny.train_one_step(X_tiny, Y_tiny, lr=0.1)
    tiny.forward(X_tiny, Y_tiny)
    assert tiny.loss() < tiny_initial * 0.5, "Tiny-set fit failed"
<<<NEW>>>
    X_tiny = np.zeros((8, dim_in), dtype=np.float64)
    X_tiny[:, 0] = np.array([-2.0, -1.5, -1.0, -0.5,
                              0.5, 1.0, 1.5, 2.0])
    Y_tiny = (X_tiny[:, [0]] > 0).astype(np.float64)
    tiny.forward(X_tiny, Y_tiny)
    tiny_initial = tiny.loss()
    for _ in range(2000):
        tiny.train_one_step(X_tiny, Y_tiny, lr=0.1)
    tiny.forward(X_tiny, Y_tiny)
    assert tiny.loss() < tiny_initial * 0.5, "Tiny-set loss did not fall"
    assert np.mean((tiny.Y_pred > 0.5) == Y_tiny) == 1.0, \
        "Tiny-set classification failed"

    # 測試專用錯誤反傳：刻意漏掉 tanh 導數，不修改正式 backward。
    faulty = TwoLayerNet(1, 1, seed=8)
    faulty.W1[:] = 1.0
    faulty.b1[:] = 0.0
    faulty.W2[:] = 1.0
    faulty.b2[:] = 0.0
    fault_X = np.array([[1.0]])
    fault_Y = np.array([[0.0]])
    faulty.forward(fault_X, fault_Y)
    dZ2 = faulty.Y_pred - fault_Y  # B=1
    bad_dW1 = (fault_X.T @ (dZ2 @ faulty.W2.T))[0, 0]
    numeric_dW1 = faulty.numerical_gradient("W1", (0, 0))
    fault_scaled_err = abs(bad_dW1 - numeric_dW1) / max(
        1.0, abs(bad_dW1), abs(numeric_dW1))
    assert fault_scaled_err >= 1e-5, "Faulty backward escaped detection"
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
4. **訓練測試**：在小型合成資料集上執行訓練，觀察損失下降，並嚴格執行訓練集、驗證集與保留測試集的切分，以檢測過擬合（Overfitting）。
<<<NEW>>>
4. **訓練測試**：用獨立小資料檢查訓練程式能否擬合，並將另一組合成資料留作最終測試。本章範例的訓練設定預先固定，未建立驗證集，也不根據測試結果調參；需要選擇設定或診斷泛化過擬合時，須另設驗證集。
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
3. **反例**：
   當 $Y \to 1$，$\sigma'(z) \to 0$。梯度 $dW_2 = H^T (Y-Y_{true})$。若 $Y_{true}=1$，則 $Y-Y_{true} \to 0$。故梯度趨近 0。
   這**不是**梯度檢查失敗，而是數學特性。梯度檢查會通過，因為解析梯度和數值梯度都趨近於 0。

4. **整合**：
   $db = \sum_{i=1}^{B} dZ_{i}$。
   若 $B=1$，則 $db$ 應等於 $dZ$ 的對應元素。
   測試：隨機 $X, W$。計算 `self.db1`。
   手動計算：`dZ1 = ...`; `sum_dZ = np.sum(dZ1, axis=0)`。
   斷言 `np.allclose(self.db1, sum_dZ)`。
<<<NEW>>>
3. **反例**：
   真標籤為 $y=1$ 且 $z\to+\infty$ 時，$\sigma(z)-1\to0$；這是 logits BCE 的解析導數，不必再乘一次 $\sigma'(z)$。但這**不保證**任意步長下的差分可靠：步長太小時，兩側損失的差可能因捨入而消失；步長太大時，差分不再準確代表該點的局部斜率。飽和區的梯度檢查亦可能對錯誤不敏感，應比較不同步長，並同時查看絕對誤差。

4. **整合**：
   取單筆資料，先執行正式前向及反向。因 $B=1$，輸出層的 $dZ_2=Y_{pred}-Y_{true}$；隱藏層的 $dZ_1=(dZ_2W_2^T)\odot(1-H^2)$。偏置的廣播反向沿 batch 軸求和，對單筆資料恰好取出唯一一列。完整測試如下；它也驗證求和後的一維 shape，避免僅因廣播而得到看似相同的數值。

   ```python
   net = TwoLayerNet(2, 3, seed=11)
   x = np.array([[1.0, -0.5]])
   y = np.array([[1.0]])
   net.forward(x, y)
   net.backward()
   dz2 = net.Y_pred - y
   dz1 = (dz2 @ net.W2.T) * (1.0 - net.H ** 2)
   assert net.db2.shape == net.b2.shape
   assert net.db1.shape == net.b1.shape
   assert np.allclose(net.db2, dz2[0])
   assert np.allclose(net.db1, dz1[0])
   ```
<<<END>>>