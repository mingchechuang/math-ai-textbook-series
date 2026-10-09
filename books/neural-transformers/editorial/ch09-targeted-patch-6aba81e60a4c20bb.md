<<<PATCH 09>>>
<<<OLD>>>
        max_abs_err = 0.0
        max_rel_err = 0.0
        total_checks = 0
        
        for name, (W, dW) in params.items():
            it = np.nditer(W, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                num_grad = self.numerical_gradient(name, idx, h)
                ana_grad = dW[idx]
                
                abs_err = abs(num_grad - ana_grad)
                # Scaled error: the denominator floor keeps near-zero gradients stable.
                denom = max(1.0, abs(num_grad), abs(ana_grad))
                scaled_err = abs_err / denom
                
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                total_checks += 1
                it.iternext()
                
        status = "PASS" if max_rel_err < tol else "FAIL"
        print(f"Gradient Check: {status}, Max Abs Err: {max_abs_err:.2e}, Max Rel Err: {max_rel_err:.2e} (Checks: {total_checks})")
        return max_rel_err
<<<NEW>>>
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        max_abs_err = 0.0
        max_scaled_err = 0.0
        total_checks = 0

        for name, (W, dW) in params.items():
            it = np.nditer(W, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                num_grad = self.numerical_gradient(name, idx, h)
                ana_grad = dW[idx]
                if not np.isfinite(ana_grad):
                    raise ValueError(f"Non-finite analytic gradient: {name}{idx}")

                abs_err = abs(num_grad - ana_grad)
                # 分母下限為 1；小梯度時此指標等於絕對誤差。
                scaled_err = abs_err / max(1.0, abs(num_grad), abs(ana_grad))
                max_abs_err = max(max_abs_err, abs_err)
                max_scaled_err = max(max_scaled_err, scaled_err)
                total_checks += 1
                it.iternext()

        status = "PASS" if max_scaled_err < tol else "FAIL"
        print(f"Gradient Check: {status}, Max Abs Err: {max_abs_err:.2e}, Max Scaled Err: {max_scaled_err:.2e} (Checks: {total_checks})")
        return max_scaled_err
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
            abs_err = abs(analytic_dd - numeric_dd)
            scaled_err = abs_err / max(1.0, abs(analytic_dd), abs(numeric_dd))
<<<NEW>>>
            if not np.isfinite(analytic_dd) or not np.isfinite(numeric_dd):
                raise ValueError("Non-finite directional derivative")
            abs_err = abs(analytic_dd - numeric_dd)
            scaled_err = abs_err / max(1.0, abs(analytic_dd), abs(numeric_dd))
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    def train_one_step(self, X, Y_true, lr=0.01):
        self.forward(X, Y_true)
<<<NEW>>>
    def train_one_step(self, X, Y_true, lr=0.01):
        if not np.isscalar(lr) or not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.forward(X, Y_true)
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    # Overfitting Check: Expect train loss to drop significantly and acc to be high
    # Test loss should be reasonable, but might be higher than train if overfitting
    if final_train_loss > initial_train_loss * 0.1:
        print("Warning: Model may not be fitting training data well.")
    if test_acc < 0.5:
        print("Warning: Model performance on test set is poor.")
<<<NEW>>>
    # 獨立小資料除錯：只取訓練資料；門檻預先固定，失敗即中止。
    tiny = TwoLayerNet(dim_in, dim_hid, seed=7)
    X_tiny, Y_tiny = X_train[:8], Y_train[:8]
    tiny.forward(X_tiny, Y_tiny)
    tiny_initial = tiny.loss()
    for _ in range(2000):
        tiny.train_one_step(X_tiny, Y_tiny, lr=0.1)
    tiny.forward(X_tiny, Y_tiny)
    assert tiny.loss() < tiny_initial * 0.5, "Tiny-set fit failed"

    # 邊界、參數往返及輸入故障測試；不讀取測試集作模型選擇。
    theta = model.pack_params().copy()
    model.unpack_params(theta)
    assert np.array_equal(theta, model.pack_params())
    for bad in (theta[:-1], np.r_[theta, 0.0],
                np.r_[theta[:-1], np.nan]):
        try:
            model.unpack_params(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid parameter vector accepted")
    for bad_X, bad_Y in (
        (X_train[:2], Y_train[:2, 0]),
        (X_train[:0], Y_train[:0]),
        (X_train[:1], np.array([[0.5]])),
        (np.full((1, dim_in), np.nan), Y_train[:1]),
        (np.zeros((1, dim_in + 1)), Y_train[:1]),
    ):
        try:
            model.forward(bad_X, bad_Y)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    model.forward(X_train[:1], Y_train[:1])
    assert model.Y_pred.shape == (1, 1)
    edge = TwoLayerNet(1, 1, seed=3)
    edge.W1[:] = 0.0
    edge.W2[:] = 0.0
    for z, y, expected_grad in ((100.0, 0.0, 1.0),
                                (-100.0, 1.0, -1.0)):
        edge.b2[:] = z
        edge.forward(np.zeros((1, 1)), np.array([[y]]))
        assert np.isfinite(edge.loss()) and np.isclose(edge.loss(), 100.0)
        edge.backward()
        assert np.isclose(edge.db2[0], expected_grad)
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
**預期**：返回的 `max_rel_err` 應小於 $10^{-5}$。程式應輸出 `PASS`。方向導數檢查也應通過。
<<<NEW>>>
**預期**：返回的最大尺度化誤差應小於 $10^{-5}$，程式應輸出 `PASS`；方向導數檢查亦預期通過。這些是待執行的預期，不是實測結果。
<<<END>>>