<<<PATCH 09>>>
<<<OLD>>>
1. **手算**：給定 $B=2$、$X=I_2$、$W_1=I_2$、$b_1=(0,0)$、$W_2=(0.5,0.5)^T$、$b_2=0$、$Y_{true}=(1,0)^T$，並令 $H=\\tanh(X)$。計算 $Z_2$、$dZ_2$、$dW_2$ 和 $db_2$。
2. **程式**：修改程式碼，使用 ReLU 替代 tanh，並調整梯度檢查邏輯。
3. **反例**：對真標籤 $y=1$，當 logit $z\\to+\\infty$ 時，BCE 對 logit 的梯度 $\\sigma(z)-y$ 趨近何值？解析梯度趨近零，是否足以保證有限差分檢查在任意步長下都可靠？說明理由。
<<<NEW>>>
1. **手算**：給定 $B=2$、$X=I_2$、$W_1=I_2$、$b_1=(0,0)$、$W_2=(0.5,0.5)^T$、$b_2=0$、$Y_{true}=(1,0)^T$，並令 $H=\tanh(X)$。計算 $Z_2$、$dZ_2$、$dW_2$ 和 $db_2$。
2. **程式**：修改程式碼，使用 ReLU 替代 tanh，並調整梯度檢查邏輯。
3. **反例**：對真標籤 $y=1$，當 logit $z\to+\infty$ 時，BCE 對 logit 的梯度 $\sigma(z)-y$ 趨近何值？解析梯度趨近零，是否足以保證有限差分檢查在任意步長下都可靠？說明理由。
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 6: Optimization)
2. NumPy Documentation: `numpy.logaddexp`, `numpy.tanh`, `np.nditer`. (未逐條核對)
3. PyTorch Documentation on Numerical Stability. (未逐條核對)
<<<NEW>>>
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.（延伸閱讀；此處未核對特定章節）
2. NumPy 官方文件：`numpy.logaddexp`、`numpy.tanh`、`numpy.nditer`（API 名稱供查找；未逐條核對，不作已驗證實作的證據）。
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    def loss(self):
        # BCE = logaddexp(0, z) - y*z。z 趨向負無窮時，
<<<NEW>>>
    def loss(self):
        if self.Z2 is None or self.Y_true is None:
            raise ValueError("Run forward before loss")
        # BCE = logaddexp(0, z) - y*z。z 趨向負無窮時，
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    def backward(self):
        B = self.Y_true.shape[0]
<<<NEW>>>
    def backward(self):
        if any(value is None for value in
               (self.X, self.H, self.Z2, self.Y_pred, self.Y_true)):
            raise ValueError("Run forward before backward")
        B = self.Y_true.shape[0]
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    # 邊界、參數往返及輸入故障測試；不讀取測試集作模型選擇。
    theta = model.pack_params().copy()
<<<NEW>>>
    # 驗證新建模型的狀態契約及非法設定拒絕路徑。
    def expect_value_error(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Expected ValueError")

    fresh = TwoLayerNet(1, 1)
    expect_value_error(fresh.loss)
    expect_value_error(fresh.backward)
    expect_value_error(lambda: TwoLayerNet(0, 2))
    expect_value_error(lambda: TwoLayerNet(True, 2))
    expect_value_error(lambda: TwoLayerNet(2, -1))
    expect_value_error(lambda: TwoLayerNet(2.0, 2))
    expect_value_error(lambda: model.check_gradients(
        X_check, Y_check, h=0))
    expect_value_error(lambda: model.check_gradients(
        X_check, Y_check, tol=np.inf))
    expect_value_error(lambda: model.directional_derivative_check(
        X_check, Y_check, np.zeros(model.pack_params().size)))
    expect_value_error(lambda: model.directional_derivative_check(
        X_check, Y_check, np.full(model.pack_params().size, np.nan)))
    expect_value_error(lambda: model.train_one_step(
        X_train[:1], Y_train[:1], lr=0))

    # 邊界、參數往返及輸入故障測試；不讀取測試集作模型選擇。
    theta = model.pack_params().copy()
<<<END>>>