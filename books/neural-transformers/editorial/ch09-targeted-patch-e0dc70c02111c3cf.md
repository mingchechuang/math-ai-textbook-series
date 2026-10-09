<<<PATCH 09>>>
<<<OLD>>>
    def numerical_gradient(self, param_name, idx, h=1e-6):
        """
        Compute numerical gradient for a specific parameter element.
        """
        if param_name == 'W1':
            param = self.W1
        elif param_name == 'b1':
            param = self.b1
        elif param_name == 'W2':
            param = self.W2
        elif param_name == 'b2':
            param = self.b2
        else:
            raise ValueError(f"Unknown param name: {param_name}")
            
        original_value = param[idx]
        
        # f(x+h)
        param[idx] = original_value + h
        self.forward(self.X, self.Y_true)
        loss_plus_val = self.loss()
        
        # f(x-h)
        param[idx] = original_value - h
        self.forward(self.X, self.Y_true)
        loss_minus_val = self.loss()
        
        # Restore
        param[idx] = original_value
        
        # Recalculate forward/backward to restore cache
        self.forward(self.X, self.Y_true)
        self.backward()
        
        return (loss_plus_val - loss_minus_val) / (2 * h)
<<<NEW>>>
    def numerical_gradient(self, param_name, idx, h=1e-6):
        """Compute one parameter's central-difference gradient and restore state."""
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        if self.X is None or self.Y_true is None:
            raise ValueError("Run forward before requesting a numerical gradient")
        if param_name == 'W1':
            param = self.W1
        elif param_name == 'b1':
            param = self.b1
        elif param_name == 'W2':
            param = self.W2
        elif param_name == 'b2':
            param = self.b2
        else:
            raise ValueError(f"Unknown param name: {param_name}")

        original_value = param[idx]
        try:
            param[idx] = original_value + h
            self.forward(self.X, self.Y_true)
            loss_plus_val = self.loss()

            param[idx] = original_value - h
            self.forward(self.X, self.Y_true)
            loss_minus_val = self.loss()
        finally:
            param[idx] = original_value
            self.forward(self.X, self.Y_true)
            self.backward()

        numeric = (loss_plus_val - loss_minus_val) / (2 * h)
        if not np.isfinite(numeric):
            raise ValueError("Numerical gradient is non-finite")
        return numeric
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
    def directional_derivative_check(self, X, Y_true, v, h=1e-6, tol=1e-5):
        """
        Check directional derivative: grad(theta) . v approx (L(theta+hv) - L(theta-hv)) / 2h
        v is a random direction vector with same shape as pack_params()
        """
        theta = self.pack_params()
        
        # Analytic directional derivative
        grad_flat = np.concatenate([self.dW1.flatten(), self.db1.flatten(), 
                                    self.dW2.flatten(), self.db2.flatten()])
        analytic_dd = np.dot(grad_flat, v)
        
        # Numerical directional derivative
        theta_plus = theta + h * v
        theta_minus = theta - h * v
        
        self.unpack_params(theta_plus)
        self.forward(X, Y_true)
        L_plus = self.loss()
        
        self.unpack_params(theta_minus)
        self.forward(X, Y_true)
        L_minus = self.loss()
        
        # Restore original
        self.unpack_params(theta)
        
        numeric_dd = (L_plus - L_minus) / (2 * h)
        
        abs_err = abs(analytic_dd - numeric_dd)
        rel_err = abs_err / max(1.0, abs(analytic_dd), abs(numeric_dd))
        
        status = "PASS" if rel_err < tol else "FAIL"
        print(f"Directional Derivative Check: {status}, Abs Err: {abs_err:.2e}, Rel Err: {rel_err:.2e}")
        return rel_err
<<<NEW>>>
    def directional_derivative_check(self, X, Y_true, v, h=1e-6, tol=1e-5):
        """Compare the analytic and central-difference derivatives along v."""
        if not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("h and tol must be finite and positive")
        theta = self.pack_params()
        v = np.asarray(v, dtype=np.float64)
        if (v.ndim != 1 or v.size != theta.size
                or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0):
            raise ValueError("v must be a finite, nonzero flat vector matching parameters")

        try:
            self.forward(X, Y_true)
            self.backward()
            grad_flat = np.concatenate([
                self.dW1.ravel(), self.db1.ravel(),
                self.dW2.ravel(), self.db2.ravel()
            ])
            analytic_dd = np.dot(grad_flat, v)

            self.unpack_params(theta + h * v)
            self.forward(X, Y_true)
            L_plus = self.loss()

            self.unpack_params(theta - h * v)
            self.forward(X, Y_true)
            L_minus = self.loss()
            numeric_dd = (L_plus - L_minus) / (2 * h)

            abs_err = abs(analytic_dd - numeric_dd)
            scaled_err = abs_err / max(1.0, abs(analytic_dd), abs(numeric_dd))
            status = "PASS" if scaled_err < tol else "FAIL"
            print(f"Directional Derivative Check: {status}, Abs Err: {abs_err:.2e}, Scaled Err: {scaled_err:.2e}")
            return scaled_err
        finally:
            self.unpack_params(theta)
            self.forward(X, Y_true)
            self.backward()
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
                # Use symmetric relative error to handle small gradients
                denom = max(1.0, abs(num_grad), abs(ana_grad))
                rel_err = abs_err / denom
<<<NEW>>>
                # Scaled error: the denominator floor keeps near-zero gradients stable.
                denom = max(1.0, abs(num_grad), abs(ana_grad))
                scaled_err = abs_err / denom
<<<END>>>

<<<PATCH 09>>>
<<<OLD>>>
   將 `self.H = np.tanh(Z1)` 改為 `self.H = np.maximum(0, Z1)`。
   導數改為 `dZ1 = dH * (Z1 > 0)`。
<<<NEW>>>
   將 `self.H = np.tanh(self.Z1)` 改為 `self.H = np.maximum(0.0, self.Z1)`。
   導數改為 `dZ1 = dH * (self.Z1 > 0.0)`；此例在零點採用次梯度 0。有限差分測試須選擇遠離零點的參數與輸入，並確認擾動不跨過零點。
<<<END>>>