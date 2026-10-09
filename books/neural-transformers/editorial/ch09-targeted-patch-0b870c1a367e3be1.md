<<<PATCH 01>>>
<<<OLD>>>
### 命題與證明
**命題 9.1**：對於上述兩層網路，若前向傳播計算正確，且使用中心差分（Central Difference）進行數值梯度檢查，則解析梯度與數值梯度應在浮點精度與步長誤差範圍內一致。

**證明**：
設損失函數 $L(\theta)$ 在某點 $\theta$ 附近三階可微。
泰勒展開（Taylor Expansion）：
$$ L(\theta + h) = L(\theta) + h \nabla L(\theta) + \frac{h^2}{2} \nabla^2 L(\theta) h + O(h^3) $$
$$ L(\theta - h) = L(\theta) - h \nabla L(\theta) + \frac{h^2}{2} \nabla^2 L(\theta) h + O(h^3) $$
兩式相減：
$$ L(\theta + h) - L(\theta - h) = 2h \nabla L(\theta) + O(h^3) $$
$$ \frac{L(\theta + h) - L(\theta - h)}{2h} = \nabla L(\theta) + O(h^2) $$
因此，中心差分的截斷誤差為 $O(h^2)$。
考慮浮點運算，每個運算引入相對誤差 $\epsilon_{mach}$。
數值計算的總誤差 $E$ 可估計為截斷誤差加上捨入誤差：
$$ E(h) \approx C_1 h^2 + \frac{C_2 \epsilon_{mach}}{h} $$
為了最小化 $E(h)$，對 $h$ 求導並令其為零：
$$ \frac{dE}{dh} = 2 C_1 h - \frac{C_2 \epsilon_{mach}}{h^2} = 0 \implies h^3 \propto \epsilon_{mach} \implies h \propto \epsilon_{mach}^{1/3} $$
對於 `float64`，$\epsilon_{mach} \approx 10^{-16}$，故最佳步長 $h \approx 10^{-5}$ 至 $10^{-6}$。
若我們實作的反向傳播邏輯與鏈式法則推導完全一致，則解析梯度與數值梯度之差異應小於容差 $\delta$。
證明完畢。$\blacksquare$
<<<NEW>>>
### 命題與證明
**命題 9.1（中心差分的截斷誤差）**：令 $L:\mathbb{R}^n\to\mathbb{R}$，固定參數向量 $\theta$ 與方向向量 $v$，並定義純量函數 $\phi(t)=L(\theta+tv)$。若 $\phi$ 在包含 $[-h,h]$ 的區間三階可微，且該區間內 $|\phi'''(t)|\le M$，則
$$ \left|\frac{\phi(h)-\phi(-h)}{2h}-\nabla L(\theta)^T v\right|\le \frac{M h^2}{6}. $$
此命題只描述中心差分對方向導數的截斷誤差，不保證任一手寫反向傳播必然正確。

**證明**：對 $\phi$ 在 $0$ 附近作 Taylor 展開。由帶餘項的 Taylor 定理，存在 $\xi_+\in(0,h)$ 與 $\xi_-\in(-h,0)$，使得
$$ \phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)+\frac{h^3}{6}\phi'''(\xi_+), $$
$$ \phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)-\frac{h^3}{6}\phi'''(\xi_-). $$
相減並除以 $2h$，得到
$$ \frac{\phi(h)-\phi(-h)}{2h}=\phi'(0)+\frac{h^2}{12}\bigl(\phi'''(\xi_+)+\phi'''(\xi_-)\bigr). $$
由鏈式法則，$\phi'(0)=\nabla L(\theta)^T v$。再用三階導數的界 $M$，即可得所述誤差上界。證明完畢。$\blacksquare$

浮點計算另會引入捨入誤差；在簡化的誤差模型下，總誤差常以 $E(h)\approx C_1h^2+C_2\epsilon_{mach}/h$ 描述。這個模型的常數依賴參數與損失的尺度，不能據此指定普遍最佳步長。對 `float64` 而言，$10^{-5}$ 至 $10^{-6}$ 可作為尺度約為一時的候選量級；實作時宜比較數個步長，並同時檢查絕對誤差與方向導數。逐項有限差分及方向導數核對，提供的是對目前資料、參數點與步長的局部證據；通過檢查不構成對所有輸入或所有參數的證明。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
        # Stable Sigmoid for prediction
        # Use stable calculation: sigma(z) = 1 / (1 + exp(-z))
        # To avoid overflow in exp, we can split or clip. 
        # For float64, exp(-100) is safe, exp(100) is safe. 
        # exp(-z) might overflow if z < -700.
        # Let's use a stable sigmoid implementation.
        Z2_safe = np.clip(self.Z2, -500, 500)
        self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))
<<<NEW>>>
        # 分正負區間計算 sigmoid，避免 exp(-z) 在負大值時溢位；
        # 不裁切 logits，確保此值仍是未裁切 BCE 的精確導數。
        self.Y_pred = np.empty_like(self.Z2, dtype=np.float64)
        pos = self.Z2 >= 0
        self.Y_pred[pos] = 1.0 / (1.0 + np.exp(-self.Z2[pos]))
        ez = np.exp(self.Z2[~pos])
        self.Y_pred[~pos] = ez / (1.0 + ez)
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
        # For large negative Z2, log(1 + exp(Z2)) ~ Z2, so loss ~ Z2 - Y*Z2.
        # If Y=1, loss ~ 0. If Y=0, loss ~ Z2 (which is negative? No.)
        # Wait, log(1+exp(z)) is always positive.
        # Let's verify the stable BCE formula again.
        # L = -y log(sigma(z)) - (1-y) log(1-sigma(z))
        # L = log(1+exp(z)) - y z
        # This formula is correct and stable.
<<<NEW>>>
        # 當 Z2 很負時，log(1 + exp(Z2)) 約為 exp(Z2)，趨近 0；
        # 當 Z2 很正時，它約為 Z2。logaddexp 可穩定計算兩側。
        # BCE logits 公式為 log(1 + exp(Z2)) - Y_true * Z2。
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
    def _check_shapes(self, X, Y_true):
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if Y_true.ndim != 2 or Y_true.shape[1] != 1:
            raise ValueError("Y_true must be shape (B, 1)")
        if X.shape[0] != Y_true.shape[0]:
            raise ValueError("Batch size mismatch")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y_true)):
            raise ValueError("Input contains non-finite values")
        if not np.all((Y_true == 0) | (Y_true == 1)):
            raise ValueError("Y_true must be binary (0 or 1)")
<<<NEW>>>
    def _check_shapes(self, X, Y_true):
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if Y_true.ndim != 2 or Y_true.shape[1] != 1:
            raise ValueError("Y_true must be shape (B, 1)")
        if X.shape[0] == 0:
            raise ValueError("Batch size cannot be 0")
        if X.shape[0] != Y_true.shape[0]:
            raise ValueError("Batch size mismatch")
        if X.shape[1] != self.W1.shape[0]:
            raise ValueError("Input feature dimension mismatch")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y_true)):
            raise ValueError("Input contains non-finite values")
        if not np.all((Y_true == 0) | (Y_true == 1)):
            raise ValueError("Y_true must be binary (0 or 1)")
<<<END>>>

<<<PATCH 05>>>
<<<OLD>>>
    def unpack_params(self, params):
        """Restore parameters from a flat vector."""
        # 1. Determine slices
        size_W1 = self.W1.size
        size_b1 = self.b1.size
        size_W2 = self.W2.size
        
        self.W1 = params[:size_W1].reshape(self.W1.shape)
        self.b1 = params[size_W1:size_W1+size_b1].reshape(self.b1.shape)
        self.W2 = params[size_W1+size_b1:size_W1+size_b1+size_W2].reshape(self.W2.shape)
        self.b2 = params[-1:].reshape(self.b2.shape)
<<<NEW>>>
    def unpack_params(self, params):
        """Restore parameters from a finite, correctly sized flat vector."""
        expected = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        if params.ndim != 1 or params.size != expected:
            raise ValueError("params must be a 1D vector of the exact parameter size")
        if not np.all(np.isfinite(params)):
            raise ValueError("params contains non-finite values")
        shapes = (self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape)
        sizes = (self.W1.size, self.b1.size, self.W2.size, self.b2.size)
        values = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            values.append(params[offset:offset + size].reshape(shape).copy())
            offset += size
        self.W1, self.b1, self.W2, self.b2 = values
<<<END>>>