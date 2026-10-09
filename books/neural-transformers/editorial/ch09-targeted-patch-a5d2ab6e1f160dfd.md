<<<PATCH 01>>>
<<<OLD>>>
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
**命題 9.1**：固定參數向量 $\theta$ 與方向 $v$，令 $\phi(t)=L(\theta+tv)$。若 $\phi$ 在零附近三階連續可微，則中心差分滿足
$$\frac{L(\theta+hv)-L(\theta-hv)}{2h}=\nabla L(\theta)^Tv+O(h^2).$$

**證明**：對純量函數 $\phi$ 在零點作三階泰勒展開，正負兩側分別為
$$\phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)+\frac{h^3}{6}\phi'''(\xi_+),$$
$$\phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)-\frac{h^3}{6}\phi'''(\xi_-),$$
其中 $\xi_+$ 與 $\xi_-$ 分別位於零與對應的擾動點之間。相減後，常數項及二次項抵消；三階導數在該鄰域有界，因此除以 $2h$ 後的餘項為 $O(h^2)$。由鏈式法則，$\phi'(0)=\nabla L(\theta)^Tv$，命題得證。$\blacksquare$

取 $v$ 為某個座標的單位向量，就得到逐項梯度檢查。此命題只保證精確算術下的截斷誤差，**不**保證寫錯的反傳與差分一致。浮點相減還會放大捨入誤差；以 $C_1h^2+C_2\epsilon_{mach}/h$ 作局部誤差模型時，候選步長為 $(C_2\epsilon_{mach}/(2C_1))^{1/3}$。常數取決於函數、方向和尺度，故 `float64` 的 $10^{-5}$ 至 $10^{-6}$ 只是可試的量級，不是普遍最佳值；檢查失敗時也應比較鄰近步長，而非逕自放寬容差。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
        if X.shape[0] != Y_true.shape[0]:
            raise ValueError("Batch size mismatch")
<<<NEW>>>
        if X.shape[0] != Y_true.shape[0] or X.shape[0] == 0:
            raise ValueError("Batch size mismatch or empty batch")
        if X.shape[1] != self.W1.shape[0]:
            raise ValueError("Input feature dimension mismatch")
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
        # 1. Determine slices
        size_W1 = self.W1.size
        size_b1 = self.b1.size
        size_W2 = self.W2.size
        
        self.W1 = params[:size_W1].reshape(self.W1.shape)
        self.b1 = params[size_W1:size_W1+size_b1].reshape(self.b1.shape)
        self.W2 = params[size_W1+size_b1:size_W1+size_b1+size_W2].reshape(self.W2.shape)
        self.b2 = params[-1:].reshape(self.b2.shape)
<<<NEW>>>
        params = np.asarray(params)
        shapes = (self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape)
        if (params.ndim != 1 or params.size != sum(np.prod(s) for s in shapes)
                or not np.all(np.isfinite(params))):
            raise ValueError("Expected a finite flat vector of exact parameter length")
        offset = 0
        restored = []
        for shape in shapes:
            size = int(np.prod(shape))
            restored.append(params[offset:offset + size].copy().reshape(shape))
            offset += size
        self.W1, self.b1, self.W2, self.b2 = restored
<<<END>>>
<<<PATCH 04>>>
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
        # 與未裁切 logits 的 BCE 保持一致；分支避免 exp 溢出。
        if not np.all(np.isfinite(self.Z2)):
            raise ValueError("Z2 contains non-finite values")
        self.Y_pred = np.empty_like(self.Z2)
        pos = self.Z2 >= 0
        self.Y_pred[pos] = 1.0 / (1.0 + np.exp(-self.Z2[pos]))
        ez = np.exp(self.Z2[~pos])
        self.Y_pred[~pos] = ez / (1.0 + ez)
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
        # Stable BCE Loss: logaddexp(0, Z2) - Y_true * Z2
        # np.logaddexp(0, Z2) is equivalent to log(1 + exp(Z2))
        # This is stable for large positive Z2.
        # For large negative Z2, log(1 + exp(Z2)) ~ Z2, so loss ~ Z2 - Y*Z2.
        # If Y=1, loss ~ 0. If Y=0, loss ~ Z2 (which is negative? No.)
        # Wait, log(1+exp(z)) is always positive.
        # Let's verify the stable BCE formula again.
        # L = -y log(sigma(z)) - (1-y) log(1-sigma(z))
        # L = log(1+exp(z)) - y z
        # This formula is correct and stable.
<<<NEW>>>
        # BCE = logaddexp(0, z) - y*z；z 趨向負無窮時
        # logaddexp(0, z) 趨近 0，而不是趨近 z。
<<<END>>>