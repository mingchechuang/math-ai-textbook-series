# 第09章 NumPy兩層網路：逐項梯度驗證

## 學習目標與先備知識

本章旨在建立一個完全自足、可驗證的兩層神經網路訓練流程。學習者需掌握矩陣乘法、鏈式法則、`tanh` 函數導數以及 NumPy 的基本操作。我們不僅要實現前向傳播（Forward Pass）與反向傳播（Backward Pass），更要通過「有限差分法」（Finite Difference Method）對每一個參數的梯度進行嚴格核對。這是確保深度學習程式碼正確性的重要工具：數學推導必須與數值微分在合理的浮點誤差與步長範圍內一致。

本章的核心目標分為四個層次：
1. **數學推導**：明確兩層網路（`Input -> Linear -> Tanh -> Linear -> Stable BCE`）的損失函數及其對所有參數（$W_1, b_1, W_2, b_2$）的解析梯度。
2. **實作驗證**：使用 NumPy 實現前向與反向傳播，並撰寫一個獨立函數計算數值梯度。引入參數展平（Flatten）機制，以便進行方向導數（Directional Derivative）與逐項梯度核對。
3. **數值穩定性**：使用 `np.logaddexp` 實現穩定的二元交叉熵（Binary Cross-Entropy, BCE），避免直接計算 `log(sigmoid(z))` 導致的數值不穩定。
4. **訓練測試**：在小型合成資料集上執行訓練，觀察損失下降，並嚴格執行訓練集、驗證集與保留測試集的切分，以檢測過擬合（Overfitting）。

**先備知識**：
- **矩陣微分**：理解 $\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Z}$ 的幾何意義。
- **浮點精度**：理解 `float64` 的機械精度（machine epsilon, $\epsilon_{mach}$）約為 $10^{-16}$。中心差分近似一階導數的截斷誤差與 $h^2$ 成正比，而捨入誤差與 $h$ 成反比。因此，選擇合適的步長 $h$ 至關重要。
- **Shape 約定**：本卷採用批次優先（Batch-first）約定。輸入 $X \in \mathbb{R}^{B \times D_{in}}$，權重 $W_1 \in \mathbb{R}^{D_{in} \times D_{hid}}$，輸出 $Z_1 = X W_1 + b_1 \in \mathbb{R}^{B \times D_{hid}}$。標籤 $Y_{true}$ 必須嚴格保持 $(B, 1)$ 形狀，以避免 NumPy 廣播（Broadcasting）產生的維度錯誤。

## 問題與直覺

為什麼不直接使用 PyTorch 或 TensorFlow？因為框架的黑盒特性可能隱藏實現細節錯誤。例如，若我們在反向傳播中錯誤地處理了轉置（Transpose），或使用錯誤的 activation 函數導數，框架不會報錯，但模型將永遠無法收斂，或者收斂到局部極小值。透過 NumPy 實現最簡模型，我們能徹底掌控每個中間變量的形狀（shape）與計算邏輯。

**直覺建立**：
想像神經網路是一個複雜的函數 $L(\theta)$，其中 $\theta$ 是參數向量。我們希望找到 $\theta$ 使得 $L(\theta)$ 最小。梯度下降法要求我們知道 $\nabla_{\theta} L$。
- **前向傳播**：計算 $L$ 的值。
- **反向傳播**：利用鏈式法則，從輸出層向輸入層逐層計算梯度。
- **核心問題**：如何確認計算機算出的 $\nabla_{\theta} L$ 是正確的？
- **解決方法**：使用有限差分近似。對於每個參數 $w_i$，計算 $L(w_i + \epsilon)$ 和 $L(w_i - \epsilon)$，並用 $\frac{L(w_i + \epsilon) - L(w_i - \epsilon)}{2\epsilon}$ 來近似偏導數。如果解析梯度與數值梯度一致，則實現大概率正確。

本節將定義一個二分類問題，輸入維度為 5，隱藏層為 8，輸出為 1。我們使用穩定的二元交叉熵（Stable BCE）作為目標函數。

## 定義、定理與推導

### 模型定義
設輸入為 $X \in \mathbb{R}^{B \times D_{in}}$，批次大小 $B$。
1. **第一層仿射變換**：$Z_1 = X W_1 + b_1$
   - $W_1 \in \mathbb{R}^{D_{in} \times D_{hid}}$
   - $b_1 \in \mathbb{R}^{D_{hid}}$
   - $Z_1 \in \mathbb{R}^{B \times D_{hid}}$
2. **非線性激活**：$H = \tanh(Z_1)$
   - $H \in \mathbb{R}^{B \times D_{hid}}$
3. **第二層仿射變換**：$Z_2 = H W_2 + b_2$
   - $W_2 \in \mathbb{R}^{D_{hid} \times D_{out}}$ (此處 $D_{out}=1$)
   - $b_2 \in \mathbb{R}^{D_{out}}$
   - $Z_2 \in \mathbb{R}^{B \times D_{out}}$
4. **損失函數**：穩定 BCE

### 穩定的二元交叉熵 (Stable BCE)
標準 BCE 定義為 $L = -[y \log(\sigma(z)) + (1-y) \log(1-\sigma(z))]$。
直接計算 $\sigma(z) = \frac{1}{1+e^{-z}}$ 在 $z$ 極大時會導致 $e^{-z}$ 趨近 0，進而使 $\sigma(z)$ 接近 1。若 $z$ 極小，$e^{-z}$ 溢出。更嚴峻的是，$\log(\sigma(z))$ 在數值上極不穩定。

我們使用恆等式 $\log(\sigma(z)) = -\log(1+e^{-z})$ 及 $\log(1-\sigma(z)) = -\log(1+e^z)$。
通過代數變形，可以發現：
$$ L = -y \log(\sigma(z)) - (1-y) \log(1-\sigma(z)) $$
$$ = -y (-\log(1+e^{-z})) - (1-y) (-\log(1+e^z)) $$
$$ = y \log(1+e^{-z}) + (1-y) \log(1+e^z) $$
為了避免指數溢出，我們使用 `logsumexp` 的技巧。注意：
$$ \log(1+e^z) = \log(e^0 + e^z) = \text{logaddexp}(0, z) $$
同理，$\log(1+e^{-z}) = \text{logaddexp}(0, -z)$。
因此，對於單一變數 $z$ 和標籤 $y$：
$$ L(z, y) = \text{logaddexp}(0, z) - y z $$
*推導驗證*：
若 $y=1$: $L = \log(1+e^z) - z = \log(\frac{1+e^z}{e^z}) = \log(e^{-z} + 1) = \log(1+e^{-z})$。這與 $y=1$ 時的標準式一致。
若 $y=0$: $L = \log(1+e^z) - 0 = \log(1+e^z)$。這與 $y=0$ 時的標準式一致。
因此，穩定的 BCE 損失（每筆樣本）為：
$$ L_i = \text{logaddexp}(0, z_i) - y_i z_i $$
批次損失取平均：
$$ L = \frac{1}{B} \sum_{i=1}^{B} (\text{logaddexp}(0, z_i) - y_i z_i) $$

### 梯度推導
我們使用矩陣微分法。
已知 $L = \frac{1}{B} \sum (\text{logaddexp}(0, Z_2) - Y_{true} Z_2)$。
對 $Z_2$ 求導：
$$ \frac{\partial L}{\partial Z_2} = \frac{1}{B} \left( \frac{\partial}{\partial Z_2} \text{logaddexp}(0, Z_2) - Y_{true} \right) $$
由於 $\frac{\partial}{\partial z} \text{logaddexp}(0, z) = \sigma(z)$，故：
$$ dZ_2 = \frac{1}{B} (Y_{pred} - Y_{true}) $$
其中 $Y_{pred} = \sigma(Z_2)$。
*註：此處 $Y_{pred}$ 必須使用穩定的 sigmoid 實作，或者在數值範圍內直接計算。*

1. **輸出層梯度**：
   $$ dW_2 = H^T dZ_2 $$
   $$ db_2 = \sum_{i=1}^{B} dZ_{2,i} $$
   *(註：$db_2$ 是沿批次維度求和)*

2. **隱藏層梯度**：
   將 $dZ_2$ 傳回至 $H$：
   $$ dH = dZ_2 W_2^T $$
   通過 $\tanh$ 激活函數。$\tanh(z)$ 的導數為 $1 - \tanh^2(z) = 1 - H^2$。
   $$ dZ_1 = dH \odot (1 - H^2) $$
   *(註：$\odot$ 表示逐元素相乘)*

3. **輸入層梯度**：
   $$ dW_1 = X^T dZ_1 $$
   $$ db_1 = \sum_{i=1}^{B} dZ_{1,i} $$

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

## 逐步手算例題

### 例題 1：前向傳播與穩定 Loss
考慮極小案例：$B=1, D_{in}=2, D_{hid}=2, D_{out}=1$。
輸入 $X = \begin{bmatrix} 1 & 0 \end{bmatrix}$。
參數：
$W_1 = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & -0.3 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0.1 & 0.0 \end{bmatrix}$
$W_2 = \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 0.0 \end{bmatrix}$
目標 $Y_{true} = \begin{bmatrix} 1 \end{bmatrix}$。

1. **計算 $Z_1$**：
   $$ Z_1 = X W_1 + b_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & -0.3 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.0 \end{bmatrix} $$
   $$ Z_1 = \begin{bmatrix} 0.5 & 0.1 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.6 & 0.1 \end{bmatrix} $$

2. **計算 $H$**：
   $$ H = \tanh(Z_1) \approx \begin{bmatrix} \tanh(0.6) & \tanh(0.1) \end{bmatrix} \approx \begin{bmatrix} 0.5370 & 0.0997 \end{bmatrix} $$

3. **計算 $Z_2$**：
   $$ Z_2 = H W_2 + b_2 = \begin{bmatrix} 0.5370 & 0.0997 \end{bmatrix} \begin{bmatrix} 0.4 \\ -0.2 \end{bmatrix} + 0 $$
   $$ Z_2 = 0.5370 \times 0.4 + 0.0997 \times (-0.2) = 0.2148 - 0.01994 = 0.19486 $$

4. **計算 Loss (Stable BCE)**：
   $$ L = \text{logaddexp}(0, 0.19486) - 1 \times 0.19486 $$
   $$ \text{logaddexp}(0, 0.19486) = \log(1 + e^{0.19486}) \approx \log(1 + 1.2151) \approx \log(2.2151) \approx 0.7952 $$
   $$ L = 0.7952 - 0.19486 = 0.60034 $$

### 例題 2：反向傳播手算
接續例題 1 的數值。

1. **計算 $Y_{pred}$**：
   $$ Y_{pred} = \sigma(0.19486) \approx 0.5485 $$

2. **計算 $dZ_2$**：
   $$ dZ_2 = \frac{1}{B}(Y_{pred} - Y_{true}) = \frac{1}{1}(0.5485 - 1) = -0.4515 $$

3. **計算 $dW_2$**：
   $$ dW_2 = H^T dZ_2 = \begin{bmatrix} 0.5370 \\ 0.0997 \end{bmatrix} \times (-0.4515) \approx \begin{bmatrix} -0.2425 \\ -0.0450 \end{bmatrix} $$

4. **計算 $db_2$**：
   $$ db_2 = \sum dZ_2 = -0.4515 $$

5. **計算 $dH$**：
   $$ dH = dZ_2 W_2^T = [-0.4515] \begin{bmatrix} 0.4 & -0.2 \end{bmatrix} = \begin{bmatrix} -0.1806 & 0.0903 \end{bmatrix} $$

6. **計算 $dZ_1$**：
   $$ 1 - H^2 = \begin{bmatrix} 1 - 0.5370^2 & 1 - 0.0997^2 \end{bmatrix} \approx \begin{bmatrix} 0.7115 & 0.9901 \end{bmatrix} $$
   $$ dZ_1 = dH \odot (1 - H^2) = \begin{bmatrix} -0.1806 \times 0.7115 & 0.0903 \times 0.9901 \end{bmatrix} \approx \begin{bmatrix} -0.1285 & 0.0894 \end{bmatrix} $$

7. **計算 $dW_1$**：
   $$ dW_1 = X^T dZ_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} -0.1285 & 0.0894 \end{bmatrix} = \begin{bmatrix} -0.1285 & 0.0894 \\ 0 & 0 \end{bmatrix} $$

若 NumPy 計算結果與此接近（考慮浮點舍入），則實現正確。

## 實作與程式

以下程式碼實現自足模型、前向、反向、參數展平及有限差分核對。
**注意**：為了確保數值微分準確，我們使用 `float64` 數據類型。

```python
import numpy as np

class TwoLayerNet:
    def __init__(self, dim_in, dim_hid, seed=42):
        self.rng = np.random.default_rng(seed)
        # Xavier 初始化
        self.W1 = self.rng.normal(0, np.sqrt(1.0/dim_in), (dim_in, dim_hid)).astype(np.float64)
        self.b1 = np.zeros(dim_hid, dtype=np.float64)
        self.W2 = self.rng.normal(0, np.sqrt(1.0/dim_hid), (dim_hid, 1)).astype(np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)
        
        # 緩存中間變量
        self.X = None
        self.Z1 = None
        self.H = None
        self.Z2 = None
        self.Y_pred = None
        self.Y_true = None

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

    def pack_params(self):
        """Flatten parameters into a single vector for directional derivatives."""
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), 
                               self.W2.flatten(), self.b2.flatten()])

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

    def forward(self, X, Y_true):
        self._check_shapes(X, Y_true)
        self.X = X
        self.Y_true = Y_true
        
        # Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.H = np.tanh(self.Z1)
        # Layer 2
        self.Z2 = self.H @ self.W2 + self.b2
        
        # Stable Sigmoid for prediction
        # Use stable calculation: sigma(z) = 1 / (1 + exp(-z))
        # To avoid overflow in exp, we can split or clip. 
        # For float64, exp(-100) is safe, exp(100) is safe. 
        # exp(-z) might overflow if z < -700.
        # Let's use a stable sigmoid implementation.
        Z2_safe = np.clip(self.Z2, -500, 500)
        self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))
        
        return self.Y_pred

    def loss(self):
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
        
        # Check for non-finite Z2
        if not np.all(np.isfinite(self.Z2)):
            raise ValueError("Z2 contains non-finite values")
            
        loss_per_sample = np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2
        B = self.Y_true.shape[0]
        L = np.mean(loss_per_sample)
        return L

    def backward(self):
        B = self.Y_true.shape[0]
        if B == 0:
            raise ValueError("Batch size cannot be 0")
            
        # dLoss/dZ2
        # Gradient of stable BCE w.r.t Z2 is sigma(Z2) - Y_true
        dZ2 = (self.Y_pred - self.Y_true) / B
        
        # Gradients for W2, b2
        # H: (B, D_hid), dZ2: (B, 1) -> dW2: (D_hid, 1)
        self.dW2 = self.H.T @ dZ2
        # db2: (1,)
        self.db2 = np.sum(dZ2, axis=0)
        
        # Backprop to Z1
        # dZ2: (B, 1), W2: (D_hid, 1) -> dH: (B, D_hid)
        dH = dZ2 @ self.W2.T
        # dZ1: (B, D_hid)
        dZ1 = dH * (1 - self.H ** 2)
        
        # Gradients for W1, b1
        # X: (B, D_in), dZ1: (B, D_hid) -> dW1: (D_in, D_hid)
        self.dW1 = self.X.T @ dZ1
        # db1: (D_hid,)
        self.db1 = np.sum(dZ1, axis=0)

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

    def check_gradients(self, X, Y_true, h=1e-6, tol=1e-5):
        """
        Check all gradients using finite difference.
        """
        self.forward(X, Y_true)
        self.backward()
        
        params = {
            'W1': (self.W1, self.dW1),
            'b1': (self.b1, self.db1),
            'W2': (self.W2, self.dW2),
            'b2': (self.b2, self.db2)
        }
        
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
                # Use symmetric relative error to handle small gradients
                denom = max(1.0, abs(num_grad), abs(ana_grad))
                rel_err = abs_err / denom
                
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                total_checks += 1
                it.iternext()
                
        status = "PASS" if max_rel_err < tol else "FAIL"
        print(f"Gradient Check: {status}, Max Abs Err: {max_abs_err:.2e}, Max Rel Err: {max_rel_err:.2e} (Checks: {total_checks})")
        return max_rel_err

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

    def train_one_step(self, X, Y_true, lr=0.01):
        self.forward(X, Y_true)
        L = self.loss()
        self.backward()
        
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        
        return L

if __name__ == "__main__":
    # 1. Data Generation
    dim_in, dim_hid = 5, 8
    B_train, B_test = 100, 20
    # Use different seeds for train and test to ensure they are distinct
    X_train = np.random.default_rng(0).normal(0, 1, (B_train, dim_in)).astype(np.float64)
    # Ensure Y is (B, 1)
    Y_train = (X_train[:, [0]] > 0).astype(np.float64) 
    
    X_test = np.random.default_rng(1).normal(0, 1, (B_test, dim_in)).astype(np.float64)
    Y_test = (X_test[:, [0]] > 0).astype(np.float64)
    
    # 2. Initialize Model
    model = TwoLayerNet(dim_in, dim_hid)
    
    # 3. Gradient Check
    # Use a small subset for speed
    X_check = X_train[:10]
    Y_check = Y_train[:10]
    err = model.check_gradients(X_check, Y_check)
    if err >= 1e-5:
        raise Exception("Gradient Check Failed! Do not proceed.")
        
    # Directional Derivative Check
    v = np.random.default_rng(42).normal(0, 1, model.pack_params().size).astype(np.float64)
    v = v / np.linalg.norm(v) # Normalize direction
    model.forward(X_check, Y_check)
    model.backward()
    dd_err = model.directional_derivative_check(X_check, Y_check, v)
    if dd_err >= 1e-5:
        raise Exception("Directional Derivative Check Failed!")

    # 4. Training Loop
    # Calculate initial loss on full training set
    model.forward(X_train, Y_train)
    initial_train_loss = model.loss()
    print(f"Initial Train Loss: {initial_train_loss:.4f}")
    
    for i in range(100):
        L = model.train_one_step(X_train, Y_train, lr=0.1)
        if i % 10 == 0:
            print(f"Iter {i}, Train Loss: {L:.4f}")
            
    # 5. Evaluation
    model.forward(X_train, Y_train)
    final_train_loss = model.loss()
    train_preds = (model.Y_pred > 0.5).flatten()
    train_labels = Y_train.flatten()
    train_acc = np.mean(train_preds == train_labels)
    
    model.forward(X_test, Y_test)
    final_test_loss = model.loss()
    test_preds = (model.Y_pred > 0.5).flatten()
    test_labels = Y_test.flatten()
    test_acc = np.mean(test_preds == test_labels)
    
    print(f"Final Train Loss: {final_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Final Test Loss: {final_test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Overfitting Check: Expect train loss to drop significantly and acc to be high
    # Test loss should be reasonable, but might be higher than train if overfitting
    if final_train_loss > initial_train_loss * 0.1:
        print("Warning: Model may not be fitting training data well.")
    if test_acc < 0.5:
        print("Warning: Model performance on test set is poor.")
```

## 測試與預期結果

### 正常測試
隨機生成 $B=10, D_{in}=5$ 資料。執行 `check_gradients`。
**預期**：返回的 `max_rel_err` 應小於 $10^{-5}$。程式應輸出 `PASS`。方向導數檢查也應通過。

### 邊界測試
1. **極大 Logit**：當 $Z_2 = 100$ 且 $Y_{true}=0$。
   - **Loss**：應為 $\text{logaddexp}(0, 100) - 0 \approx 100$。
   - **Gradient**：$Y_{pred} \approx 1$，$dZ_2 \approx (1-0)/B = 1$。
   - **預期**：Loss 為有限值，梯度非零。
2. **極小 Logit**：當 $Z_2 = -100$ 且 $Y_{true}=1$。
   - **Loss**：應為 $\text{logaddexp}(0, -100) - 1(-100) \approx 0 - (-100) = 100$。
   - **Gradient**：$Y_{pred} \approx 0$，$dZ_2 \approx (0-1)/B = -1$。
3. **Shape 錯誤**：若 $Y_{true}$ 為 $(B,)$，程式應拋出 `ValueError`。

### 故障測試
**錯誤 Scenario**：在 `backward` 中，誤將 `dH = dZ2 @ self.W2.T` 改為 `dH = dZ2 * self.W2.T`。
**預期**：由於 `W2` 是 $(D_{hid}, 1)$，`W2.T` 是 $(1, D_{hid})$。`dZ2` 是 $(B, 1)$。
逐元素乘法 `dZ2 * self.W2.T` 會廣播為 $(B, D_{hid})$。
在這種特定情況下（輸出維度為 1），逐元素乘法與矩陣乘法結果**相同**。
因此，**這個特定的故障測試不會觸發失敗**。
**正確的故障測試**：例如漏掉 `1 - H**2`，即 `dZ1 = dH`。
**預期**：梯度核對將失敗，`max_rel_err` 將遠大於 $10^{-5}$。

## 反例與常見陷阱

1. **Shape 錯誤**：$X$ 為 $(B, D_{in})$，$W_1$ 為 $(D_{in}, D_{hid})$。若標籤 $Y_{true}$ 為 $(B,)$，則 $Y_{pred} - Y_{true}$ 會廣播為 $(B, B)$ 或錯誤維度，導致損失計算錯誤。NumPy 不會報錯，但結果無意義。
2. **Sigmoid 飽和**：當 $|Z| > 20$，$\sigma(Z)$ 接近 0 或 1。若直接計算 $\log(\sigma(Z))$，會產生 `log(0)` 即 `-inf` 或數值精度問題。使用 `logaddexp` 可避免此問題。
3. **未平均 Loss**：若 Loss 定義為 Sum 而非 Mean，反向傳播的 $dZ_2$ 不應除以 $B$。若混淆，梯度檢查會失敗。本卷統一使用 Mean Loss。
4. **Bias 梯度**：Bias 的梯度是對應層輸出的梯度在批次軸上的和。若誤除以 $B$，梯度檢查將失敗。

## AI、幾何與養殖案例

在養殖水質監控中，可建立此兩層網路預測「溶氧量異常」。
- **輸入**：pH、溫度、DO 當前值（3維）。
- **輸出**：異常概率。
- **驗證**：使用歷史數據切分為 Train、Validation 和 Test。
- **陷阱**：若訓練集包含未來數據（Look-ahead bias），模型表現好但實戰失效。必須嚴格切分 Time Series，確保 Test 集時間大於 Train 集最大時間戳。
- **注意**：此處僅為合成資料實驗，不應直接解讀為真實養殖部署能力。

## 習題

1. **手算**：給定 $B=2, D_{in}=2, D_{hid}=2$，計算 $dW_2$ 和 $db_2$。
2. **程式**：修改程式碼，使用 ReLU 替代 tanh，並調整梯度檢查邏輯。
3. **反例**：解釋為何當 $Y=1$ 且 $W_2$ 極大時，梯度可能趨近 0？這是否意味著梯度檢查失敗？
4. **整合**：設計一個測試，驗證當 $B=1$ 時，bias 梯度是否等於 $dZ$ 的元素。

## 習題解答

1. **手算**：
   令 $X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $W_1 = I$, $b_1=0$。
   $H = \tanh(X) = \begin{bmatrix} 0.76 & 0 \\ 0 & 0.76 \end{bmatrix}$。
   $W_2 = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$, $b_2=0$。
   $Y_{true} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$。
   $Z_2 = \begin{bmatrix} 0.76 \times 0.5 \\ 0.76 \times 0.5 \end{bmatrix} = \begin{bmatrix} 0.38 \\ 0.38 \end{bmatrix}$。
   $Y_{pred} = \sigma(Z_2) \approx \begin{bmatrix} 0.594 \\ 0.594 \end{bmatrix}$。
   $dZ_2 = \frac{1}{2} (Y_{pred} - Y_{true}) = \frac{1}{2} \begin{bmatrix} 0.594 - 1 \\ 0.594 - 0 \end{bmatrix} = \begin{bmatrix} -0.203 \\ 0.297 \end{bmatrix}$。
   $dW_2 = H^T dZ_2 = \begin{bmatrix} 0.76 & 0 \\ 0 & 0.76 \end{bmatrix} \begin{bmatrix} -0.203 \\ 0.297 \end{bmatrix} = \begin{bmatrix} -0.154 \\ 0.226 \end{bmatrix}$。
   $db_2 = \sum dZ_2 = -0.203 + 0.297 = 0.094$。

2. **程式**：
   將 `self.H = np.tanh(Z1)` 改為 `self.H = np.maximum(0, Z1)`。
   導數改為 `dZ1 = dH * (Z1 > 0)`。
   注意：ReLU 在 0 處不可導，數值梯度檢查在 0 附近可能不穩定，建議選擇非零點進行檢查。

3. **反例**：
   當 $Y \to 1$，$\sigma'(z) \to 0$。梯度 $dW_2 = H^T (Y-Y_{true})$。若 $Y_{true}=1$，則 $Y-Y_{true} \to 0$。故梯度趨近 0。
   這**不是**梯度檢查失敗，而是數學特性。梯度檢查會通過，因為解析梯度和數值梯度都趨近於 0。

4. **整合**：
   $db = \sum_{i=1}^{B} dZ_{i}$。
   若 $B=1$，則 $db$ 應等於 $dZ$ 的對應元素。
   測試：隨機 $X, W$。計算 `self.db1`。
   手動計算：`dZ1 = ...`; `sum_dZ = np.sum(dZ1, axis=0)`。
   斷言 `np.allclose(self.db1, sum_dZ)`。

## 本章小結

本章通過 NumPy 實現了兩層網路的完整梯度驗證。關鍵在於：
1. **Shape 嚴謹**：每次乘法後檢查 shape，避免隱性轉置錯誤和廣播錯誤。
2. **Loss 定義明確**：使用穩定的 BCE 公式，Mean vs Sum 決定是否除以 $B$。
3. **有限差分核對**：這是偵測實現錯誤的最有效方法之一。
4. **邊界處理**：Sigmoid 飽和與 Log 域定義，使用 `logaddexp` 防止 NaN。
5. **數據切分**：保留測試集以檢測過擬合，不使用測試集調參。

## 參考來源

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 6: Optimization)
2. NumPy Documentation: `numpy.logaddexp`, `numpy.tanh`, `np.nditer`. (未逐條核對)
3. PyTorch Documentation on Numerical Stability. (未逐條核對)