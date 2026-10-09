<<<PATCH 09>>>
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
**命題 9.1（中心差分的截斷誤差）**：令 $L:\mathbb{R}^n\to\mathbb{R}$，固定參數向量 $\theta$ 與方向向量 $v$，定義純量函數 $\phi(t)=L(\theta+tv)$。若 $\phi$ 在包含 $[-h,h]$ 的區間三階可微，且該區間內 $|\phi'''(t)|\le M$，則
$$ \left|\frac{\phi(h)-\phi(-h)}{2h}-\nabla L(\theta)^T v\right|\le \frac{M h^2}{6}. $$
此命題只說明中心差分的截斷誤差；前向計算正確本身並不能保證手寫反向傳播正確。

**證明**：由 Taylor 定理，存在 $\xi_+\in(0,h)$ 與 $\xi_-\in(-h,0)$，使得
$$ \phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)+\frac{h^3}{6}\phi'''(\xi_+), $$
$$ \phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)-\frac{h^3}{6}\phi'''(\xi_-). $$
相減並除以 $2h$，得到
$$ \frac{\phi(h)-\phi(-h)}{2h}=\phi'(0)+\frac{h^2}{12}\bigl(\phi'''(\xi_+)+\phi'''(\xi_-)\bigr). $$
由鏈式法則，$\phi'(0)=\nabla L(\theta)^T v$。再由 $|\phi'''(t)|\le M$，可得誤差至多為 $Mh^2/6$。證明完畢。$\blacksquare$

浮點捨入誤差會與截斷誤差共同影響數值結果。在簡化模型 $E(h)\approx C_1h^2+C_2\epsilon_{mach}/h$ 下，步長量級受常數、參數尺度、損失尺度及局部導數影響。對 `float64` 而言，$10^{-5}$ 至 $10^{-6}$ 只能作為尺度約為一時的候選量級，不是普遍最佳步長；可比較數個步長，並同時檢查絕對誤差及方向導數。核對通過僅提供目前參數點、資料與步長下的局部證據，不是對所有輸入的正確性證明。
<<<END>>>

<<<PATCH 09>>>
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
        # 分正負區間計算 sigmoid，避免負大 logits 造成指數溢位；
        # 不裁切 logits，使此值仍是未裁切 BCE 對 logits 的導數。
        self.Y_pred = np.empty_like(self.Z2, dtype=np.float64)
        pos = self.Z2 >= 0
        self.Y_pred[pos] = 1.0 / (1.0 + np.exp(-self.Z2[pos]))
        ez = np.exp(self.Z2[~pos])
        self.Y_pred[~pos] = ez / (1.0 + ez)
<<<END>>>