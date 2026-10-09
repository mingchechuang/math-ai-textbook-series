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
此命題只界定中心差分的截斷誤差，不保證手寫反向傳播正確。

**證明**：由 Taylor 定理，存在 $\xi_+\in(0,h)$ 與 $\xi_-\in(-h,0)$，使得
$$ \phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)+\frac{h^3}{6}\phi'''(\xi_+), $$
$$ \phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)-\frac{h^3}{6}\phi'''(\xi_-). $$
相減並除以 $2h$，得到
$$ \frac{\phi(h)-\phi(-h)}{2h}=\phi'(0)+\frac{h^2}{12}\bigl(\phi'''(\xi_+)+\phi'''(\xi_-)\bigr). $$
由鏈式法則，$\phi'(0)=\nabla L(\theta)^T v$。又因兩個三階導數的絕對值都不超過 $M$，誤差至多為 $Mh^2/6$。這證明的是有限差分近似方向導數的性質，沒有假設或推出手寫 backward 正確。證明完畢。$\blacksquare$

浮點捨入誤差會與截斷誤差共同影響實際計算。在簡化誤差模型 $E(h)\approx C_1h^2+C_2\epsilon_{mach}/h$ 下，誤差最小值依賴常數、參數尺度、損失尺度及局部導數。因此 `float64` 的 $10^{-5}$ 至 $10^{-6}$ 只能作為尺度約為一時的候選步長，不是普遍最佳值；實務上可比較數個步長，觀察誤差是否先下降再上升。

逐項有限差分與方向導數檢查提供的證據範圍也不同。逐項檢查會分別擾動每個參數元素，較容易定位是哪一組權重或偏置的梯度有誤，但每個元素都需重新計算兩次損失，成本隨參數數量增加。方向導數把展平後的參數向量沿固定方向 $v$ 擾動，只比較一個內積與一個差分，成本較低；但單一方向可能碰巧沒有揭露某些錯誤，故不能取代逐項核對。兩者都應在相同資料、相同目標函數及相同 reduction 下計算，而且每次擾動後都必須重新執行前向傳播。

若檢查失敗，應先確認 loss 採用 batch mean 時，輸出梯度只除以 $B$ 一次；再核對 bias 梯度是否沿 batch 軸求和、矩陣乘法的轉置是否正確、tanh 導數是否為 $1-H^2$，以及展平與還原參數時的順序是否一致。接著可改用幾個不同步長重做檢查：若步長變小後誤差先下降，通常表示截斷誤差在減少；若繼續縮小反而使結果不穩，可能是浮點相消或參數尺度造成。若改用 ReLU，還要確認擾動沒有跨過零點，因為中心差分跨越折點時不能直接驗證單一側的次梯度。通過檢查只支持目前測試點附近的梯度實作，不能代替邊界測試、shape 測試或小資料訓練檢查。
<<<END>>>