<<<PATCH 04>>>
<<<OLD>>>
### 幾何視角
在統計流形中，KL 散度的局部二階近似與 Fisher 資訊矩陣相關。在適當正則條件及小 $\delta$ 下：
$$ D_{KL}(p_\theta \| p_{\theta+\delta}) \approx \frac{1}{2} \delta^T I(θ) \delta $$
這表明 KL 在局部由 Fisher 計量支配。但這只是局部近似，不表示 KL 是全域對稱距離，也不保證 NLL 優化等同於最短 Fisher geodesic。
<<<NEW>>>
### 幾何視角
以下是有限樣本空間上的局部結果。假設各類別機率在 $\theta$ 的鄰域內嚴格為正且二次連續可微，Fisher 資訊矩陣 $I(\theta)$ 存在，並令 $\delta\to0$，則：
$$ D_{KL}(p_\theta \| p_{\theta+\delta}) = \frac{1}{2} \delta^T I(\theta) \delta + o(\|\delta\|^2) $$
這裡 $I(\theta)=\sum_x p_\theta(x)\nabla_\theta\ln p_\theta(x)\nabla_\theta\ln p_\theta(x)^T$。在上述有限空間及正機率條件下，可逐項對 $\sum_xp_\theta(x)=1$ 微分：score 的期望為零，對數機率 Hessian 的負期望等於 $I(\theta)$；將 $\ln p_{\theta+\delta}(x)$ 作二階 Taylor 展開並代入 KL 定義，即得所示二階項。這個推導不表示 KL 是全域對稱距離，也不保證 NLL 優化等同於最短 Fisher geodesic。
<<<END>>>