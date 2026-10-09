<<<PATCH 01>>>
<<<OLD>>>
由於 $\mathbf{u}$ 任意，故 $H$ 半正定。負定情況類推。$\blacksquare$

**定理 14.3 (二階充分條件)**
設 $f \in C^2(U)$ 且 $\mathbf{x}_0$ 是駐點（$\nabla f(\mathbf{x}_0)=0$）。
1.  若 $H$ 是正定的，則 $\mathbf{x}_0$ 是 $f$ 的**嚴格局部極小值點**。
2.  若 $H$ 是負定的，則 $\mathbf{x}_0$ 是 $f$ 的**嚴格局部極大值點**。
3.  若 $H$ 是不定的，則 $\mathbf{x}_0$ 既不是局部極小值點也不是局部極大值點（即為鞍點）。

*證明*：
由譜定理，對稱矩陣 $H$ 可正交對角化：$H = Q \Lambda Q^T$，其中 $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$。
$$ \mathbf{h}^T H \mathbf{h} = \mathbf{h}^T Q \Lambda Q^T \mathbf{h} = (Q^T \mathbf{h})^T \Lambda (Q^T \mathbf{h}) = \sum_{i=1}^n \lambda_i y_i^2 $$
其中 $\mathbf{y} = Q^T \mathbf{h}$。由於 $Q$ 是正交矩陣，$\|\mathbf{y}\| = \|\mathbf{h}\|$。

1.  **正定情況**：$\lambda_i > 0$ 對所有 $i$。設 $\lambda_{\min} = \min_i \lambda_i > 0$。
    $$ \frac{1}{2} \mathbf{h}^T H \mathbf{h} = \frac{1}{2} \sum \lambda_i y_i^2 \ge \frac{\lambda_{\min}}{2} \sum y_i^2 = \frac{\lambda_{\min}}{2} \|\mathbf{h}\|^2 $$
    由Taylor餘項性質，存在 $\delta_1 > 0$ 使得當 $\|\mathbf{h}\| < \delta_1$ 時，$|r(\mathbf{h})| < \frac{\lambda_{\min}}{4} \|\mathbf{h}\|^2$。
    取 $\delta = \min(\delta_1, \dots)$（考慮定義域開集限制），則對 $0 < \|\mathbf{h}\| < \delta$：
<<<NEW>>>
由於 $\mathbf{u}$ 任意，故 $H$ 半正定。局部極大推出半負定的情況類推。$\blacksquare$

**定理 14.3 (二階充分條件)**
設 $f \in C^2(U)$ 且 $\mathbf{x}_0$ 是駐點（$\nabla f(\mathbf{x}_0)=0$）。
1.  若 $H$ 是正定的，則 $\mathbf{x}_0$ 是 $f$ 的**嚴格局部極小值點**。
2.  若 $H$ 是負定的，則 $\mathbf{x}_0$ 是 $f$ 的**嚴格局部極大值點**。
3.  若 $H$ 是不定的，則 $\mathbf{x}_0$ 既不是局部極小值點也不是局部極大值點（即為鞍點）。

*證明*：
由譜定理，對稱矩陣 $H$ 可正交對角化：$H = Q \Lambda Q^T$，其中 $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$。
$$ \mathbf{h}^T H \mathbf{h} = \mathbf{h}^T Q \Lambda Q^T \mathbf{h} = (Q^T \mathbf{h})^T \Lambda (Q^T \mathbf{h}) = \sum_{i=1}^n \lambda_i y_i^2 $$
其中 $\mathbf{y} = Q^T \mathbf{h}$。由於 $Q$ 是正交矩陣，$\|\mathbf{y}\| = \|\mathbf{h}\|$。

1.  **正定情況**：$\lambda_i > 0$ 對所有 $i$。設 $\lambda_{\min} = \min_i \lambda_i > 0$。
    $$ \frac{1}{2} \mathbf{h}^T H \mathbf{h} = \frac{1}{2} \sum \lambda_i y_i^2 \ge \frac{\lambda_{\min}}{2} \sum y_i^2 = \frac{\lambda_{\min}}{2} \|\mathbf{h}\|^2 $$
    由Taylor餘項性質，存在 $\delta_1 > 0$ 使得當 $0<\|\mathbf{h}\| < \delta_1$ 時，$|r(\mathbf{h})| < \frac{\lambda_{\min}}{4} \|\mathbf{h}\|^2$。因 $U$ 開且 $\mathbf{x}_0\in U$，可取 $\rho>0$ 使 $B(\mathbf{x}_0,\rho)\subset U$。
    取 $\delta = \min(\delta_1,\rho)$，則對 $0 < \|\mathbf{h}\| < \delta$：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    令 $a = q(\mathbf{u})/2 > 0$，$b = -q(\mathbf{v})/2 > 0$。
    由 $r(\mathbf{h})/ \|\mathbf{h}\|^2 \to 0$，存在 $\delta > 0$ 使得當 $\|\mathbf{h}\| < \delta$ 時，$|r(\mathbf{h})| < \frac{1}{2} \min(a, b) \|\mathbf{h}\|^2 / \|\mathbf{h}\|^2$? 不，更精確地：
    對於 $\mathbf{h}_1 = t \frac{\mathbf{u}}{\|\mathbf{u}\|}$，$\mathbf{h}_1^T H \mathbf{h}_1 = t^2 \frac{\mathbf{u}^T H \mathbf{u}}{\|\mathbf{u}\|^2} = c_1 t^2$，其中 $c_1 > 0$。
    $f(\mathbf{x}_0 + \mathbf{h}_1) - f(\mathbf{x}_0) = \frac{1}{2} c_1 t^2 + r(\mathbf{h}_1)$。
    因為 $\lim_{t\to 0} \frac{r(t \mathbf{u}/\|\mathbf{u}\|)}{t^2} = 0$，存在 $t_1$ 足夠小使得 $|r(\mathbf{h}_1)| < \frac{1}{4} c_1 t^2$。
    於是 $f(\mathbf{x}_0 + \mathbf{h}_1) - f(\mathbf{x}_0) > \frac{1}{4} c_1 t^2 > 0$。
    同理，對於 $\mathbf{h}_2 = s \frac{\mathbf{v}}{\|\mathbf{v}\|}$，$\mathbf{h}_2^T H \mathbf{h}_2 = c_2 s^2$ 其中 $c_2 < 0$。
    存在 $s$ 足夠小使得 $f(\mathbf{x}_0 + \mathbf{h}_2) - f(\mathbf{x}_0) < 0$。
<<<NEW>>>
    令 $\hat{\mathbf{u}}=\mathbf{u}/\|\mathbf{u}\|$、$\hat{\mathbf{v}}=\mathbf{v}/\|\mathbf{v}\|$，並設 $c_1=\hat{\mathbf{u}}^TH\hat{\mathbf{u}}>0$、$c_2=\hat{\mathbf{v}}^TH\hat{\mathbf{v}}<0$。沿這兩個單位方向，餘項性質分別給出 $r(t\hat{\mathbf{u}})/t^2\to0$ 與 $r(s\hat{\mathbf{v}})/s^2\to0$。因此，對所有充分小的非零 $t$，有 $|r(t\hat{\mathbf{u}})|<c_1t^2/4$，從而
    $$f(\mathbf{x}_0+t\hat{\mathbf{u}})-f(\mathbf{x}_0)>\frac{c_1}{4}t^2>0.$$
    同理，對所有充分小的非零 $s$，有 $|r(s\hat{\mathbf{v}})|<(-c_2)s^2/4$，從而
    $$f(\mathbf{x}_0+s\hat{\mathbf{v}})-f(\mathbf{x}_0)<\frac{c_2}{4}s^2<0.$$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    # 4. 特徵值
    eigvals = np.linalg.eigvalsh(H_sym)
    
    if np.all(eigvals > tol):
        return "positive definite"
    elif np.all(eigvals < -tol):
        return "negative definite"
    elif np.all(eigvals >= -tol) and not np.all(eigvals > tol):
        return "semi-positive definite (degenerate)"
    elif np.all(eigvals <= tol) and not np.all(eigvals < -tol):
        return "semi-negative definite (degenerate)"
    else:
        return "indefinite (saddle point)"
<<<NEW>>>
    # 4. 特徵值；容差帶內的符號不能作為精確半定性的證明
    eigvals = np.linalg.eigvalsh(H_sym)
    eig_tol = tol * max(1.0, np.linalg.norm(H_sym, ord=2))
    if np.any(np.abs(eigvals) <= eig_tol):
        if np.any(eigvals > 0) and np.any(eigvals < 0):
            return "numerically indeterminate (opposite signs near zero)"
        return "numerically indeterminate (eigenvalue near zero)"
    if np.all(eigvals > 0):
        return "positive definite"
    if np.all(eigvals < 0):
        return "negative definite"
    return "indefinite (saddle point)"
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    assert "Indeterminate" in res2, f"Expected Indeterminate, got {res2}"

    # Case 3: Positive Definite
<<<NEW>>>
    assert "Indeterminate" in res2, f"Expected Indeterminate, got {res2}"

    # 故障輸入與小尺度異號特徵值：以下均為預期斷言，未執行
    bad = classify_hessian(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert bad.startswith("error: Hessian is not symmetric")
    tiny = classify_hessian(np.diag([-5e-9, 5e-9]))
    assert tiny.startswith("numerically indeterminate")

    # Case 3: Positive Definite
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
為了示範**退化極大值**，考慮：
$$ G(\Delta O_2, \Delta T) = - (\Delta O_2)^2 - c (\Delta T)^4, \quad c > 0 $$
1.  **駐點**：$(0,0)$。
2.  **Hessian**：
    $\frac{\partial^2 G}{\partial (\Delta O_2)^2} = -2$，$\frac{\partial^2 G}{\partial (\Delta T)^2} = 0$（在 0 處）。
    $H = \begin{pmatrix} -2 & 0 \\ 0 & 0 \end{pmatrix}$。
3.  **判斷**：
    $H$ 是半負定（特徵值 $-2, 0$）。
    二階測試無法直接判斷極大。
    分析高階項：$G \approx - (\Delta O_2)^2 - c (\Delta T)^4$。
    由於 $- (\Delta O_2)^2 \le 0$ 且 $-c (\Delta T)^4 \le 0$，故 $G \le G(0,0)$。
    因此 $(0,0)$ 是**局部極大值**（非嚴格，因為沿 $\Delta O_2=0$ 方向四階下降，但仍是極大）。
<<<NEW>>>
為了示範**退化極大值**，先取溶氧偏差 $\Delta O_2$（SI單位 $\mathrm{kg\,m^{-3}}$）及水溫偏差 $\Delta T$（SI單位 $\mathrm K$）。選定正尺度 $O_s$（$\mathrm{kg\,m^{-3}}$）、$T_s$（$\mathrm K$），定義無因次變量 $u=\Delta O_2/O_s$、$v=\Delta T/T_s$。令合成生長率基準 $G_0$ 及正係數 $a,b$ 具有與 $G$ 相同的單位，例如 $\mathrm{s^{-1}}$，並考慮：
$$ G(u,v)=G_0-au^2-bv^4,\qquad a,b>0. $$
1.  **駐點**：$(u,v)=(0,0)$，對應物理偏差 $(\Delta O_2,\Delta T)=(0,0)$；物理量可由 $\Delta O_2=O_su$、$\Delta T=T_sv$ 返回。
2.  **Hessian**：在無因次座標下，
    $$H(0,0)=\begin{pmatrix}-2a&0\\0&0\end{pmatrix}.$$
3.  **判斷**：
    $H$ 半負定，二階測試無法單獨判斷極大。但此模型的完整式子給出 $G(u,v)-G_0=-au^2-bv^4$；任意非零 $(u,v)$ 都使差值嚴格小於零。因此原點是**嚴格局部極大值，實際也是此模型在整個 $\mathbb R^2$ 上的嚴格全局極大值**。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
    且 $f(x_0 + v) = f(x_0) + v^T Q v$（因 $Qx_0=0$）。
<<<NEW>>>
    且 $f(x_0 + v) = f(x_0) + \frac{1}{2}v^T Q v$（因 $Qx_0=0$）。
<<<END>>>