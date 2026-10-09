<<<PATCH 01>>>
<<<OLD>>>
$K_c$是緊集，且$K_c\subset B_s(x_\ast)\subset D$，因為球面上的$V$至少為$m_s>c$。若初值滿足$V(x_0)<c$，則在解仍存在時，
<<<NEW>>>
$K_c$是閉子水平集與緊球的交集，故為緊集；球面上的$V$至少為$m_s>c$，所以$K_c\subset B_s(x_\ast)\subset D$。由$V(x_\ast)=0$及連續性，可令初值充分接近$x_\ast$，使$x_0\in B_s(x_\ast)$且$V(x_0)<c$。在解仍存在並留於球內時，
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
所以軌跡不能碰到$K_c$的外側邊界，也不能穿越球面$S_s$。因此軌跡留在$D$內的緊集$K_c$，由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。

若$\dot V<0$，再對任意不含$x_\ast$的緊環帶利用$\dot V$的嚴格負上界，可排除軌跡長時間停留在該環帶，從而得到$x(t)\to x_\ast$。這是負定導數推出局部漸近穩定的標準緊緻性論證。
<<<NEW>>>
若軌跡離開球，連續性使它必先碰到球面$S_s$；但該處$V\ge m_s>c$，與上式矛盾。因此軌跡始終留在緊集$K_c\subset D$，由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。

若$\dot V<0$，對不變緊集$K_c$內與$x_\ast$距離至少為$\varepsilon>0$的部分，連續的$\dot V$有嚴格負的最大值$-\eta$（若該部分非空）。軌跡若始終留在此處，便有$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。軌跡進入較小球後，再以該球面上$V$的正最小值選取不變子水平集，可排除其日後離開指定鄰域；由$\varepsilon$的任意性得到$x(t)\to x_\ast$。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    p = np.linalg.solve(K, rhs)
    P = p.reshape((n, n), order="F")
    return 0.5 * (P + P.T)
<<<NEW>>>
    p = np.linalg.solve(K, rhs)
    P = p.reshape((n, n), order="F")
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("求解結果的對稱誤差過大")
    if not np.allclose(A.T @ P + P @ A, -Q,
                       rtol=1e-10, atol=1e-12):
        raise ArithmeticError("求解結果的 Lyapunov 殘差過大")
    P = 0.5 * (P + P.T)
    if not np.allclose(A.T @ P + P @ A, -Q,
                       rtol=1e-10, atol=1e-12):
        raise ArithmeticError("對稱化後的 Lyapunov 殘差過大")
    return P
<<<END>>>