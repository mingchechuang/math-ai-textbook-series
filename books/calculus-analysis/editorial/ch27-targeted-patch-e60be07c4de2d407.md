<<<PATCH 01>>>
<<<OLD>>>
$K_c$是緊集，且$K_c\subset B_s(x_\ast)\subset D$，因為球面上的$V$至少為$m_s>c$。若初值滿足$V(x_0)<c$，則在解仍存在時，

$$
V(x(t))\le V(x_0)<c,
$$

所以軌跡不能碰到$K_c$的外側邊界，也不能穿越球面$S_s$。因此軌跡留在$D$內的緊集$K_c$，由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。

若$\dot V<0$，再對任意不含$x_\ast$的緊環帶利用$\dot V$的嚴格負上界，可排除軌跡長時間停留在該環帶，從而得到$x(t)\to x_\ast$。這是負定導數推出局部漸近穩定的標準緊緻性論證。
<<<NEW>>>
$K_c$是閉集與緊球$\overline B_s(x_\ast)$的交集，故為緊集；又因球面上的$V$至少為$m_s>c$，有$K_c\subset B_s(x_\ast)\subset D$。由$V(x_\ast)=0$及連續性，可選初值充分接近$x_\ast$，使$x_0\in B_s(x_\ast)$且$V(x_0)<c$。在解仍存在並留於該球內時，

$$
V(x(t))\le V(x_0)<c.
$$

若連續軌跡離開球，必先碰到球面$S_s$，但球面上$V\ge m_s>c$，與上式矛盾。因此軌跡始終留在$K_c$；由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。

若$\dot V<0$，對任意$0<\varepsilon<s$，考慮已建立的不變緊集$K_c$中與$x_\ast$距離至少為$\varepsilon$的部分。若它非空，$\dot V$在其上的最大值為$-\eta<0$。假使軌跡始終不進入$B_\varepsilon(x_\ast)$，便有$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。軌跡一旦進入足夠小的子水平集，又不能穿越對應球面；對每個$\varepsilon$如此選取，即得$x(t)\to x_\ast$。這是負定導數推出局部漸近穩定的緊緻性論證。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    p = np.linalg.solve(K, rhs)
    P = p.reshape((n, n), order="F")
    return 0.5 * (P + P.T)
<<<NEW>>>
    p = np.linalg.solve(K, rhs)
    P = p.reshape((n, n), order="F")
    scale = max(1.0, np.linalg.norm(Q, ord=np.inf))
    if np.linalg.norm(P - P.T, ord=np.inf) > 1e-10 * max(
        1.0, np.linalg.norm(P, ord=np.inf)
    ):
        raise ArithmeticError("求解結果的對稱誤差過大")
    if np.linalg.norm(A.T @ P + P @ A + Q, ord=np.inf) > 1e-10 * scale:
        raise ArithmeticError("求解結果的 Lyapunov 殘差過大")
    P = 0.5 * (P + P.T)
    if np.linalg.norm(A.T @ P + P @ A + Q, ord=np.inf) > 1e-10 * scale:
        raise ArithmeticError("對稱化後的 Lyapunov 殘差過大")
    return P
<<<END>>>