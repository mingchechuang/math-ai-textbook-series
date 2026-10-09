<<<PATCH 01>>>
<<<OLD>>>
若$\dot V<0$，再對任意不含$x_\ast$的緊環帶利用$\dot V$的嚴格負上界，可排除軌跡長時間停留在該環帶，從而得到$x(t)\to x_\ast$。這是負定導數推出局部漸近穩定的標準緊緻性論證。
<<<NEW>>>
若$\dot V<0$，對已建立的不變緊集$K_c$中與$x_\ast$距離至少為$\varepsilon>0$的部分，$\dot V$有嚴格負的最大值$-\eta$（若該部分非空）。假使軌跡始終留在這部分，便有$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。再利用任意較小球面上的正最小值及子水平集不變性，可知軌跡進入充分小的鄰域後不會離開；因此$x(t)\to x_\ast$。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    return 0.5 * (P + P.T)
<<<NEW>>>
    if not np.allclose(P, P.T):
        raise ArithmeticError("求解結果的對稱誤差過大")
    if not np.allclose(A.T @ P + P @ A, -Q):
        raise ArithmeticError("求解結果的 Lyapunov 殘差過大")
    P = 0.5 * (P + P.T)
    if not np.allclose(A.T @ P + P @ A, -Q):
        raise ArithmeticError("對稱化後的 Lyapunov 殘差過大")
    return P
<<<END>>>