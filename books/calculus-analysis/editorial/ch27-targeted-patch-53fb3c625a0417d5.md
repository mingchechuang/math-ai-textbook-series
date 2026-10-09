<<<PATCH 01>>>
<<<OLD>>>
設$U\subset\mathbb R^n$為開集，$f\in C^1(U,\mathbb R^n)$，且$f(x_\ast)=0$。
<<<NEW>>>
設$U\subset\mathbb R^n$為開集，$x_\ast\in U$，$f\in C^1(U,\mathbb R^n)$，且$f(x_\ast)=0$。此處的$C^1$假設特別保證$f$在平衡點的一個開鄰域內局部Lipschitz，故鄰近初值有局部唯一解。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
若$\dot V<0$，對不變緊集$K_c$內與$x_\ast$距離至少為$\varepsilon>0$的部分，連續的$\dot V$有嚴格負的最大值$-\eta$（若該部分非空）。軌跡若始終留在此處，便有$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。軌跡進入較小球後，再以該球面上$V$的正最小值選取不變子水平集，可排除其日後離開指定鄰域；由$\varepsilon$的任意性得到$x(t)\to x_\ast$。
<<<NEW>>>
若$\dot V<0$，還須證明軌跡不只偶爾靠近原點。任取$0<\varepsilon<s$，在先前正向不變的$K_c$內，取$m_\varepsilon=\min_{\|x-x_\ast\|_2=\varepsilon}V(x)>0$，並選$0<d<\min\{m_\varepsilon,c\}$。集合$K_d=\{x\in\overline B_s(x_\ast):V(x)\le d\}$是正向不變的緊集：軌跡無法穿過外球面，且$V$沿解不增。它也完全落在$B_\varepsilon(x_\ast)$內；否則從$x_\ast$到集合中某點的線段未必仍在集合中，所以不能只靠球面最小值直接推斷這點。正確做法是把目標集合改取為$K_d$中包含$x_\ast$的連通分支，或更直接地利用軌跡：一旦軌跡在$\varepsilon$球內且$V<d$，便不能再次穿過其球面。

證明這樣的時刻必然出現：在緊集$K_c\setminus B_\varepsilon(x_\ast)$上，若非空，連續函數$\dot V$有嚴格負的最大值$-\eta$。若軌跡始終不進入$\varepsilon$球，則$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。不過首次進球時未必有$V<d$。為補足此步，改在緊集$K_c\cap\{V\ge d\}$上取$\dot V$的嚴格負上界$-\eta_d$（非空時）；若軌跡永不進入$V<d$，同樣導致矛盾。因此它有限時間內到達$V<d$，此後因$V$不增而保持$V<d$。到達時若仍在$\varepsilon$球外，則繼續利用$K_c\setminus B_\varepsilon$上的下降界，可知它有限時間內進球；進球後又不能穿過$V\ge m_\varepsilon>d$的球面。故軌跡最終留在任意指定的$\varepsilon$球內，即$x(t)\to x_\ast$。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    if final_time < 0:
        raise ValueError("final_time 必須非負")
    steps = int(round(final_time / dt))
<<<NEW>>>
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt 必須是正的有限數")
    if not np.isfinite(final_time) or final_time < 0:
        raise ValueError("final_time 必須是非負的有限數")
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time, rtol=1e-12, atol=1e-12):
        raise ValueError("final_time 必須是 dt 的整數倍")
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("求解結果的對稱誤差過大")
    if not np.allclose(A.T @ P + P @ A, -Q,
                       rtol=1e-10, atol=1e-12):
        raise ArithmeticError("求解結果的 Lyapunov 殘差過大")
    P = 0.5 * (P + P.T)
    if not np.allclose(A.T @ P + P @ A, -Q,
                       rtol=1e-10, atol=1e-12):
        raise ArithmeticError("對稱化後的 Lyapunov 殘差過大")
<<<NEW>>>
    # 以下容差僅作本章小型、適度縮放例子的示範檢查。
    # 同時報出尺度化誤差，不能用通過檢查推論病態問題也可靠。
    symmetry_error = np.linalg.norm(P - P.T, ord=np.inf)
    symmetry_scale = max(np.linalg.norm(P, ord=np.inf), 1e-300)
    if symmetry_error / symmetry_scale > 1e-10:
        raise ArithmeticError("求解結果的相對對稱誤差過大")
    residual = A.T @ P + P @ A + Q
    residual_scale = max(
        np.linalg.norm(A, ord=np.inf) * np.linalg.norm(P, ord=np.inf)
        + np.linalg.norm(Q, ord=np.inf), 1e-300
    )
    if np.linalg.norm(residual, ord=np.inf) / residual_scale > 1e-10:
        raise ArithmeticError("求解結果的尺度化 Lyapunov 殘差過大")
    P = 0.5 * (P + P.T)
    residual = A.T @ P + P @ A + Q
    if np.linalg.norm(residual, ord=np.inf) / residual_scale > 1e-10:
        raise ArithmeticError("對稱化後的尺度化 Lyapunov 殘差過大")
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
一般有限步長RK4不保證保存連續系統的Lyapunov單調性。因此程式不斷言離散$V_k$必定下降，而只輸出`max_discrete_V_increment`。即使該值非正，也只支持指定步長、初值與時間窗內的觀察。
<<<NEW>>>
一般有限步長RK4不保證保存連續系統的Lyapunov單調性。因此程式不斷言離散$V_k$必定下降，而只輸出`max_discrete_V_increment`。即使該值非正，也只支持指定步長、初值與時間窗內的觀察。`numerical_report`要求`final_time`為`dt`的整數倍，避免把四捨五入後實際積分到的時刻誤稱為指定終點；其他終點可另行實作較短的最後一步。求解器的尺度化殘差是代數方程的後向誤差指標，不是$P$的前向誤差保證；接近奇異的Kronecker系統仍可能使解高度敏感。
<<<END>>>