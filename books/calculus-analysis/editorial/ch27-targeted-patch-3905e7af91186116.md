<<<PATCH 27>>>
<<<OLD>>>
再選$0<s<r$。有限維Heine–Borel定理保證球面
<<<NEW>>>
為證明對任意指定的$\varepsilon>0$均穩定，選$0<s<\min\{r,\varepsilon\}$。有限維Heine–Borel定理保證球面
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
若$\dot V<0$，對不變緊集$K_c$內與$x_\ast$距離至少為$\varepsilon>0$的部分，連續的$\dot V$有嚴格負的最大值$-\eta$（若該部分非空）。軌跡若始終留在此處，便有$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。軌跡進入較小球後，再以該球面上$V$的正最小值選取不變子水平集，可排除其日後離開指定鄰域；由$\varepsilon$的任意性得到$x(t)\to x_\ast$。
<<<NEW>>>
若$\dot V<0$，由$V(x(t))$不增且非負，存在極限$L\ge0$。若$L>0$，軌跡始終位於緊集$\{x\in K_c:V(x)\ge L\}$；此集不含$x_\ast$，故連續的$\dot V$在其上有嚴格負的最大值$-\eta$。於是$V(x(t))\le V(x_0)-\eta t$，矛盾。因此$L=0$。對任意$0<\varepsilon<s$，緊集$\{x\in K_c:\|x-x_\ast\|_2\ge\varepsilon\}$若非空，$V$在其上的最小值嚴格為正；既然$V(x(t))\to0$，軌跡最終不可能落在此集內，故$x(t)\to x_\ast$。
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
def numerical_report(A, P, Q, x0, dt, final_time):
    if final_time < 0:
        raise ValueError("final_time 必須非負")
    steps = int(round(final_time / dt))
<<<NEW>>>
def numerical_report(A, P, Q, x0, dt, final_time):
    if dt <= 0:
        raise ValueError("dt 必須為正")
    if final_time < 0:
        raise ValueError("final_time 必須非負")
    steps = int(round(final_time / dt))
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
步長掃描可能顯示較小步長下的離散結果更接近連續軌跡，但本章未執行程式，故不宣稱具體數值或觀察到的收斂階。
<<<NEW>>>
步長掃描可能顯示較小步長下的離散結果更接近連續軌跡，但本章未執行程式，故不宣稱具體數值或觀察到的收斂階。程式以`round(final_time / dt)`決定步數；若終止時間不是步長的整數倍，實際末端時間為`steps * dt`，不一定等於要求的`final_time`。
<<<END>>>