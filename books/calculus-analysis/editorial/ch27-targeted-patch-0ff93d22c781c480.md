<<<PATCH 27>>>
<<<OLD>>>
非正規矩陣的特徵方向可能高度非正交，使不同模態短時間疊加。適當的Lyapunov函數則提供另一種加權幾何，使對應橢球半徑單調下降。
<<<NEW>>>
非正規矩陣的特徵方向可能高度非正交，使不同模態短時間疊加。適當的正定二次Lyapunov函數則提供另一種加權幾何，使其橢球子水平集沿軌跡單調收縮。
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
設$U\subset\mathbb R^n$為開集，$f\in C^1(U,\mathbb R^n)$，且$f(x_\ast)=0$。
<<<NEW>>>
設$U\subset\mathbb R^n$為開集，$x_\ast\in U$，$f\in C^1(U,\mathbb R^n)$，且$f(x_\ast)=0$。因此$f$在平衡點的一個開鄰域內為$C^1$，鄰近初值的方程局部存在唯一解。
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
V(x_\ast)=0,\qquad V(x)>0\quad(x\ne x_\ast),
<<<NEW>>>
V(x_\ast)=0,\qquad V(x)>0\quad(x\in D\setminus\{x_\ast\}),
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
\dot V(x)<0\qquad(x\ne x_\ast),
<<<NEW>>>
\dot V(x)<0\qquad(x\in D\setminus\{x_\ast\}),
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
        "dt": dt,
        "eig_A": np.linalg.eigvals(A),
<<<NEW>>>
        "dt": dt,
        "actual_final_time": steps * dt,
        "eig_A": np.linalg.eigvals(A),
<<<END>>>