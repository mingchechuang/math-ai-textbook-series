<<<PATCH 01>>>
<<<OLD>>>
另一個常見誤解是把暫態放大當成不穩定。即使$A$為Hurwitz，Euclidean長度$\|e^{tA}z_0\|_2$仍可能先增加再減少。非正規矩陣的特徵方向可能高度非正交，使不同模態短時間疊加。適當的Lyapunov函數則提供另一種加權幾何，使對應橢球半徑單調下降。
<<<NEW>>>
另一個常見誤解是把暫態放大當成不穩定。即使$A$為Hurwitz，Euclidean長度$\|e^{tA}z_0\|_2$仍可能先增加再減少。非正規矩陣的特徵方向可能高度非正交，使不同模態短時間疊加。適當的正定二次Lyapunov函數則提供另一種加權幾何，使其橢球子水平集沿軌跡向內收縮；一般Lyapunov函數的子水平集不必是橢球。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
假設$f$局部Lipschitz，且存在開鄰域$D$與$V\in C^1(D)$，使

$$
V(x_\ast)=0,\qquad V(x)>0\quad(x\ne x_\ast),
$$

以及

$$
\dot V(x)\le0.
$$

則$x_\ast$穩定。若另外有

$$
\dot V(x)<0\qquad(x\ne x_\ast),
$$

則$x_\ast$局部漸近穩定。
<<<NEW>>>
假設$f$局部Lipschitz，且存在開鄰域$D$與$V\in C^1(D)$，使

$$
V(x_\ast)=0,\qquad V(x)>0\quad(x\in D\setminus\{x_\ast\}),
$$

以及

$$
\dot V(x)\le0\qquad(x\in D).
$$

則$x_\ast$穩定。若另外有

$$
\dot V(x)<0\qquad(x\in D\setminus\{x_\ast\}),
$$

則$x_\ast$局部漸近穩定。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
$K_c$是閉子水平集與緊球的交集，故為緊集；球面上的$V$至少為$m_s>c$，所以$K_c\subset B_s(x_\ast)\subset D$。由$V(x_\ast)=0$及連續性，可令初值充分接近$x_\ast$，使$x_0\in B_s(x_\ast)$且$V(x_0)<c$。在解仍存在並留於球內時，
<<<NEW>>>
$K_c$是閉子水平集與緊球的交集，故為緊集；球面上的$V$至少為$m_s>c$，所以$K_c\subset B_s(x_\ast)\subset D$。由$V(x_\ast)=0$及連續性，可取$0<\delta<s$，使$\|x_0-x_\ast\|_2<\delta$同時保證$x_0\in B_s(x_\ast)$及$V(x_0)<c$。先前已選$s<\varepsilon$，因此這個$\delta$適用於指定的$\varepsilon$。在解仍存在並留於球內時，
<<<END>>>