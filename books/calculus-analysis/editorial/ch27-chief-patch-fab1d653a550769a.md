<<<PATCH 01>>>
<<<OLD>>>
接著保持矩陣乘法次序。因$A^T$與其矩陣指數$e^{A^Tt}$交換，且$A$與$e^{At}$交換，

$$
A^TP+PA
=
\int_0^\infty
\left(
A^Te^{A^Tt}Qe^{At}
+
e^{A^Tt}Qe^{At}A
\right)\,dt.
$$

另一方面，

$$
\frac{d}{dt}
\left(e^{A^Tt}Qe^{At}\right)
=
A^Te^{A^Tt}Qe^{At}
+
e^{A^Tt}Qe^{At}A.
$$

等價地，第一項也可寫成$e^{A^Tt}A^TQe^{At}$，第二項也可寫成$e^{A^Tt}QAe^{At}$；這只使用矩陣與其自身指數交換，並未假設$A$與$Q$交換。因此

$$
A^TP+PA
=
\left[e^{A^Tt}Qe^{At}\right]_{0}^{\infty}.
$$

Hurwitz條件使上限趨於零，而$t=0$時矩陣為$Q$，故

$$
A^TP+PA=-Q.
$$
<<<NEW>>>
接著保持矩陣乘法次序。記$F(t)=e^{A^Tt}Qe^{At}$。因$A^T$與$e^{A^Tt}$交換，且$A$與$e^{At}$交換，

$$
A^TP+PA
=
\int_0^\infty
\left(
A^T F(t)+F(t)A
\right)\,dt.
$$

另一方面，乘積法則與交換性給出

$$
F'(t)
=
A^Te^{A^Tt}Qe^{At}
+
e^{A^Tt}Qe^{At}A
=
A^T F(t)+F(t)A.
$$

因此被積函數恰為$F'(t)$，所以

$$
A^TP+PA
=
\int_0^\infty F'(t)\,dt
=
\left[F(t)\right]_{0}^{\infty}.
$$

Hurwitz條件使上限趨於零，而$F(0)=Q$，故

$$
A^TP+PA=-Q.
$$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
第二點可由複特徵向量驗證。若$Av=\lambda v$且$v\ne0$，則

$$
v^\ast(A^TP+PA)v
=
2\operatorname{Re}(\lambda)\,v^\ast Pv
=
-v^\ast Qv<0.
$$

因$v^\ast Pv>0$，必有$\operatorname{Re}\lambda<0$。這是必要充分判準，而不是有限採樣所得的經驗規則。
<<<NEW>>>
第二點可由複特徵向量驗證。若$Av=\lambda v$且$v\ne0$，由$v^\ast A^T=(Av)^\ast=\bar\lambda v^\ast$及$P,Q$實對稱，

$$
v^\ast A^TPv=\bar\lambda\,v^\ast Pv,\qquad
v^\ast PAv=\lambda\,v^\ast Pv.
$$

相加得

$$
v^\ast(A^TP+PA)v
=
2\operatorname{Re}(\lambda)\,v^\ast Pv
=
-v^\ast Qv<0.
$$

因$v^\ast Pv>0$，必有$\operatorname{Re}\lambda<0$。這是必要充分判準，而不是有限採樣所得的經驗規則。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
所以軌跡不能碰到$K_c$的外側邊界，也不能穿越球面$S_s$。因此軌跡留在$D$內的緊集$K_c$，由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。
<<<NEW>>>
所以軌跡不能碰到$K_c$的外側邊界。又因$S_s$上$V\ge m_s>c$與$V(x(t))\le V(x_0)<c$矛盾，連續軌跡若離開$B_s(x_\ast)$必先穿越$S_s$，故它不會離開$B_s(x_\ast)$。於是軌跡留在$K_c$；$K_c$是閉集與緊球$\overline B_s(x_\ast)$的交集，故為緊集。由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
因此$0<|x|<\sqrt2$時$\dot V<0$。若縮小到$|x|\le1$，則

$$
\dot V\le-x^2=-2V.
$$

區間$[-1,1]$正向不變，故對$|x_0|<1$有

$$
V(t)\le e^{-2t}V(0),
\qquad
|x(t)|\le e^{-t}|x_0|.
$$
<<<NEW>>>
因此$0<|x|<\sqrt2$時$\dot V<0$。若縮小到$|x|\le1$，則

$$
\dot V\le-x^2=-2V.
$$

因在$|x|\le1$內$\dot V\le-2V$，$V$不增，所以$|x(t)|\le|x_0|<1$，軌跡留在該區間內。區間$[-1,1]$正向不變，故對$|x_0|<1$有

$$
V(t)\le e^{-2t}V(0),
\qquad
|x(t)|\le e^{-t}|x_0|.
$$
<<<END>>>