<<<PATCH 27>>>
<<<OLD>>>
若$\dot V<0$，由$V(x(t))$不增且非負，存在極限$L\ge0$。若$L>0$，軌跡始終位於緊集$\{x\in K_c:V(x)\ge L\}$；此集不含$x_\ast$，故連續的$\dot V$在其上有嚴格負的最大值$-\eta$。於是$V(x(t))\le V(x_0)-\eta t$，矛盾。
<<<NEW>>>
若$\dot V<0$，由$V(x(t))$不增且非負，存在極限$L\ge0$。若$L>0$，取$0<a<L$。因不增函數的值不小於其極限，軌跡始終位於緊集$\{x\in K_c:V(x)\ge a\}$；此集不含$x_\ast$，故連續的$\dot V$在其上有嚴格負的最大值$-\eta$。於是$V(x(t))\le V(x_0)-\eta t$，矛盾。
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
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
<<<NEW>>>
另一方面，先依乘積法則按原次序求導：

$$
\frac{d}{dt}
\left(e^{A^Tt}Qe^{At}\right)
=
e^{A^Tt}A^TQe^{At}
+
e^{A^Tt}QAe^{At}.
$$

第一項利用$e^{A^Tt}A^T=A^Te^{A^Tt}$，第二項利用$Ae^{At}=e^{At}A$，便分別等於上式積分中的兩項；兩次交換都沒有讓矩陣跨越$Q$。因此
<<<END>>>