<<<PATCH 01>>>
<<<OLD>>>
若$\dot V<0$，由$V(x(t))$不增且非負，存在極限$L\ge0$。若$L>0$，軌跡始終位於緊集$\{x\in K_c:V(x)\ge L\}$；此集不含$x_\ast$，故連續的$\dot V$在其上有嚴格負的最大值$-\eta$。於是$V(x(t))\le V(x_0)-\eta t$，矛盾。因此$L=0$。對任意$0<\varepsilon<s$，緊集$\{x\in K_c:\|x-x_\ast\|_2\ge\varepsilon\}$若非空，$V$在其上的最小值嚴格為正；既然$V(x(t))\to0$，軌跡最終不可能落在此集內，故$x(t)\to x_\ast$。
<<<NEW>>>
若$\dot V<0$，由$V(x(t))$不增且非負，存在極限$L\ge0$。若$L>0$，取$0<a<L$。因$V(x(t))$單調不增並趨於$L$，每個$t\ge0$均有$V(x(t))\ge L>a$，故軌跡始終位於緊集$\{x\in K_c:V(x)\ge a\}$。此集不含$x_\ast$，連續的$\dot V$在其上有嚴格負的最大值$-\eta$。於是$V(x(t))\le V(x_0)-\eta t$，與$V\ge0$矛盾。因此$L=0$。對任意$0<\varepsilon<s$，緊集$\{x\in K_c:\|x-x_\ast\|_2\ge\varepsilon\}$若非空，$V$在其上的最小值嚴格為正；既然$V(x(t))\to0$，軌跡最終不可能落在此集內，故$x(t)\to x_\ast$。
<<<END>>>