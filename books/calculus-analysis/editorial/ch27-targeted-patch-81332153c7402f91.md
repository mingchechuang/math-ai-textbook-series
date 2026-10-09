<<<PATCH 01>>>
<<<OLD>>>
適當的Lyapunov函數則提供另一種加權幾何，使對應橢球半徑單調下降。
<<<NEW>>>
適當的正定二次Lyapunov函數則提供另一種加權幾何，使其橢球子水平集沿軌跡向內收縮；一般Lyapunov函數的子水平集不必是橢球。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
V(x_\ast)=0,\qquad V(x)>0\quad(x\ne x_\ast),
<<<NEW>>>
V(x_\ast)=0,\qquad V(x)>0\quad(x\in D\setminus\{x_\ast\}),
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
由$V(x_\ast)=0$及連續性，可令初值充分接近$x_\ast$，使$x_0\in B_s(x_\ast)$且$V(x_0)<c$。
<<<NEW>>>
由$V(x_\ast)=0$及連續性，可取$0<\delta<s$，使$\|x_0-x_\ast\|_2<\delta$同時保證$x_0\in B_s(x_\ast)$及$V(x_0)<c$。因先前已選$s<\varepsilon$，這也明確給出穩定性定義所需的$\delta$。
<<<END>>>