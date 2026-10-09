<<<PATCH 01>>>
<<<OLD>>>
令

$$
x=
\begin{pmatrix}
p\\q
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
\nu\\v
\end{pmatrix},
$$

並定義

$$
F(x,y)=
\begin{pmatrix}
\nu+v+p-3\\
\nu^2+2v-q-6
\end{pmatrix}.
$$
<<<NEW>>>
令

$$
x=
\begin{pmatrix}
p\\q
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
u\\v
\end{pmatrix},
$$

並定義

$$
F(x,y)=
\begin{pmatrix}
u+v+p-3\\
u^2+2v-q-6
\end{pmatrix}.
$$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
若 $(p,q)=\varepsilon(1,-2)$，則通過 $u=2$ 的分支恰有

$$
\nu=2,
\qquad
v=1-\varepsilon.
$$

所以此方向的解恰為線性，不適合用來觀察一般前向差分的截斷誤差。

### 例三：兩個候選分支與合併點

由上式可得

$$
\nu=1\pm\sqrt{1+2p+q}.
$$
<<<NEW>>>
若 $(p,q)=\varepsilon(1,-2)$，則通過 $u=2$ 的分支恰有

$$
u=2,
\qquad
v=1-\varepsilon.
$$

所以此方向的解恰為線性，不適合用來觀察一般前向差分的截斷誤差。

### 例三：兩個候選分支與合併點

由上式可得

$$
u=1\pm\sqrt{1+2p+q}.
$$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
沿 $d=(1,0)^T$ 的明確分支為

$$
\nu=1+\sqrt{1+2p},
\qquad
v=2-p-\sqrt{1+2p},
$$
<<<NEW>>>
沿 $d=(1,0)^T$ 的明確分支為

$$
u=1+\sqrt{1+2p},
\qquad
v=2-p-\sqrt{1+2p},
$$
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
令

$$
F(p,u,v)=
\begin{pmatrix}
\nu+v-p\\
\nu^2+v-2
\end{pmatrix}.
$$
<<<NEW>>>
令

$$
F(p,u,v)=
\begin{pmatrix}
u+v-p\\
u^2+v-2
\end{pmatrix}.
$$
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc/multivariable-calculus-fall-2010/>
<<<NEW>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
<<<END>>>