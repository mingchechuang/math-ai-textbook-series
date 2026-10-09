<<<PATCH 12>>>
<<<OLD>>>
此唯一性是指：適當縮小 $V$ 與 $W$ 後，滿足 $g(a)=b$ 且其圖形位於 $V\times W$ 的局部分支唯一；更精確地，
<<<NEW>>>
此唯一性是指：在適當縮小並固定的 $V,W$ 上，滿足 $g(a)=b$ 且圖形位於 $V\times W$ 的分支唯一；更精確地，
<<<END>>>
<<<PATCH 12>>>
<<<OLD>>>
還須明確處理鄰域：選取足夠小的 $V,W$，使 $V\times W\subset N$，並使 $(x,0)\in M$。若 $(x,y)\in V\times W$ 且 $F(x,y)=0$，則
<<<NEW>>>
還須明確處理鄰域：先選取 $b$ 的足夠小的開鄰域 $W$，再利用 $g$ 的連續性選取 $a$ 的開鄰域 $V$，使 $V\times W\subset N$，對每個 $x\in V$ 都有 $(x,0)\in M$，並且 $g(V)\subset W$。若 $(x,y)\in V\times W$ 且 $F(x,y)=0$，則
<<<END>>>
<<<PATCH 12>>>
<<<OLD>>>
此外，

$$
Dg(x)
=
-\bigl(D_yF(x,g(x))\bigr)^{-1}D_xF(x,g(x)).
$$
<<<NEW>>>
由 $D_yF$ 的連續性及其在 $(a,b)$ 可逆，再縮小 $V$，可使 $D_yF(x,g(x))$ 對每個 $x\in V$ 仍可逆。此外，

$$
Dg(x)
=
-\bigl(D_yF(x,g(x))\bigr)^{-1}D_xF(x,g(x)).
$$
<<<END>>>
<<<PATCH 12>>>
<<<OLD>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc/multivariable-calculus-fall-2010/>
<<<NEW>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
<<<END>>>