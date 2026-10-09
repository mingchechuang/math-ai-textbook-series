<<<PATCH 01>>>
<<<OLD>>>
還須明確處理鄰域：選取足夠小的 $V,W$，使 $V\times W\subset N$，並使 $(x,0)\in M$。若 $(x,y)\in V\times W$ 且 $F(x,y)=0$，則
<<<NEW>>>
還須明確處理鄰域：先取 $b$ 的足夠小的開鄰域 $W$，再利用 $\psi$ 的連續性選取 $a$ 的開鄰域 $V$，使 $V\times W\subset N$、對所有 $x\in V$ 均有 $(x,0)\in M$，且 $g(V)=\psi(V,0)\subset W$。若 $(x,y)\in V\times W$ 且 $F(x,y)=0$，則
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
當其他尺度固定時，

$$
2k_O(C-C_*)-\alpha\gamma
$$

接近零可作輔助警示，但不能取代奇異值分析。
<<<NEW>>>
此處未尺度化的行列式為

$$
\det(D_yF)=2k_O(C-C_*)-\alpha\gamma.
$$

它等於零可判定精確秩失效，但其數值大小受單位與尺度影響。若要把行列式大小用作輔助警示，應先固定並交代尺度，計算無因次矩陣 $S_F^{-1}D_yF S_y$ 的行列式；即使如此，仍不能以它取代最小奇異值或條件數分析。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc/multivariable-calculus-fall-2010/>
<<<NEW>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
<<<END>>>