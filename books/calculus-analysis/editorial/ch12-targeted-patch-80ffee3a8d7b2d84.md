<<<PATCH 01>>>
<<<OLD>>>
包含該端點的完整圓弧不能在此寫成定義於 $x=1$ 之開區間上的可微函數 $y=g(x)$。這不是只說單側公式的導數變大，而是說任何以 $x$ 為局部座標的圖形表示，都不能同時涵蓋端點兩側的完整曲線鄰域。
<<<NEW>>>
上半弧在 $x<1$ 一側仍可寫成 $y=\sqrt{1-x^2}$，但不能延伸為定義於包含 $x=1$ 的開區間上的實值隱函數分支：該開區間必含 $x>1$，而圓在那一側沒有實數解。包含端點的完整曲線鄰域則可改用 $y$ 作局部座標。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
還須明確處理鄰域：選取足夠小的 $V,W$，使 $V\times W\subset N$，並使 $(x,0)\in M$。
<<<NEW>>>
還須明確處理鄰域：先取足夠小的 $W$，再利用 $\psi$ 的連續性縮小 $V$，使 $V\times W\subset N$、$(x,0)\in M$，且 $g(V)=\psi(V,0)\subset W$。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc/multivariable-calculus-fall-2010/>
<<<NEW>>>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
<<<END>>>