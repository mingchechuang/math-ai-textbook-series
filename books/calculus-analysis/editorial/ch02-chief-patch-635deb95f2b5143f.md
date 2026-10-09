<<<PATCH 02>>>
<<<OLD>>>
也可構造發散延伸

$$
a_{N+k}=(-1)^k,\qquad k\ge1.
$$

兩者有完全相同的已觀察前綴，卻有不同的收斂行為。因此，有限資料本身不能決定任意未知無窮延伸是否收斂。
<<<NEW>>>
也可構造發散延伸

$$
a_{N+k}=(-1)^k,\qquad k\ge1.
$$

第二種延伸在 $N+1$ 項之後輪流取 $-1$ 與 $1$，因此含兩個分別趨向 $-1$ 與 $1$ 的常數子列，整列不收斂；有限前綴（包含 $N=0$ 的無前綴情形）不影響這個尾端振盪。兩者有完全相同的已觀察前綴，卻有不同的收斂行為。因此，有限資料本身不能決定任意未知無窮延伸是否收斂。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
由標準結果 $H_n-\log n\to\gamma$，

$$
T_N\to
\log(4N)-\frac12\log(2N)-\frac12\log N
=\frac32\log2.
$$
<<<NEW>>>
由標準結果 $H_n=\log n+\gamma+o(1)$，故

$$
\begin{aligned}
T_N
&=H_{4N}-\frac12H_{2N}-\frac12H_N\\
&=\left(\log(4N)+\gamma+o(1)\right)
-\frac12\left(\log(2N)+\gamma+o(1)\right)
-\frac12\left(\log N+\gamma+o(1)\right)\\
&=\frac32\log2+o(1).
\end{aligned}
$$

因此 $T_N\to\frac32\log2$。
<<<END>>>