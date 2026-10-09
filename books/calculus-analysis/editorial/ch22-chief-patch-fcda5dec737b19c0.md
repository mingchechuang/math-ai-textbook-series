<<<PATCH 22>>>
<<<OLD>>>
設 $U\subset\mathbb R^n$ 為開集。若存在 $\phi\in C^1(U)$ 使
<<<NEW>>>
設 $U\subset\mathbb R^n$ 為開集（本章所稱的「域」通常指開且連通的集合）。若存在 $\phi\in C^1(U)$ 使
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
設 $U\subset\mathbb R^n$ 為開且路徑連通的集合，$F:U\to\mathbb R^n$ 連續。下列敘述等價：
<<<NEW>>>
設 $U\subset\mathbb R^n$ 為開且連通的集合，$F:U\to\mathbb R^n$ 連續。下列敘述等價：
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
最後假設路徑獨立。固定 $x_0\in U$，定義

$$
\phi(x)
=
\int_{x_0}^{x}F\cdot dr.
$$

因路徑獨立，這個值不依賴所選路徑。
<<<NEW>>>
最後假設路徑獨立。固定 $x_0\in U$。因 $U$ 為開且連通，任兩點都可用域內分段 $C^1$ 路徑連接；定義

$$
\phi(x)
=
\int_{x_0}^{x}F\cdot dr.
$$

因路徑獨立，這個值不依賴所選路徑。
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
$$
\nabla_{(u,v)}\Phi
=
\begin{pmatrix}
\nu+v\\
\nu+4v
\end{pmatrix}.
$$
<<<NEW>>>
$$
\nabla_{(u,v)}\Phi
=
\begin{pmatrix}
u+v\\
u+4v
\end{pmatrix}.
$$
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
    assert_raises(
        ValueError,
        lambda: line_integral(
            vortex_field,
            curve,
            dcurve,
            0.0,
            2.0 * np.pi,
            n=1
        ),
        "too few panels"
    )

    # 故障測試：曲線在 t=pi 通過原點。
<<<NEW>>>
    assert_raises(
        ValueError,
        lambda: line_integral(
            vortex_field,
            curve,
            dcurve,
            0.0,
            2.0 * np.pi,
            n=1
        ),
        "too few panels"
    )

    # 故障測試：回傳錯誤 shape 的場應丟出 ValueError。
    def bad_field(points):
        return np.zeros((points.shape[0], 3))

    assert_raises(
        ValueError,
        lambda: line_integral(
            bad_field,
            curve,
            dcurve,
            0.0,
            2.0 * np.pi
        ),
        "bad field shape"
    )

    # 故障測試：曲線在 t=pi 通過原點。
<<<END>>>