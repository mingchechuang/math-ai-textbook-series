<<<PATCH 22>>>
<<<OLD>>>
為開且路徑連通的集合
<<<NEW>>>
為開且連通的集合
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
設 $U\subset\mathbb R^n$ 為開集。
<<<NEW>>>
設 $U\subset\mathbb R^n$ 為開集（本章所稱的「域」指開且連通的集合）。
<<<END>>>
<<<PATCH 22>>>
<<<OLD>>>
最後假設路徑獨立。固定
<<<NEW>>>
最後假設路徑獨立。因 $U$ 為開連通，任兩點皆可用域內分段 $C^1$ 路徑連接。固定
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
        "too few panels"
    )

    # 故障測試：曲線在 t=pi 通過原點。
<<<NEW>>>
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