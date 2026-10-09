<<<PATCH 01>>>
<<<OLD>>>
因 $\\mathbb R^n$ 的開連通集是路徑連通的，任取域內兩點，可用域內分段 $C^1$ 路徑連接。沿該路徑使用鏈式法則，$\\phi-\\psi$ 的導數恆為零，因此 $\\phi-\\psi$ 沿路徑恆定。
<<<NEW>>>
歐氏空間的開連通集可由域內有限段折線連接任意兩點，因此可選取分段 $C^1$ 路徑。沿該路徑使用鏈式法則，$\\phi-\\psi$ 的導數恆為零，因此 $\\phi-\\psi$ 沿路徑恆定。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
設 $U$ 路徑連通。若對任意 $A,B\\in U$，以及任意兩條由 $A$ 到 $B$ 的域內分段 $C^1$ 曲線 $\\gamma_1,\\gamma_2$，皆有
<<<NEW>>>
設 $U$ 為開且路徑連通的集合。歐氏空間中，這也保證任意兩點可由域內分段 $C^1$ 曲線連接。若對任意 $A,B\\in U$，以及任意兩條由 $A$ 到 $B$ 的域內分段 $C^1$ 曲線 $\\gamma_1,\\gamma_2$，皆有
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
$$
\\nu=\\frac{T-T_0}{s_T},
\\qquad
v=\\frac{S-S_0}{s_S},
$$
<<<NEW>>>
$$
u=\\frac{T-T_0}{s_T},
\\qquad
v=\\frac{S-S_0}{s_S},
$$
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
$$
\\nabla_{(u,v)}\\Phi
=
\\begin{pmatrix}
\\nu+v\\\\
\\nu+4v
\\end{pmatrix}.
$$
<<<NEW>>>
$$
\\nabla_{(u,v)}\\Phi
=
\\begin{pmatrix}
u+v\\\\
u+4v
\\end{pmatrix}.
$$
<<<END>>>

<<<PATCH 05>>>
<<<OLD>>>
    # 故障測試：非法半徑及分割數。
    assert_raises(
        ValueError,
        lambda: circle(radius=0.0),
        "zero radius"
    )
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
    # 故障測試：非法半徑、分割數及場值 shape。
    assert_raises(
        ValueError,
        lambda: circle(radius=0.0),
        "zero radius"
    )
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
    assert_raises(
        ValueError,
        lambda: line_integral(
            lambda points: points[:, 0],
            curve,
            dcurve,
            0.0,
            2.0 * np.pi
        ),
        "wrong field shape"
    )

    # 故障測試：曲線在 t=pi 通過原點。
<<<END>>>