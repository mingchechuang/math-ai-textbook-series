<<<PATCH 01>>>
<<<OLD>>>
由內積的定義與對稱性，

$$
\nu^TJ_f(x)^Tv=\langle u,J_f(x)^Tv\rangle=\langle J_f(x)^Tv,u\rangle.
$$
<<<NEW>>>
由內積的定義與對稱性，

$$
u^TJ_f(x)^Tv=\langle u,J_f(x)^Tv\rangle=\langle J_f(x)^Tv,u\rangle.
$$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
當 $\nabla f(x)\ne0$ 時，取

$$
\nu=\frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

可達到等號，因此梯度是歐氏距離下的局部最陡上升方向。
<<<NEW>>>
當 $\nabla f(x)\ne0$ 時，取

$$
u=\frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

可達到等號，因此梯度是歐氏距離下的局部最陡上升方向。達到等號的條件是方向 $u$ 與梯度同向且長度為一；只要方向偏離梯度任何角度，內積就會嚴格小於梯度範數。這說明梯度作為方向只是一個局部性質，並未告訴我們沿此方向走多遠會使函數下降或上升多少。步長選擇與收斂分析是另一層問題，不能由一階最陡方向本身保證。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    expect_value_error(
        "non-finite logit",
        lambda: bce_logits_mean(
            np.array([[np.inf]]), np.array([[1.0]])
        ),
    )
<<<NEW>>>
    expect_value_error(
        "non-finite logit",
        lambda: bce_logits_mean(
            np.array([[np.inf]]), np.array([[1.0]])
        ),
    )
    expect_value_error(
        "NaN logit",
        lambda: bce_logits_mean(
            np.array([[np.nan]]), np.array([[1.0]])
        ),
    )
<<<END>>>