<<<PATCH 01>>>
<<<OLD>>>
\nu^TJ_f(x)^Tv=\langle u,J_f(x)^Tv\rangle=\langle J_f(x)^Tv,u\rangle.
<<<NEW>>>
u^TJ_f(x)^Tv=\langle u,J_f(x)^Tv\rangle=\langle J_f(x)^Tv,u\rangle.
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

可達到等號，因此梯度是歐氏距離下的局部最陡上升方向。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
實作只需計算這個乘積，不必配置完整的 $m\times n$ Jacobian。

### 小命題：JVP與VJP的內積對偶
<<<NEW>>>
實作只需計算這個乘積，不必配置完整的 $m\times n$ Jacobian。

當輸入維度 $n$ 遠大於輸出維度 $m$ 時，VJP 只付出一次前向與一次反向的成本，即可得到與輸入同 shape 的梯度；反之 JVP 要逐一走過每個輸入方向，才能湊出完整 Jacobian。訓練損失是純量，因此 $m=1$，反向傳播的 VJP 形式幾乎總是最有效率的選擇。前向模式微分則適合輸出維度遠大於輸入維度的情形，例如少量參數影響大量輸出。選擇模式只看哪一側維度較小，與函數是否可微無關。

### 小命題：JVP與VJP的內積對偶
<<<END>>>
<<<PATCH 04>>>
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