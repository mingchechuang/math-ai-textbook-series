<<<PATCH 01>>>
<<<OLD>>>
$$
\nu^TJ_f(x)^Tv=\langle u,J_f(x)^Tv\rangle=\langle J_f(x)^Tv,u\rangle.
$$
<<<NEW>>>
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
<<<NEW>>>
當 $\nabla f(x)\ne0$ 時，取

$$
u=\frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
反向傳播是鏈式法則的有效實作。每個節點接收下游梯度，以局部導數計算對其輸入的 VJP，再把結果傳向上游。若同一值流向多個分支，各分支都會產生梯度貢獻，這些貢獻必須相加，不能互相覆寫。
<<<NEW>>>
反向傳播是鏈式法則的有效實作。每個節點接收下游梯度，以局部導數計算對其輸入的 VJP，再把結果傳向上游。若同一值流向多個分支，各分支都會產生梯度貢獻，這些貢獻必須相加，不能互相覆寫。

可以把上游梯度理解為「下游損失對目前節點輸出的敏感度」，而局部反傳回答的是「這份敏感度應如何分配給節點的各個輸入」。例如仿射節點同時接收資料、權重與偏置，三者取得的梯度形狀不同，卻來自同一份輸出端梯度。執行時應先記錄前向計算所需的值與形狀，再核對每條反向路徑：乘法要注意因子的次序，廣播要沿被複製的軸求和，分支要在共同來源處累加。這些局部規則組合起來才是整張圖的梯度；只核對最終損失數值，無法發現某條反向路徑被漏掉。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    expect_value_error(
        "non-finite logit",
        lambda: bce_logits_mean(
            np.array([[np.inf]]), np.array([[1.0]])
        ),
    )
    expect_value_error(
        "wrong label shape",
<<<NEW>>>
    expect_value_error(
        "infinite logit",
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
    expect_value_error(
        "wrong label shape",
<<<END>>>