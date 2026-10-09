<<<PATCH 01>>>
<<<OLD>>>
若分支尚未全部處理便提早往上游傳播，所得梯度會缺少路徑貢獻。
<<<NEW>>>
若分支尚未全部處理便提早往上游傳播，所得梯度會缺少路徑貢獻。實作時可為每個節點準備一個與其輸出同形狀的梯度槽，初值為零；每處理一條下游路徑，就把該路徑的貢獻加進槽中。只有所有使用該節點輸出的路徑都處理完畢，才用累積結果計算更上游的梯度。這說明反向拓撲順序不只是走訪方向，也規定了何時可以傳播已累加的梯度。

除檢查最終損失外，除錯還應逐節點核對前向值與反向形狀。矩陣乘法要注意因子次序，偏置廣播要沿被複製的軸求和，平均損失的除數不可重複套用。若解析梯度與有限差分不符，應先排查這些局部規則，再檢查差分步長及輸入是否落在不可微點；不能只靠放寬容差掩蓋錯誤。
<<<END>>>
<<<PATCH 02>>>
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