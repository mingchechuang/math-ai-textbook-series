<<<PATCH 07>>>
<<<OLD>>>
若分支尚未全部處理便提早往上游傳播，所得梯度會缺少路徑貢獻。
<<<NEW>>>
若分支尚未全部處理便提早往上游傳播，所得梯度會缺少路徑貢獻。實作時可為每個節點準備與其輸出同形狀的梯度槽，初值為零。每處理一條下游路徑，就把傳回的貢獻加入梯度槽；待所有路徑處理完畢，才用累積結果向更上游傳播。這說明反向拓撲順序不只是走訪方向，也決定了何時能傳播一個節點的梯度。

除錯時，應先核對前向值和各節點的梯度形狀，再檢查矩陣乘法的因子次序、偏置廣播所需的求和軸，以及平均損失的除數是否只套用一次。若解析梯度與有限差分不符，還須檢查差分步長及輸入是否位於不可微點；不能只靠放寬容差掩蓋錯誤。數值吻合也不能取代局部規則的數學推導。
<<<END>>>
<<<PATCH 07>>>
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