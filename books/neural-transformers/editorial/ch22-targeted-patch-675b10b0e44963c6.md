<<<PATCH 01>>>
<<<OLD>>>
即使溫度為零，也要先驗證 top-k 與 top-p 是否在合法範圍。合法的 top-k/top-p 不會改變零溫度的 one-hot 結果；這是本章明定的「greedy 優先」政策，而不是讓零溫度繞過參數檢查。
<<<NEW>>>
即使溫度為零，也要先驗證 top-k 與 top-p 是否在合法範圍。合法的 top-k/top-p 不會改變零溫度的 one-hot 結果；這是本章明定的「greedy 優先」政策，而不是讓零溫度繞過參數檢查。上述公式及下述排序命題使用精確實數運算；浮點實作先計算 `logits / temperature`，有限 logits 遇極小正溫度仍可能溢位。此時後續 `stable_softmax` 會拒絕非有限結果，而非保證產生機率；減去最大值無法補救已發生的除法溢位。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]), temperature=-1.0),
        ValueError
    )
<<<NEW>>>
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]), temperature=-1.0),
        ValueError
    )
    with np.errstate(over="ignore"):
        must_raise(
            lambda: next_token_distribution(
                np.array([0.0, 1.0]), temperature=1e-320),
            ValueError
        )
<<<END>>>