<<<PATCH 01>>>
<<<OLD>>>
對 $T_{\mathrm{temp}}>0$：
<<<NEW>>>
在精確實數運算中，對 $T_{\mathrm{temp}}>0$：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
即使溫度為零，也要先驗證 top-k 與 top-p 是否在合法範圍。合法的 top-k/top-p 不會改變零溫度的 one-hot 結果；這是本章明定的「greedy 優先」政策，而不是讓零溫度繞過參數檢查。
<<<NEW>>>
即使溫度為零，也要先驗證 top-k 與 top-p 是否在合法範圍。合法的 top-k/top-p 不會改變零溫度的 one-hot 結果；這是本章明定的「greedy 優先」政策，而不是讓零溫度繞過參數檢查。下述排序命題也以精確實數運算為前提。實作則以浮點數先計算 `logits / temperature`：即使原始 logits 與正溫度都有限，極小溫度仍可能使縮放結果溢位；此時 `stable_softmax` 的有限值檢查會拒絕，而非產生機率。減去最大值不能補救已發生的除法溢位。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]), temperature=-1.0),
        ValueError
    )
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]),
            temperature=0.0, top_k=0),
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
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]),
            temperature=0.0, top_k=0),
        ValueError
    )
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
- 負溫度；
- 零溫度配非法 top-k 或 top-p；
<<<NEW>>>
- 負溫度，以及有限 logits 除以極小正溫度導致的浮點溢位；後者預期被拒絕；
- 零溫度配非法 top-k 或 top-p；
<<<END>>>