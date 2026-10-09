<<<PATCH 01>>>
<<<OLD>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。
<<<NEW>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。

此命題是實數運算的陳述。固定精度浮點實作中，`logits / temperature` 若因極小正溫度溢位成非有限值，本 API 的契約是拒絕該步輸入，而不是宣稱仍可比較機率。因此下方 `next_token_distribution` 在縮放後、softmax 前檢查縮放結果是否全為有限值；不滿足就拋出例外。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    prob = stable_softmax(logits / temperature)
    prob = apply_top_k(prob, top_k)
    prob = apply_top_p(prob, top_p)
    return renormalize(prob)
<<<NEW>>>
    scaled = logits / temperature
    if not np.all(np.isfinite(scaled)):
        raise ValueError(
            "logits 除以 temperature 後必須全部有限；請避免極小正溫度"
        )

    prob = stable_softmax(scaled)
    prob = apply_top_k(prob, top_k)
    prob = apply_top_p(prob, top_p)
    return renormalize(prob)
<<<END>>>
<<<PATCH 03>>>
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
    # 極小正溫度可能讓 logits/temperature 溢位成無限值；
    # 契約要求拒絕，而不是回傳未定義分布。
    must_raise(
        lambda: next_token_distribution(
            np.array([1e308, 0.0]), temperature=1e-300),
        ValueError
    )
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
故障測試檢查：

- 非有限模型輸出；
- 負溫度；
- 零溫度配非法 top-k 或 top-p；
<<<NEW>>>
故障測試檢查：

- 非有限模型輸出；
- 負溫度；
- 極小正溫度導致 logits 縮放溢位；
- 零溫度配非法 top-k 或 top-p；
<<<END>>>