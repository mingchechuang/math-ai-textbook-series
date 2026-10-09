<<<PATCH 01>>>
<<<OLD>>>
**命題。** 若 $z_i,z_j\in\mathbb{R}$ 且 $T_{\mathrm{temp}}>0$，則：
<<<NEW>>>
**命題（實數運算）。** 若 $z_i,z_j\in\mathbb{R}$ 且 $T_{\mathrm{temp}}>0$，並以精確實數運算定義 softmax，則：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。
<<<NEW>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。

這是實數命題，不是任意浮點輸入皆能成功計算的保證。程式採用 float64 計算 `logits / temperature`；即使原始 logits 與正溫度皆有限，極小溫度仍可能使縮放結果溢位。本章對這種情形明確拒絕，不以可能產生的無限值計算機率；呼叫者可改用 `greedy=True` 或本章定義的零溫度政策，但不能把它當成原正溫度分布的精確數值結果。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    prob = stable_softmax(logits / temperature)
    prob = apply_top_k(prob, top_k)
<<<NEW>>>
    scaled = logits / temperature
    if not np.all(np.isfinite(scaled)):
        raise ValueError("溫度縮放後 logits 非有限")
    prob = stable_softmax(scaled)
    prob = apply_top_k(prob, top_k)
<<<END>>>
<<<PATCH 04>>>
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
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]), temperature=1e-320),
        ValueError
    )
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
- 負溫度；
- 零溫度配非法 top-k 或 top-p；
<<<NEW>>>
- 負溫度，以及有限正溫度使浮點縮放溢位的情形；後者應拋出 `ValueError`，浮點環境也可能先發出溢位警告；
- 零溫度配非法 top-k 或 top-p；
<<<END>>>