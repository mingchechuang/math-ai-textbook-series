<<<PATCH 01>>>
<<<OLD>>>
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau) = \\frac{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(\\hat{y}_i \\neq y_i)}{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(s(x_i) \\ge \\tau)} $$
<<<NEW>>>
*   定義接受指標 $a_i(\\tau)=\\mathbb{I}(s(x_i)\\ge\\tau)$。
*   **Coverage $C(\\tau)$**：模型同意回答的比例。
    $$ C(\\tau)=\\frac{1}{N}\\sum_{i=1}^{N}a_i(\\tau) $$
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau)=\\frac{\\sum_{i=1}^{N}a_i(\\tau)\\mathbb{I}(\\hat{y}_i\\neq y_i)}{\\sum_{i=1}^{N}a_i(\\tau)} $$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    """
    Compute Global Token-Weighted Perplexity.
    
    Args:
    - logits: np.ndarray (N, L, V). Model outputs.
    - labels: np.ndarray (N, L). Ground truth indices.
    - valid_mask: np.ndarray (N, L). Boolean, True for valid tokens.
    
    Returns:
    - perplexity: float
    - mean_nll: float
    """
    # 1. Validation
    if logits.ndim != 3:
<<<NEW>>>
    """
    Compute Global Token-Weighted Perplexity.
    
    Args:
    - logits: array-like (N, L, V). Every value, including masked positions,
      must be finite.
    - labels: array-like (N, L). Ground truth indices.
    - valid_mask: boolean array-like (N, L). Controls loss reduction only;
      it does not exempt masked logits from the finite-value requirement.
    
    Returns:
    - perplexity: float; may be inf if exponentiation overflows.
    - mean_nll: float
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask)
    # 1. Validation
    if logits.ndim != 3:
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    mean_nll = total_nll / num_valid
    perplexity = np.exp(mean_nll)
    
    if not np.isfinite(perplexity):
        print(f"Warning: PPL overflow. Mean NLL is {mean_nll}.")
        
    return perplexity, mean_nll
<<<NEW>>>
    mean_nll = total_nll / num_valid
    # Overflow is represented by inf; the numerical routine has no print side effect.
    with np.errstate(over="ignore"):
        perplexity = np.exp(mean_nll)
    return float(perplexity), float(mean_nll)
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
        if probs.ndim != 2:
            raise ValueError("Probs must be 2D.")
        N, C = probs.shape
        if labels.shape != (N,):
<<<NEW>>>
        if probs.ndim != 2:
            raise ValueError("Probs must be 2D.")
        N, C = probs.shape
        if N == 0 or C == 0:
            raise ValueError("N and C must be positive.")
        if labels.shape != (N,):
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
ECE (5 bins) | 0.05 | 0.20 |
<<<NEW>>>
ECE (5 bins；校準事件數未提供) | 0.05 | 0.20 |
<<<END>>>