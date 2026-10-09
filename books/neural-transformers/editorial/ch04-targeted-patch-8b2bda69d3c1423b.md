<<<PATCH 04>>>
<<<OLD>>>
    # 防止 overflow
    if mean_nll > 700: # ln(float_max) approx 709
        return np.inf
<<<NEW>>>
    # 超出 float64 可表示範圍才回傳 inf
    if mean_nll > np.log(np.finfo(np.float64).max):
        return np.inf
<<<END>>>