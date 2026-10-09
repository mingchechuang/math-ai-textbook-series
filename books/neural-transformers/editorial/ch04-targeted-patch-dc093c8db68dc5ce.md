<<<PATCH 01>>>
<<<OLD>>>
    if mean_nll > 700: # ln(float_max) approx 709
<<<NEW>>>
    if mean_nll > np.log(np.finfo(np.float64).max):
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
<<<NEW>>>
        safe = p_support & (q > 0)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
<<<END>>>