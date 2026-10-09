<<<PATCH 04>>>
<<<OLD>>>
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
<<<NEW>>>
        safe = p_support & (q > 0)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
<<<END>>>