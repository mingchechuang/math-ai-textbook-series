<<<PATCH 15>>>
<<<OLD>>>
    for bad_mask in (
        np.zeros((3, 3), dtype=bool),
        np.ones(3, dtype=bool),
        np.ones((3, 2), dtype=bool),
    ):
        try:
            attention(x, x, x, bad_mask)
        except ValueError:
            pass
        else:
            raise AssertionError("非法遮罩應被拒絕")
<<<NEW>>>
    for bad_mask in (
        np.zeros((3, 3), dtype=bool),
        np.ones(3, dtype=bool),
        np.ones((3, 2), dtype=bool),
    ):
        rejected = False
        try:
            attention(x, x, x, bad_mask)
        except ValueError:
            rejected = True
        if not rejected:
            raise AssertionError("非法遮罩應被拒絕")
<<<END>>>

<<<PATCH 15>>>
<<<OLD>>>
    try:
        masked_token_mean(
            np.array([0.2, 0.4]), np.array([False, False])
        )
    except ValueError:
        pass
    else:
        raise AssertionError("零有效 token 應被拒絕")
<<<NEW>>>
    rejected = False
    try:
        masked_token_mean(
            np.array([0.2, 0.4]), np.array([False, False])
        )
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("零有效 token 應被拒絕")
<<<END>>>

<<<PATCH 15>>>
<<<OLD>>>
    huge = np.array([[1e308]])
    try:
        attention(huge, huge, np.array([[1.0]]),
                  np.array([[True]]))
    except ValueError:
        pass
    else:
        raise AssertionError("非有限注意力分數應被拒絕")
<<<NEW>>>
    huge = np.array([[1e308]])
    rejected = False
    try:
        attention(huge, huge, np.array([[1.0]]),
                  np.array([[True]]))
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("非有限注意力分數應被拒絕")
<<<END>>>