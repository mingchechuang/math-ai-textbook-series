<<<PATCH 01>>>
<<<OLD>>>
def run_checks():
    # 正常：三個 token，最後一個是 padding key。
<<<NEW>>>
def assert_raises_value_error(fn, message):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(message)


def run_checks():
    # 正常：三個 token，最後一個是 padding key。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
    # 故障：全遮罩列、零有效目標、畸形遮罩均須拒絕。
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

    try:
        masked_token_mean(
            np.array([0.2, 0.4]), np.array([False, False])
        )
    except ValueError:
        pass
    else:
        raise AssertionError("零有效 token 應被拒絕")

    # 有限但極大的輸入可能使點積溢位；應拒絕非有限分數。
    huge = np.array([[1e308]])
    try:
        attention(huge, huge, np.array([[1.0]]),
                  np.array([[True]]))
    except ValueError:
        pass
    else:
        raise AssertionError("非有限注意力分數應被拒絕")
<<<NEW>>>
    # 故障：全遮罩列、零有效目標、畸形遮罩均須拒絕。
    for bad_mask in (
        np.zeros((3, 3), dtype=bool),
        np.ones(3, dtype=bool),
        np.ones((3, 2), dtype=bool),
    ):
        assert_raises_value_error(
            lambda bad_mask=bad_mask: attention(x, x, x, bad_mask),
            "非法遮罩應被拒絕"
        )

    assert_raises_value_error(
        lambda: masked_token_mean(
            np.array([0.2, 0.4]), np.array([False, False])
        ),
        "零有效 token 應被拒絕"
    )

    # 有限但極大的輸入可能使點積溢位；應拒絕非有限分數。
    huge = np.array([[1e308]])
    assert_raises_value_error(
        lambda: attention(
            huge, huge, np.array([[1.0]]), np.array([[True]])
        ),
        "非有限注意力分數應被拒絕"
    )
<<<END>>>