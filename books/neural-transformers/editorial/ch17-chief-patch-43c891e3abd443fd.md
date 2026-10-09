<<<PATCH 17>>>
<<<OLD>>>
    # 故障：奇數 dh。
    try:
        rope(
            np.zeros((1, 1, 2, 3), dtype=np.float64),
            np.array([0, 1], dtype=np.int64)
        )
        raise AssertionError("odd dh was not rejected")
    except ValueError:
        pass

    # 故障：位置長度與 T 不一致。
    try:
        rope(
            np.zeros((1, 1, 2, 4), dtype=np.float64),
            np.array([0], dtype=np.int64)
        )
        raise AssertionError("position mismatch was not rejected")
    except ValueError:
        pass

    # 故障：sinusoidal 輸出 dtype 不可為整數。
    try:
        sinusoidal_positions(2, 4, dtype=np.int64)
        raise AssertionError("integer dtype was not rejected")
    except TypeError:
        pass

    # 故障：布林值不可冒充整數參數。
    try:
        sinusoidal_positions(True, 4)
        raise AssertionError("boolean length was not rejected")
    except ValueError:
        pass

    # 故障：非有限輸入。
    bad = np.zeros((1, 1, 1, 4), dtype=np.float64)
    bad[0, 0, 0, 0] = np.nan
    try:
        rope(bad, np.array([0], dtype=np.int64))
        raise AssertionError("NaN input was not rejected")
    except ValueError:
        pass
<<<NEW>>>
    # 故障：奇數 dh。
    caught = False
    try:
        rope(
            np.zeros((1, 1, 2, 3), dtype=np.float64),
            np.array([0, 1], dtype=np.int64)
        )
        raise AssertionError("odd dh was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：位置長度與 T 不一致。
    caught = False
    try:
        rope(
            np.zeros((1, 1, 2, 4), dtype=np.float64),
            np.array([0], dtype=np.int64)
        )
        raise AssertionError("position mismatch was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：sinusoidal 輸出 dtype 不可為整數。
    caught = False
    try:
        sinusoidal_positions(2, 4, dtype=np.int64)
        raise AssertionError("integer dtype was not rejected")
    except TypeError:
        caught = True
    assert caught

    # 故障：布林值不可冒充整數參數。
    caught = False
    try:
        sinusoidal_positions(True, 4)
        raise AssertionError("boolean length was not rejected")
    except ValueError:
        caught = True
    assert caught

    # 故障：非有限輸入。
    bad = np.zeros((1, 1, 1, 4), dtype=np.float64)
    bad[0, 0, 0, 0] = np.nan
    caught = False
    try:
        rope(bad, np.array([0], dtype=np.int64))
        raise AssertionError("NaN input was not rejected")
    except ValueError:
        caught = True
    assert caught
<<<END>>>