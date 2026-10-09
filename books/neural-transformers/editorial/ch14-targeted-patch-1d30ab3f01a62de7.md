<<<PATCH 14>>>
<<<OLD>>>
def run_tests():
    # 測試 1: 正常前向
<<<NEW>>>
def run_tests():
    rng = np.random.default_rng(14)
    # 測試 1: 正常前向
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
    Q = np.random.randn(2, 3, 4).astype(np.float64)
    K = np.random.randn(2, 3, 4).astype(np.float64)
    V = np.random.randn(2, 3, 4).astype(np.float64)
    Y, cache = scaled_dot_product_attention(Q, K, V)
    assert Y.shape == (2, 3, 4)
    
    # 測試 2: 梯度驗證
    dY = np.random.randn(2, 3, 4).astype(np.float64)
<<<NEW>>>
    Q = rng.standard_normal((2, 3, 4)).astype(np.float64)
    K = rng.standard_normal((2, 3, 4)).astype(np.float64)
    V = rng.standard_normal((2, 3, 4)).astype(np.float64)
    Y, cache = scaled_dot_product_attention(Q, K, V)
    assert Y.shape == (2, 3, 4)
    
    # 測試 2: 梯度 shape
    dY = rng.standard_normal((2, 3, 4)).astype(np.float64)
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
    Q_small = np.random.randn(1, 1, 2).astype(np.float64)
    K_small = np.random.randn(1, 2, 2).astype(np.float64)
    V_small = np.random.randn(1, 2, 2).astype(np.float64)
<<<NEW>>>
    Q_small = rng.standard_normal((1, 1, 2)).astype(np.float64)
    K_small = rng.standard_normal((1, 2, 2)).astype(np.float64)
    V_small = rng.standard_normal((1, 2, 2)).astype(np.float64)
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
    Q_b = np.random.randn(2, 3, 4).astype(np.float64)
    K_b = np.random.randn(1, 3, 4).astype(np.float64)
    V_b = np.random.randn(1, 3, 4).astype(np.float64)
<<<NEW>>>
    Q_b = rng.standard_normal((2, 3, 4)).astype(np.float64)
    K_b = rng.standard_normal((1, 3, 4)).astype(np.float64)
    V_b = rng.standard_normal((1, 3, 4)).astype(np.float64)
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
    # 非對稱長度、部分硬遮罩與三路梯度
    rng = np.random.default_rng(14)
    q = rng.standard_normal((1, 5, 2))
<<<NEW>>>
    # 非對稱長度、部分硬遮罩與三路梯度
    q = rng.standard_normal((1, 5, 2))
<<<END>>>