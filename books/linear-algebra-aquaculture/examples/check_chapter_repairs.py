"""獨立核對第4/9/14章關鍵數值；不匯入或執行Markdown內程式。
使用既有NumPy環境，在CPU執行。通過不等於整章或全書已驗證。
"""
import unittest
import numpy as np


class ChapterRepairChecks(unittest.TestCase):
    def test_calibration_and_nonidentifiability(self):
        a = np.array([[20., 1.], [25., 1.]])
        np.testing.assert_allclose(np.linalg.solve(a, [8., 12.]), [.8, -8.])
        b = np.array([[1., 1., 0.], [1., 0., 1.], [0., -1., 1.]])
        self.assertEqual(np.linalg.matrix_rank(b), 2)
        for t in (-3., 0., 12.7 / 3):
            np.testing.assert_allclose(b @ [t, 6.2-t, 6.5-t], [6.2, 6.5, .3])

    def test_pca_exercise_positive_cross_term(self):
        x = np.array([[1., 0.], [0., 1.], [-1., -1.]])
        np.testing.assert_allclose(x.mean(axis=0), [0., 0.])
        s = np.cov(x, rowvar=False, ddof=1)
        np.testing.assert_allclose(s, [[1., .5], [.5, 1.]])
        w, u = np.linalg.eigh(s)
        np.testing.assert_allclose(w, [.5, 1.5])
        np.testing.assert_allclose(np.outer(u[:, -1], u[:, -1]), np.full((2, 2), .5))
        self.assertAlmostEqual(w[-1] / w.sum(), .75)

    def test_pca_nonstandardized_example(self):
        x = np.array([[1., 2.], [2., 4.], [3., 3.], [4., 5.]])
        s = np.cov(x, rowvar=False, ddof=1)
        np.testing.assert_allclose(np.linalg.eigvalsh(s), [1/3, 3.])
        direction = np.ones(2) / np.sqrt(2)
        self.assertAlmostEqual(([3., 4.] - x.mean(axis=0)) @ direction, 1/np.sqrt(2))

    def test_zero_variance_convention(self):
        x = np.full((4, 3), 2.)
        sigma = x.std(axis=0, ddof=1)
        z = (x - x.mean(axis=0)) / np.where(sigma == 0, 1., sigma)
        s = z.T @ z / (len(x)-1)
        w = np.linalg.eigvalsh(s)
        tol = 32 * np.finfo(float).eps * len(w) * max(1., float(np.max(np.abs(w))))
        self.assertLessEqual(w.sum(), tol)
        reported = np.full(len(w), np.nan)
        self.assertTrue(np.isnan(reported).all())

    def test_lora_direction_and_missing_input(self):
        a = np.array([[1., 0., 0.]])
        b = np.array([[0.], [1.], [-1.], [2.]])
        w = np.array([[.3, -.2, .1], [.1, .4, -.1], [-.2, .1, .3], [0., .2, -.2]])
        dw = 2 * b @ a
        x = np.array([1., -.5, 0.])
        self.assertEqual(np.linalg.matrix_rank(dw), 1)
        np.testing.assert_allclose((w+dw) @ x, [.4, 1.9, -2.25, 3.9])
        np.testing.assert_allclose(dw @ [0., 1., 0.], np.zeros(4))
        # 不提高秩也能改讀do；這不會補出缺失的夜間資訊。
        a_do = np.array([[0., 1., 0.]])
        self.assertEqual(np.linalg.matrix_rank(b @ a_do), 1)
        np.testing.assert_allclose((b @ a_do) @ [0., 1., 0.], b[:, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
