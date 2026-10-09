"""CPU核對跨章修訂；第19章程式已逐行檢視才在此執行。
其他數學例為獨立核對，不代表全書驗證。勿對未審查的生成程式使用此方法。
"""
import contextlib
import io
from pathlib import Path
import re
import unittest
import numpy as np


class CrossChapterChecks(unittest.TestCase):
    def test_ch5_elimination(self):
        x = np.array([[20., 6., 7., 12.], [22., 5., 8., 10.], [24., 4., 7., 8.]])
        np.testing.assert_allclose(x[1, :3]-x[0, :3], [2., -1., 1.])
        np.testing.assert_allclose(x[2, :3]-x[1, :3], [2., -1., -1.])
        self.assertEqual(np.linalg.matrix_rank(x), 3)
        np.testing.assert_allclose(x @ [0., -2., 0., 1.], np.zeros(3))

    def test_ch11_column_gradient(self):
        w = np.array([[1., 2.], [3., 4.]])
        x = np.array([[1.], [2.]])
        target = np.array([[4.], [10.]])
        gradient = w.T @ (w @ x - target)
        self.assertEqual(gradient.shape, (2, 1))
        np.testing.assert_allclose(gradient, [[4.], [6.]])
        for i in range(2):
            step = np.zeros_like(x); step[i, 0] = 1e-5
            loss = lambda value: .5 * np.sum((w @ value-target)**2)
            finite = (loss(x+step)-loss(x-step))/2e-5
            self.assertAlmostEqual(finite, gradient[i, 0], places=7)

    def test_ch13_output_weight(self):
        o = np.ones((3, 8)); wo = np.ones((5, 8))
        self.assertEqual((o @ wo.T).shape, (3, 5))

    def test_ch16_paired_loss(self):
        s = np.array([[1., -.28], [0., .96]])
        scores = s-s.max(axis=1, keepdims=True)
        probs = np.exp(scores)/np.exp(scores).sum(axis=1, keepdims=True)
        loss = -np.log(probs.diagonal()).mean()
        self.assertAlmostEqual(loss, .285, places=3)
        text = (Path(__file__).resolve().parents[1]/'chapters/16.md').read_text()
        self.assertIn('n=m', text)

    def test_ch19_reviewed_snippet_branches(self):
        text = (Path(__file__).resolve().parents[1]/'chapters/19.md').read_text()
        blocks = re.findall(r'```python\n(.*?)```', text, re.S)
        self.assertEqual(len(blocks), 1)
        scenarios = [
            ({}, 'constraint_violation'),
            ({'obs_time = 100.0': 'obs_time = None'}, 'invalid_timestamp_or_window'),
            ({'obs_time = 100.0': 'obs_time = "bad"'}, 'invalid_timestamp_or_window'),
            ({'obs_time = 100.0': 'obs_time = True'}, 'invalid_timestamp_or_window'),
            ({'obs_time = 100.0': 'obs_time = np.nan'}, 'invalid_timestamp_or_window'),
            ({'now_time = 105.0': 'now_time = np.inf'}, 'invalid_timestamp_or_window'),
            ({'obs_time = 100.0': 'obs_time = 200.0'}, 'stale_or_future'),
            ({'obs_time = 100.0': 'obs_time = 0.0'}, 'stale_or_future'),
            ({'staleness_limit = 60.0': 'staleness_limit = -1.0'}, 'invalid_window'),
            ({'invalid_x =': 'x[1, 0] = np.inf\ninvalid_x ='}, 'invalid_shape_or_nonfinite_data'),
            ({'invalid_x =': 'x[1, 0] = np.nan\ninvalid_x ='}, 'invalid_shape_or_nonfinite_data'),
            ({'delta = np.array([[3.0],': 'delta = np.array([[1.0],'}, 'review_required'),
        ]
        for replacements, reason in scenarios:
            with self.subTest(reason=reason, replacements=replacements):
                code = blocks[0]
                for old, new in replacements.items():
                    self.assertEqual(code.count(old), 1)
                    code = code.replace(old, new)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(compile(code, 'reviewed-chapter-19', 'exec'), {})
                self.assertIn(reason, out.getvalue())
                if reason != 'review_required':self.assertIn('status: reject', out.getvalue())
                else:self.assertNotIn('status: reject', out.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
