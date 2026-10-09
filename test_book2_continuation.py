import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import book2_editor as B
from continue_book2_review import continue_review


class ContinuationTests(unittest.TestCase):
    def prepare(self,root):
        B.A.initialize(root)
        state={'status':'needs_editorial_attention','calls':239,'chapters':{},'part_reviews':{},'active':{},'attention':[]}
        B.atomic(root/'status.json',json.dumps(state))
        return state

    def test_completion_stops_and_removes_call_cap(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);state=self.prepare(root)
            def run(args):
                self.assertEqual(args.max_calls,0);self.assertTrue(args.targeted_repair)
                state['status']='completed_model_review_pending_human'
                B.atomic(root/'status.json',json.dumps(state))
            with patch('continue_book2_review.B.run',side_effect=run) as mocked:
                continue_review(root)
                self.assertEqual(mocked.call_count,1)

    def test_three_no_progress_passes_stop_without_budget_reason(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.prepare(root)
            with patch('continue_book2_review.B.run') as mocked:
                continue_review(root)
                self.assertEqual(mocked.call_count,3)
            state=json.loads((root/'status.json').read_text())
            self.assertIn('連續3輪無內容',state['attention'][-1])
            self.assertNotEqual(state['status'],'completed_model_review_pending_human')


if __name__=='__main__':unittest.main()
