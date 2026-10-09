import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from book4_editor import ENGINE as E
from recover_book4_cross import recover
from test_recover_book4 import setup_root


class CrossRecoveryTests(unittest.TestCase):
    def prepare(self,root):
        setup_root(root)
        s=json.loads((root/'status.json').read_text())
        for v in s['chapters'].values():v['status']='model_reviewed'
        s['chapters']['16']['status']='error'
        E.atomic(root/'status.json',json.dumps(s))

    def test_review_only_and_preserve_all_manuscripts(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',return_value='VERDICT: APPROVE') as invoke,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.prepare(root)
            before={p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')}
            recover(root,1)
            self.assertEqual(invoke.call_count,3)
            self.assertEqual({p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')},before)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'ready_for_cross_review')
            self.assertIn('先前審稿要求改成y3=t',invoke.call_args_list[-1].args[2]['editorial_evidence'])

    def test_api_failure_leaves_changed_chapters_unapproved(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',side_effect=RuntimeError('test failure')),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.prepare(root)
            with self.assertRaises(RuntimeError):recover(root,1)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'needs_editorial_attention');self.assertEqual(state['active'],{})
            for n in (16,25,26):self.assertEqual(state['chapters'][str(n)]['status'],'needs_review')


if __name__=='__main__':unittest.main()
