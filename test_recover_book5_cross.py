import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from book5_editor import ENGINE as E
from recover_book5_cross import recover,EVIDENCE
from test_recover_book5 import prepare


class CrossRecoveryTests(unittest.TestCase):
    def setup_root(self,root):
        prepare(root);state=json.loads((root/'status.json').read_text())
        for c in state['chapters'].values():c['status']='model_reviewed'
        state['chapters']['16']['status']='error'
        E.atomic(root/'status.json',json.dumps(state))

    def test_original_reviewers_and_no_manuscript_changes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',return_value='VERDICT: APPROVE') as invoke,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.setup_root(root);before={p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')}
            recover(root,1)
            self.assertEqual(invoke.call_count,5)
            for call,n in zip(invoke.call_args_list,EVIDENCE):self.assertEqual(call.args[0],E.S.chapter_people(n)[1])
            self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')})
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'ready_for_cross_review')

    def test_single_chapter_selection(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',return_value='VERDICT: APPROVE') as invoke,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.setup_root(root)
            recover(root,1,chapters=(16,))
            self.assertEqual(invoke.call_count,1)
            self.assertEqual(invoke.call_args.args[0],E.S.chapter_people(16)[1])
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'ready_for_cross_review')
            with self.assertRaises(ValueError):recover(root,1,chapters=(16,16))

    def test_failure_invalidates_old_approvals(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',side_effect=RuntimeError('test failure')),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.setup_root(root)
            with self.assertRaises(RuntimeError):recover(root,1)
            state=json.loads((root/'status.json').read_text());self.assertEqual(state['status'],'needs_editorial_attention')
            self.assertEqual(state['active'],{})
            for n in EVIDENCE:self.assertEqual(state['chapters'][str(n)]['status'],'needs_review')


if __name__=='__main__':unittest.main()
