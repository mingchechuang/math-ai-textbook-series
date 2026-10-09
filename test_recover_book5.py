import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from book5_editor import ENGINE as E
from recover_book5 import recover
from test_book5_editor import draft


def prepare(root):
    E.run(argparse.Namespace(output=str(root),init_only=True,max_calls=0,timeout=1))
    state=json.loads((root/'status.json').read_text())
    for n in range(1,31):
        text=draft(n);(root/'chapters'/f'{n:02d}.md').write_text(text)
        state['chapters'][str(n)]={'status':'error' if n==29 else 'model_reviewed','sha256':E.digest(text)}
    E.atomic(root/'status.json',json.dumps(state))


class RecoveryTests(unittest.TestCase):
    def test_review_only_preserves_every_chapter(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',return_value='VERDICT: APPROVE') as invoke,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);prepare(root);before={p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')}
            recover(root,1)
            self.assertEqual(invoke.call_count,1)
            self.assertEqual(invoke.call_args.args[0],E.S.chapter_people(29)[1])
            self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')})
            state=json.loads((root/'status.json').read_text());self.assertEqual(state['status'],'ready_for_cross_review')
            self.assertEqual(state['active'],{})

    def test_failure_does_not_reuse_approval(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',side_effect=RuntimeError('test API failure')),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);prepare(root)
            with self.assertRaises(RuntimeError):recover(root,1)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'needs_editorial_attention')
            self.assertEqual(state['chapters']['29']['status'],'needs_review')
            self.assertEqual(state['active'],{});self.assertTrue(state['attention'])


if __name__=='__main__':unittest.main()
