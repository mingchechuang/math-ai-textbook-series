import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import book4_editor as V4
import book5_spec as S
from book5_editor import ENGINE as E,parse_patches
from continue_book5_review import preflight
from book2_notifications import inject_panel


def draft(n):
    return f'# 第{n:02d}章 {S.CHAPTERS[n-1][0]}\n\n'+'\n\n'.join('## '+s+'\n'+'神經教材'*100 for s in S.SECTIONS[:-1])+'\n```python\nassert 1+1==2\n```\n## 參考來源\nN1\n'


class FifthVolumeTests(unittest.TestCase):
    def test_contract_and_isolation(self):
        self.assertIs(E.S,S);self.assertIsNot(E,V4.ENGINE);self.assertNotEqual(V4.ENGINE.S.ROOT,S.ROOT)
        self.assertEqual(len(S.CHAPTERS),30);self.assertEqual(len(S.roster()),40)
        self.assertEqual(S.MODELS,V4.ENGINE.S.MODELS)
        for _,model in S.MODELS:self.assertEqual(sum(p['model']==model for p in S.roster()),8)
        for n in range(1,31):self.assertNotEqual(S.chapter_people(n)[0]['model'],S.chapter_people(n)[1]['model'])
        self.assertEqual([sum(c[1]==part for c in S.CHAPTERS) for part in range(5)],[6]*5)

    def test_patch_safety(self):
        text=draft(1);old='## 問題與直覺';new='## 問題與直覺\n形狀驗證。'
        block=lambda a,b:f'<<<PATCH 01>>>\n<<<OLD>>>\n{a}\n<<<NEW>>>\n{b}\n<<<END>>>'
        self.assertEqual(E.checked_patches({'1':text},parse_patches(block(old,new)))['1'],text.replace(old,new))
        with self.assertRaises(ValueError):E.checked_patches({'1':text},parse_patches(block('神經教材','教材')))

    def test_mock_workflow_resume_and_sessions(self):
        def fake(person,context,args,instructions):
            self.assertEqual(person.id,context['task_session_id'])
            return draft(context['number']) if instructions==S.AUTHOR_RULES else 'VERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch.object(E.llm_agent,'decide',side_effect=fake),patch.object(E,'wait_remote_idle'),patch.object(E,'active_discussions',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(output=temp,init_only=False,max_calls=0,timeout=1,targeted_repair=False)
            E.run(args);root=Path(temp);state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'completed_model_review_pending_human');self.assertEqual(state['calls'],65)
            for chapter in ('01','21'):self.assertTrue((root/f'agents/v5_editor_00--ch{chapter}').is_dir())
            page=(root/'index.html').read_text();self.assertIn('/book5/chapter/01',page)
            self.assertIn('/book5/notifications',inject_panel(page,'/book5','Volume V'))
            E.run(args);self.assertEqual(json.loads((root/'status.json').read_text())['calls'],65)

    def test_five_model_preflight(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E.llm_agent,'decide',return_value='READY'),patch.object(E,'wait_remote_idle'):
            root=Path(temp);E.run(argparse.Namespace(output=temp,init_only=True,max_calls=0,timeout=1))
            preflight(root,1)
            records=json.loads((root/'editorial/preflight.json').read_text())
            self.assertEqual(len(records),5);self.assertEqual({r['person']['model'] for r in records},{m for _,m in S.MODELS})

    def test_other_volume_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'config.json').write_text(json.dumps({'title':V4.ENGINE.S.TITLE}))
            with self.assertRaises(ValueError):E.run(argparse.Namespace(output=temp,init_only=True,max_calls=0,timeout=1))
            self.assertFalse((root/'TOC.md').exists())


if __name__=='__main__':unittest.main()
