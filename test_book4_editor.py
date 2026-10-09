import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import book2_editor as V2
import book3_editor as V3
import book4_spec as S
from book4_editor import ENGINE as E,parse_patches
from book2_notifications import inject_panel


def draft(n):
    return f'# 第{n:02d}章 {S.CHAPTERS[n-1][0]}\n\n'+'\n\n'.join('## '+s+'\n'+'分析教材'*100 for s in S.SECTIONS[:-1])+'\n```python\nassert 1+1==2\n```\n## 參考來源\nA1\n'


def block(old,new):return '<<<PATCH 01>>>\n<<<OLD>>>\n'+old+'\n<<<NEW>>>\n'+new+'\n<<<END>>>'


class FourthVolumeTests(unittest.TestCase):
    def test_isolated_spec(self):
        self.assertIs(E.S,S);self.assertNotEqual(V2.S.ROOT,S.ROOT);self.assertNotEqual(V3.ENGINE.S.ROOT,S.ROOT)
        self.assertEqual(len(S.CHAPTERS),30);self.assertEqual(len(S.roster()),40)
        for _,model in S.MODELS:self.assertEqual(sum(p['model']==model for p in S.roster()),8)
        for n in range(1,31):self.assertNotEqual(S.chapter_people(n)[0]['model'],S.chapter_people(n)[1]['model'])

    def test_literal_math_patch_and_overlap_safety(self):
        old=r'符號 $\nu$。';new=r'符號 $\mu$。'
        text=draft(1).replace('## 問題與直覺\n','## 問題與直覺\n'+old+'\n')
        proposal=parse_patches(block(old,new))
        self.assertEqual(E.checked_patches({'1':text},proposal)['1'],text.replace(old,new))
        with self.assertRaises(ValueError):E.checked_patches({'1':text},parse_patches(block('分析教材','新教材')))
        with self.assertRaises(ValueError):E.checked_patches({'1':text},parse_patches(block(old,new)+'\n'+block(r'$\nu$',r'$u$')))

    def test_reject_wrappers_and_truncation(self):
        for text in ['{}','```\n'+block('a','b')+'\n```',block('a','b')+'\n解說',block('a','b').replace('<<<END>>>','')]:
            with self.assertRaises(ValueError):parse_patches(text)

    def test_full_workflow_task_sessions_and_resume(self):
        observed=[]
        def fake(person,context,args,instructions):
            observed.append((person.id,str(person.workspace)))
            self.assertEqual(person.id,context['task_session_id'])
            if instructions==S.AUTHOR_RULES:return draft(context['number'])
            self.assertEqual(instructions,S.REVIEW_RULES)
            return '測試審稿\nVERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch.object(E.llm_agent,'decide',side_effect=fake),patch.object(E,'wait_remote_idle'),patch.object(E,'active_discussions',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(output=temp,init_only=False,max_calls=0,timeout=1,targeted_repair=False)
            E.run(args);root=Path(temp);s=json.loads((root/'status.json').read_text())
            self.assertEqual(s['status'],'completed_model_review_pending_human');self.assertEqual(s['calls'],65)
            self.assertTrue((root/'agents/v4_editor_00--ch01').is_dir())
            self.assertTrue((root/'agents/v4_editor_00--ch21').is_dir())
            page=(root/'index.html').read_text();self.assertIn('/book4/chapter/01',page);self.assertIn('Volume IV目錄',page)
            E.run(args);self.assertEqual(json.loads((root/'status.json').read_text())['calls'],65)
            panel=inject_panel(page,'/book4','Volume IV');self.assertIn('/book4/notifications',panel)
            self.assertEqual(inject_panel(panel,'/book4','Volume IV'),panel)

    def test_prevent_overwriting_other_volume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'config.json').write_text(json.dumps({'title':V3.ENGINE.S.TITLE}))
            with self.assertRaises(ValueError):E.run(argparse.Namespace(output=temp,init_only=True,max_calls=0,timeout=1))
            self.assertFalse((root/'TOC.md').exists())


if __name__=='__main__':unittest.main()
