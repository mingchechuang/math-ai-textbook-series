import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import book2_editor as V2
import book2_spec
import book3_spec as S
from book3_editor import ENGINE as E
from book2_notifications import inject_panel


def draft(n):
    return f'# 第{n:02d}章 {S.CHAPTERS[n-1][0]}\n\n'+'\n\n'.join('## '+section+'\n'+'場論教材'*100 for section in S.SECTIONS[:-1])+'\n```python\nassert 1 + 1 == 2\n```\n## 參考來源\nF1\n'


class VolumeThreeTests(unittest.TestCase):
    def test_isolated_contracts_and_roles(self):
        self.assertIs(V2.S,book2_spec);self.assertIs(E.S,S)
        self.assertIsNot(E.Editor,V2.Editor)
        self.assertEqual(len(S.CHAPTERS),30);self.assertEqual(len(S.roster()),40)
        for provider,model in S.MODELS:self.assertEqual(sum(p['model']==model for p in S.roster()),8)
        for n in range(1,31):
            a,r=S.chapter_people(n)
            self.assertTrue(a['id'].startswith('v3_editor_'));self.assertNotEqual(a['model'],r['model'])
        self.assertFalse(E.validate(draft(1),1))
        self.assertTrue(V2.validate(draft(1),1)) # 章級小節契約不同

    def test_other_volume_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'config.json').write_text(json.dumps({'title':book2_spec.TITLE}))
            (root/'TOC.md').write_text('existing')
            with self.assertRaises(ValueError):E.run(argparse.Namespace(output=temp,init_only=True,max_calls=0,timeout=5))
            self.assertEqual((root/'TOC.md').read_text(),'existing')

    def test_full_workflow_and_resume(self):
        def fake(person,context,args,instructions):
            self.assertTrue(person.id.startswith('v3_editor_'))
            if instructions==S.AUTHOR_RULES:return draft(context['number'])
            self.assertEqual(instructions,S.REVIEW_RULES)
            return '測試批准\nVERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch.object(E.llm_agent,'decide',side_effect=fake),patch.object(E,'wait_remote_idle'),patch.object(E,'active_discussions',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(output=temp,init_only=False,max_calls=0,timeout=5,targeted_repair=False)
            E.run(args)
            root=Path(temp);state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'completed_model_review_pending_human');self.assertEqual(state['calls'],65)
            page=(root/'index.html').read_text();self.assertIn('/book3/chapter/01',page);self.assertIn('Volume III目錄',page)
            self.assertNotIn('/book2/chapter/',page)
            E.run(args);self.assertEqual(json.loads((root/'status.json').read_text())['calls'],65)
            self.assertIs(V2.S,book2_spec)

    def test_notification_routes(self):
        page=inject_panel('<html></html>','/book3','Volume III')
        self.assertIn('id="book3-notifications"',page)
        self.assertIn('/book3/notifications',page)
        self.assertNotIn('/book2/notifications',page)


if __name__=='__main__':unittest.main()
