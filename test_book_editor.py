import argparse
import collections
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import book_editor as be
from book_spec import SECTIONS, AUTHOR_RULES


class BookTests(unittest.TestCase):
    def test_roster(self):
        people=be.roster()
        self.assertEqual(len(people),40)
        self.assertEqual(set(collections.Counter(p['model'] for p in people).values()),{8})
        for i in range(20):self.assertNotEqual(people[2*i]['model'],people[2*i+1]['model'])

    def test_count_excludes_code_and_math(self):
        self.assertEqual(be.prose_count('中文\n```python\n#不計算\n```\n$$不計算$$\n$不計算$\n## 參考來源\n不計算'),2)

    def test_structure(self):
        self.assertTrue(be.validate_chapter('# 假標題',1,True))

    def test_part_review_cache_tracks_content_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'chapters').mkdir()
            for n in range(1,21):(root/'chapters'/f'{n:02d}.md').write_text(f'章節{n}')
            before=[be.part_fingerprint(root,p) for p in range(5)]
            proof=root/'review.md';proof.write_text('VERDICT: APPROVE')
            evidence={'fingerprint':before[0], 'approved':True, 'review_file':'review.md',
                      'review_sha256':be.hashlib.sha256(proof.read_bytes()).hexdigest()}
            self.assertTrue(be.reusable_part_review(root,evidence,before[0]))
            proof.write_text('改動後\nVERDICT: APPROVE')
            self.assertFalse(be.reusable_part_review(root,evidence,before[0]))
            (root/'chapters/19.md').write_text('修訂第19章')
            after=[be.part_fingerprint(root,p) for p in range(5)]
            self.assertEqual(before[:4],after[:4]);self.assertNotEqual(before[4],after[4])
            with patch('book_editor.CONVENTIONS',be.CONVENTIONS+'新契約'):
                self.assertNotEqual(after[0],be.part_fingerprint(root,0))

    def test_publication_links_only_for_current_manuscript(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);be.init_book(root)
            state={'status':'initialized','chapters':{}}
            be.build(root,state)
            pub=root/'published';pub.mkdir()
            (pub/'volume-1.html').write_text('test html')
            (pub/'volume-1.pdf').write_bytes(b'%PDF-test')
            (pub/'publication.json').write_text(json.dumps({'manuscript_sha256':be.hashlib.sha256((root/'book.md').read_bytes()).hexdigest()}))
            be.build(root,state)
            self.assertIn('/book/html',(root/'index.html').read_text())
            self.assertIn('/book/pdf',(root/'index.html').read_text())
            state['status']='changed'
            be.build(root,state)
            self.assertNotIn('/book/html',(root/'index.html').read_text())

    def test_upper_length_is_advisory(self):
        text='# 第09章 測試\n'+'\n'.join('## '+s for s in SECTIONS[:-1])
        text+='\n'+('字'*3000)+'\n```python\npass\n```\n## 參考來源\n'
        self.assertFalse(be.validate_chapter(text,9,True))
        self.assertTrue(be.validate_chapter(text.replace('字'*3000,'字'*1000),9,True))

    def test_overlong_book_can_complete(self):
        def fake(agent,context,options,instructions):
            if instructions != AUTHOR_RULES:
                if 'draft' in context:
                    self.assertTrue(context['length_requirement_passed'])
                    self.assertIn('允許超字',context['length_policy'])
                return '# 審稿\n'+('測試核對。'*30)+'\nVERDICT: APPROVE'
            return context['heading']+'\n'+'\n'.join('## '+s for s in SECTIONS[:-1])+'\n'+('字'*3000)+'\n```python\npass\n```\n## 參考來源\n'
        with tempfile.TemporaryDirectory() as temp, patch('llm_agent.decide',side_effect=fake), patch('book_editor.wait_remote_idle'), contextlib.redirect_stdout(io.StringIO()):
            be.run(argparse.Namespace(output=temp,build_only=False,wait_discussion=False,timeout=5))
            state=json.loads((Path(temp)/'status.json').read_text())
            self.assertGreater(state['word_count']['total'],55000)
            self.assertFalse(state['word_count']['upper_limit_enforced'])
            self.assertEqual(state['status'],'completed_model_review_pending_human')

    def test_complete_workflow_without_real_calls(self):
        calls=[]
        def fake(agent,context,options,instructions):
            calls.append(agent.id)
            if instructions!=AUTHOR_RULES:
                if 'draft' in context:
                    self.assertEqual(context['measured_chinese_characters'],be.prose_count(context['draft']))
                return '# 審稿\n\n'+('測試核對。'*30)+'\nVERDICT: APPROVE'
            # 僅測試夾具使用重複字元，絕不輸出到正式教材。
            return context['heading']+'\n\n'+'\n\n'.join('## '+s+'\n'+('測試'*117) for s in SECTIONS)+'\n```python\nprint(1)\n```'
        with tempfile.TemporaryDirectory() as temp, patch('llm_agent.decide',side_effect=fake), patch('book_editor.wait_remote_idle'), contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(output=temp,build_only=False,wait_discussion=False,timeout=5)
            be.run(args)
            state=json.loads((Path(temp)/'status.json').read_text())
            self.assertEqual(state['status'],'completed_model_review_pending_human')
            self.assertEqual(len(list((Path(temp)/'chapters').glob('*.md'))),20)
            self.assertEqual(len(calls),85)
            self.assertTrue(45000<=state['word_count']['total']<=55000)
            self.assertIn('figures/roadmap.svg',(Path(temp)/'book.md').read_text())
            be.run(args)
            self.assertEqual(len(calls),85)
            # 模擬兩章待修，其中一章被無效占位文字取代。應找回最佳有效版本。
            state['chapters']['2']['status']='needs_revision'
            state['chapters']['4']['status']='needs_revision'
            (Path(temp)/'chapters/04.md').write_text('I will check files later.')
            (Path(temp)/'status.json').write_text(json.dumps(state))
            old=(Path(temp)/'chapters/01.md').read_bytes()
            args.repair=True
            be.run(args)
            repaired=json.loads((Path(temp)/'status.json').read_text())
            self.assertEqual(repaired['status'],'completed_model_review_pending_human')
            self.assertEqual(repaired['repair_generation'],1)
            self.assertEqual((Path(temp)/'chapters/01.md').read_bytes(),old)
            self.assertFalse(be.validate_chapter((Path(temp)/'chapters/04.md').read_text(),4,True))
            self.assertTrue((Path(temp)/'backups/repair-001/chapters/04.md').exists())
            self.assertEqual(len(calls),87) # 兩章複審；復原後內容未變，沿用五部已通過審查
            # 定點修訂複審不得在拒稿後自動重寫章節。
            repaired['chapters']['4']['status']='needs_revision'
            (Path(temp)/'status.json').write_text(json.dumps(repaired))
            before=(Path(temp)/'chapters/04.md').read_bytes()
            args.review_only=True
            def reject(agent,context,options,instructions):
                self.assertNotEqual(instructions,AUTHOR_RULES)
                calls.append(agent.id)
                return '# 審稿\n'+('需再核對。'*30)+'\nVERDICT: REVISE'
            with patch('llm_agent.decide',side_effect=reject):be.run(args)
            self.assertEqual(len(calls),88)
            self.assertEqual((Path(temp)/'chapters/04.md').read_bytes(),before)
            rejected=json.loads((Path(temp)/'status.json').read_text())
            self.assertEqual(rejected['chapters']['4']['status'],'needs_revision')
            self.assertEqual(rejected['part_reviews'],{})


if __name__=='__main__':unittest.main()
