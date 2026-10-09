import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import book2_editor as B
import book2_spec as S


def draft(n):
    return f'# 第{n:02d}章 {S.CHAPTERS[n-1][0]}\n\n'+'\n\n'.join('## '+s+'\n'+('教材'*160) for s in S.SECTIONS[:-1])+'\n```python\nassert 1 + 1 == 2\n```\n## 參考來源\nG1\n'


class VolumeTwoTests(unittest.TestCase):
    def test_roster_and_scope(self):
        people=S.roster()
        self.assertEqual(len(people),40)
        self.assertEqual(len(S.CHAPTERS),30)
        for model in S.MODELS:self.assertEqual(sum(p['model']==model[1] for p in people),8)
        for n in range(1,31):
            a,r=S.chapter_people(n);self.assertNotEqual(a['model'],r['model'])
        self.assertFalse(B.validate(draft(1),1))
        self.assertTrue(B.validate(draft(1).replace('assert 1 + 1 == 2','if:'),1))

    def test_semantic_fence_extraction_and_heading_normalization(self):
        text=draft(13).replace('# 第13章','# 第十三章').replace('## 學習目標與先備知識','## 1. 學習目標與先備知識')
        text=text.replace('```python\nassert 1 + 1 == 2\n```','1. 清單中的程式\n\n   ```python\n   assert 1 + 1 == 2\n   ```')
        normalized=B.normalize_structure(text,13)
        self.assertFalse(B.validate(normalized,13))
        self.assertIn('   assert 1 + 1 == 2',normalized) # 正文保留；解析器依容器處理
        self.assertEqual(B.markdown_structure(normalized)['fences'][0]['content'],'assert 1 + 1 == 2\n')
        wrong=draft(12)
        self.assertTrue(B.validate(B.normalize_structure(wrong,13),13))
        invalid=draft(13).replace('assert 1 + 1 == 2','    assert True')
        self.assertTrue(B.validate(invalid,13)) # 無清單容器的真縮排錯誤不能被掩蓋

    def test_author_retry_receives_validation_feedback(self):
        attempts=[]
        def fake(person,context,options,instructions):
            attempts.append(context)
            if len(attempts)==1:return '不是有效章稿'
            self.assertIn('output_errors',context)
            self.assertEqual(context['previous_output'],'不是有效章稿')
            return draft(1)
        with tempfile.TemporaryDirectory() as temp,patch('book2_editor.llm_agent.decide',side_effect=fake):
            root=Path(temp);B.A.initialize(root)
            state={'status':'writing','calls':0,'chapters':{},'active':{}}
            editor=B.Editor(root,state,argparse.Namespace(timeout=5,max_calls=10))
            result=editor.invoke(S.chapter_people(1)[0],'test',editor.context(1),S.AUTHOR_RULES)
            self.assertFalse(B.validate(result,1));self.assertEqual(len(attempts),2)

    def test_refuse_other_volume_before_modifying_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'config.json').write_text(json.dumps({'title':'Volume I'}))
            (root/'TOC.md').write_text('original contents')
            args=argparse.Namespace(output=temp,init_only=True,timeout=5,max_calls=240)
            with self.assertRaises(ValueError):B.run(args)
            self.assertEqual((root/'TOC.md').read_text(),'original contents')

    def test_patches_unique_and_nonoverlapping(self):
        text=draft(1)
        fixed=B.checked_patches({'1':text},{'patches':[{'chapter':1,'old':'assert 1 + 1 == 2','new':'assert 2 + 2 == 4'}]})
        self.assertIn('assert 2 + 2 == 4',fixed['1'])
        with self.assertRaises(ValueError):B.checked_patches({'1':text},{'patches':[{'chapter':1,'old':'教材','new':'x'}]})
        with self.assertRaises(ValueError):B.checked_patches({'1':text},{'patches':[{'chapter':1,'old':'assert 1 + 1 == 2','new':'pass'},{'chapter':1,'old':'1 + 1','new':'2'}]})

    def test_targeted_repair_preserves_roles_and_skips_approved(self):
        calls=[]
        author,reviewer=S.chapter_people(4)
        def fake(person,context,options,instructions):
            calls.append((person.id,instructions))
            self.assertEqual(context['repair_generation'],1)
            if instructions==S.PATCH_RULES:
                self.assertNotIn(person.id,(author['id'],reviewer['id']))
                self.assertEqual(context['minimum_characters'],3000)
                return {'patches':[{'chapter':4,'old':'assert 1 + 1 == 2','new':'assert 2 + 2 == 4'}]}
            self.assertEqual(person.id,reviewer['id'])
            return '核對測試\nVERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch('book2_editor.llm_agent.decide',side_effect=fake):
            root=Path(temp);B.A.initialize(root)
            state={'status':'writing','calls':0,'chapters':{},'active':{},'repair_generation':1}
            editor=B.Editor(root,state,argparse.Namespace(timeout=5,max_calls=10))
            editor.save_chapter(4,draft(4),'needs_revision','指定修補\nVERDICT: REVISE')
            editor.targeted_chapter(4)
            self.assertEqual(state['chapters']['4']['status'],'model_reviewed')
            self.assertEqual([r for _,r in calls],[S.PATCH_RULES,S.REVIEW_RULES])
            editor.targeted_chapter(4)
            self.assertEqual(len(calls),2)
            self.assertTrue(list((root/'editorial/history').glob('04-*.md')))
            file=root/'chapters/04.md'
            file.write_text(file.read_text().replace('assert 2 + 2 == 4','assert 3 + 3 == 6'))
            editor.targeted_chapter(4) # 變更稿件不可沿用舊批准
            self.assertEqual(len(calls),3)

    def test_repair_generation_invalidates_failed_cache(self):
        with tempfile.TemporaryDirectory() as temp,patch('book2_editor.llm_agent.decide',return_value='待修\nVERDICT: REVISE') as fake:
            root=Path(temp);B.A.initialize(root)
            state={'status':'writing','calls':0,'chapters':{},'active':{},'repair_generation':1}
            editor=B.Editor(root,state,argparse.Namespace(timeout=5,max_calls=10))
            reviewer=S.chapter_people(4)[1]
            editor.invoke(reviewer,'probe',{'number':4},S.REVIEW_RULES)
            editor.invoke(reviewer,'probe',{'number':4},S.REVIEW_RULES)
            self.assertEqual(fake.call_count,1)
            state['repair_generation']=2
            editor.invoke(reviewer,'probe',{'number':4},S.REVIEW_RULES)
            self.assertEqual(fake.call_count,2)

    def test_unlimited_calls_and_review_format_feedback(self):
        replies=[]
        def fake(person,context,options,instructions):
            replies.append(context)
            if len(replies)==1:return '缺少結論的審稿'
            self.assertIn('review_format_error',context)
            return '已核對\nVERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch('book2_editor.llm_agent.decide',side_effect=fake):
            root=Path(temp);B.A.initialize(root)
            state={'status':'writing','calls':999,'chapters':{},'active':{}}
            editor=B.Editor(root,state,argparse.Namespace(timeout=5,max_calls=0))
            editor.invoke(S.chapter_people(4)[1],'format-test',{'number':4},S.REVIEW_RULES)
            self.assertEqual(state['calls'],1001)
            self.assertTrue((root/'editorial/rejected-review-001000.txt').exists())

    def test_full_workflow_and_automatic_chief_repair(self):
        def fake(person,context,options,instructions):
            if instructions==S.AUTHOR_RULES:return draft(context['number'])
            if instructions==S.PATCH_RULES:
                return {'patches':[{'chapter':1,'old':f'# 第01章 {S.CHAPTERS[0][0]}','new':f'# 第01章 {S.CHAPTERS[0][0]}（修訂）'}]}
            if 'chapters' in context and '1' in context['chapters'] and '（修訂）' not in context['chapters']['1']:
                return '定點修正測試\nVERDICT: REVISE'
            return '核對測試\nVERDICT: APPROVE'
        with tempfile.TemporaryDirectory() as temp,patch('book2_editor.llm_agent.decide',side_effect=fake),patch('book2_editor.wait_remote_idle'),patch('book2_editor.active_discussions',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(output=temp,init_only=False,timeout=5,max_calls=240)
            B.run(args)
            root=Path(temp);state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'completed_model_review_pending_human')
            self.assertEqual(state['calls'],68)
            self.assertEqual(len(state['chapters']),30)
            self.assertTrue(all(v['approved'] for v in state['part_reviews'].values()))
            self.assertLessEqual(state['word_count']['total'],200000)
            self.assertTrue(list((root/'editorial/history').glob('01-*.md')))
            B.run(args)
            self.assertEqual(json.loads((root/'status.json').read_text())['calls'],68)
            state=json.loads((root/'status.json').read_text())
            editor=B.Editor(root,state,args)
            with patch.object(editor,'invoke',side_effect=AssertionError('不應重做已通過的同版跨章審稿')):
                editor.cross_review(0)
            with patch.object(editor,'invoke',return_value='新契約核對\nVERDICT: APPROVE') as called,patch.object(S,'REVIEW_RULES',S.REVIEW_RULES+'\n新契約'):
                editor.cross_review(0)
                self.assertEqual(called.call_count,1)


if __name__=='__main__':unittest.main()
