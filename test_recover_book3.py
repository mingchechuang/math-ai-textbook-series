import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from book3_editor import ENGINE as E
from recover_book3 import recover,split_sections,group
from test_book3_editor import draft


class RecoveryTests(unittest.TestCase):
    def test_sections_ignore_code_headings(self):
        text='## 實作與程式\n```python\n## not a heading\npass\n```\n## 測試與預期結果\n內容\n'
        self.assertEqual(list(split_sections(text)),['實作與程式','測試與預期結果'])

    def test_reject_wrapper_duplicate_or_preface(self):
        for text in ['# 第19章\n## 測試\n','## 測試\na\n## 測試\nb\n','前言\n## 測試\n']:
            with self.assertRaises(ValueError):split_sections(text)

    def test_recovery_preserves_other_chapters(self):
        def fake(self,person,phase,context,rules,mode='markdown'):
            if phase.endswith('-review'):return '测试審稿\nVERDICT: APPROVE'
            sections=[]
            for name in context['requested_sections']:
                text='## '+name+'\n'+'科學模型'*125+'\n'
                if name=='實作與程式':text+='```python\nassert True\n```\n'
                sections.append(text)
            return '\n'.join(sections)
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',new=fake),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            E.run(argparse.Namespace(output=temp,init_only=True,max_calls=0,timeout=1))
            state=json.loads((root/'status.json').read_text())
            for n in range(1,31):
                text=draft(n);(root/'chapters'/f'{n:02d}.md').write_text(text)
                state['chapters'][str(n)]={'status':'needs_revision' if n in (17,19) else 'model_reviewed','sha256':E.digest(text)}
            E.atomic(root/'status.json',json.dumps(state))
            before={n:(root/'chapters'/f'{n:02d}.md').read_bytes() for n in range(1,31)}
            recover(root,timeout=1)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'ready_for_cross_review')
            self.assertEqual(state['chapters']['19']['status'],'model_reviewed')
            for n in range(1,31):
                if n not in (17,19):self.assertEqual((root/'chapters'/f'{n:02d}.md').read_bytes(),before[n])
            self.assertTrue((root/'editorial/ch19-segmented-candidate.md').exists())
            self.assertEqual(state['active'],{})


if __name__=='__main__':unittest.main()
