import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from book4_editor import ENGINE as E
from recover_book4 import recover,sections_of_chapter
from test_book4_editor import draft


def setup_root(root):
    E.run(argparse.Namespace(output=str(root),init_only=True,max_calls=0,timeout=1))
    s=json.loads((root/'status.json').read_text())
    for n in range(1,31):
        text=draft(n);(root/'chapters'/f'{n:02d}.md').write_text(text)
        s['chapters'][str(n)]={'status':'error' if n in (9,19,29) else 'model_reviewed','sha256':E.digest(text)}
    E.atomic(root/'status.json',json.dumps(s))


def fake(self,person,phase,context,rules,mode='markdown'):
    if phase.endswith('-review'):return '測試複審\nVERDICT: APPROVE'
    chunks=[]
    for name in context['requested_sections']:
        text='## '+name+'\n'+'數學教材'*160+'\n'
        if name=='實作與程式':text+='```python\nassert True\n```\n'
        chunks.append(text)
    return '\n'.join(chunks)


class RecoveryTests(unittest.TestCase):
    def test_preserve_27_chapters_and_correct_hand_examples(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',new=fake),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);setup_root(root)
            before={p.name:p.read_bytes() for p in (root/'chapters').glob('*.md')}
            old_hand=sections_of_chapter(before['19.md'].decode())['逐步手算例題']
            recover(root,1)
            s=json.loads((root/'status.json').read_text());self.assertEqual(s['status'],'ready_for_cross_review')
            for name,text in before.items():
                if name not in ('09.md','19.md','29.md'):self.assertEqual((root/'chapters'/name).read_bytes(),text)
            self.assertEqual(sections_of_chapter((root/'chapters/19.md').read_text())['逐步手算例題'],old_hand)
            self.assertTrue((root/'editorial/ch29-segmented-candidate.md').exists())
            self.assertEqual(s['active'],{})

    def test_invalid_group_does_not_replace_chapter(self):
        def bad(self,person,phase,context,rules,mode='markdown'):
            if phase.endswith('-review'):return 'VERDICT: APPROVE'
            return '# Wrong heading\n'
        with tempfile.TemporaryDirectory() as temp,patch.object(E,'active_discussions',return_value=[]),patch.object(E.Editor,'invoke',new=bad),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);setup_root(root);old=(root/'chapters/19.md').read_bytes()
            with self.assertRaises(ValueError):recover(root,1)
            self.assertEqual((root/'chapters/19.md').read_bytes(),old)
            s=json.loads((root/'status.json').read_text());self.assertEqual(s['status'],'needs_editorial_attention')
            self.assertTrue(s['attention']);self.assertEqual(s['active'],{})


if __name__=='__main__':unittest.main()
