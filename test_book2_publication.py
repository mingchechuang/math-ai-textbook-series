import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from book2_publication import fresh_publication,inject_publication


class PublicationLinksTests(unittest.TestCase):
    def test_freshness_and_validation_required(self):
        for volume in (2,3,4):
            with self.subTest(volume=volume):self.check_volume(volume)

    def check_volume(self,volume):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'published';out.mkdir();(root/'chapters').mkdir()
            (root/'book.md').write_text('原稿')
            sources={}
            for name in ['00-preface.md','99-appendix.md','REFERENCES.md']+[f'chapters/{n:02d}.md' for n in range(1,31)]:
                (root/name).write_text(name);sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            for kind in ['html','pdf']:(out/f'volume-{volume}.{kind}').write_bytes(b'test')
            manifest={'volume':volume,'chapter_count':30,'source_files':sources,'manuscript_sha256':hashlib.sha256((root/'book.md').read_bytes()).hexdigest(),'pdf_sha256':'test','files':{'html':4,'pdf':4}}
            (out/'publication.json').write_text(json.dumps(manifest))
            self.assertFalse(fresh_publication(root,volume))
            (out/'validation.json').write_text(json.dumps({'volume':volume,'pdf_sha256':'test','blank_pages':[],'missing_glyph_pages':[]}))
            self.assertTrue(fresh_publication(root,volume))
            self.assertFalse(fresh_publication(root,3 if volume==2 else 2))
            page=inject_publication('<html><div class="status">完成</div></html>',root,volume)
            self.assertIn(f'/book{volume}/pdf',page);self.assertEqual(inject_publication(page,root,volume),page)
            (root/'chapters/01.md').write_text('變更章稿但尚未組合主稿')
            self.assertFalse(fresh_publication(root,volume))
            self.assertEqual(inject_publication('原頁',root,volume),'原頁')


if __name__=='__main__':unittest.main()
