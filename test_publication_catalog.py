"""Synthetic metadata fixtures test audit logic; they are not real PDF validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from publishing.catalog import BOOKS,audit_volume,build,render,sha


def fixture(workspace,v):
    root=workspace/'books'/BOOKS[v];out=root/'published';out.mkdir(parents=True)
    (root/'chapters').mkdir();(root/'figures').mkdir()
    count=20 if v==1 else 30
    entries=[('preface','00-preface.md')]+[(f'ch{n:02d}',f'chapters/{n:02d}.md') for n in range(1,count+1)]+[('appendix','99-appendix.md'),('references','REFERENCES.md')]
    for _,name in entries:(root/name).write_text('# 範例\n')
    (root/'figures/demo.svg').write_text('<svg/>');(root/'book.md').write_text('# 完整稿')
    state={'status':'completed_model_review_pending_human','updated':'2026-10-08T00:00:00','word_count':{'total':12345}}
    (root/'status.json').write_text(json.dumps(state));(root/'config.json').write_text(json.dumps({'title':'測試教材'}))
    source=[{'id':id,'file':file,'text':(root/file).read_text()} for id,file in entries]
    digest=hashlib.sha256(json.dumps({'source':source,'status':state['status'],'word_count':state['word_count']},ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    (out/f'volume-{v}.html').write_text('測試教材 '+state['updated']);(out/f'volume-{v}.pdf').write_bytes(b'fake PDF for audit unit test only')
    manifest={'volume':v,'chapter_count':count,'source_status':state['status'],'source_files':{f:sha(root/f) for f in [x[1] for x in entries]+['figures/demo.svg']},'manuscript_sha256':sha(root/'book.md'),'source_sha256':digest,'characters':12345,'inspection':{'missingLinks':[],'overflow':[],'unrenderedDollars':[],'cjkFontLoaded':True,'chapterCount':count},'external_requests':[],'mobile':{'viewport':390,'bodyWidth':390},'files':{},'mathCount':10,'built_at':'2026-10-08T00:00:00Z'}
    qa={'volume':v,'blank_pages':[],'missing_glyph_pages':[],'chapter_pages':{str(n):n for n in range(1,count+1)},'pages':count}
    for suffix in ('html','pdf'):
        p=out/f'volume-{v}.{suffix}';manifest[f'{suffix}_sha256']=sha(p);manifest['files'][suffix]=p.stat().st_size;qa[f'{suffix}_bytes']=p.stat().st_size
    qa['pdf_sha256']=manifest['pdf_sha256']
    (out/'publication.json').write_text(json.dumps(manifest));(out/'validation.json').write_text(json.dumps(qa));(out/'math-errors.json').write_text('[]')
    return root


class CatalogTests(unittest.TestCase):
    def test_all_volumes_and_offline_links(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t)
            for v in BOOKS:fixture(w,v)
            report=build(w,embed_font=False);self.assertEqual(len(report['volumes']),5)
            text=(w/'books/index.html').read_text()
            for v in report['volumes']:
                for artifact in v['artifacts'].values():
                    self.assertIn('href="'+artifact['path']+'"',text)
                    self.assertTrue((w/'books'/artifact['path']).is_file())
            self.assertIn('驗證快照',text)

    def test_stale_chapter_figure_artifact_or_qa_rejected(self):
        for name in ('chapters/16.md','figures/demo.svg','published/volume-5.pdf','book.md','published/validation.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as t:
                w=Path(t);root=fixture(w,5);p=root/name
                if name.endswith('validation.json'):
                    q=json.loads(p.read_text());q['pdf_sha256']='bad';p.write_text(json.dumps(q))
                else:p.write_bytes(p.read_bytes()+b'changed')
                with self.assertRaises(ValueError):audit_volume(w,5)

    def test_missing_source_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);root=fixture(w,1);p=root/'published/publication.json';m=json.loads(p.read_text());m['source_files'].pop('figures/demo.svg');p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):audit_volume(w,1)

    def test_failed_build_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t)
            for v in BOOKS:fixture(w,v)
            build(w,embed_font=False);p=w/'books/index.html';before=p.read_bytes()
            (w/'books'/BOOKS[5]/'book.md').write_text('changed')
            with self.assertRaises(ValueError):build(w,embed_font=False)
            self.assertEqual(before,p.read_bytes())

    def test_title_html_escaped(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);fixture(w,5);v=audit_volume(w,5);v['title']='<script>alert(1)</script>'
            text=render({'checked_at':'now','volumes':[v]})
            self.assertNotIn('<script>',text);self.assertIn('&lt;script&gt;',text)


if __name__=='__main__':unittest.main()
