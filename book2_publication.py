"""Volume II出版入口：只連結與現行稿件及PDF驗證紀錄一致的成品。"""
import hashlib
import json
from pathlib import Path


def fresh_publication(root,volume=2):
    root=Path(root).resolve();out=root/'published'
    try:
        manifest=json.loads((out/'publication.json').read_text())
        validation=json.loads((out/'validation.json').read_text())
        if manifest.get('volume')!=volume or manifest.get('chapter_count')!=30:return False
        if manifest.get('manuscript_sha256')!=hashlib.sha256((root/'book.md').read_bytes()).hexdigest():return False
        sources=manifest.get('source_files',{})
        expected={'00-preface.md','99-appendix.md','REFERENCES.md'}|{f'chapters/{n:02d}.md' for n in range(1,31)}
        if not expected.issubset(sources):return False
        for relative,sha in sources.items():
            file=(root/relative).resolve()
            if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:return False
        if validation.get('volume')!=volume or validation.get('pdf_sha256')!=manifest.get('pdf_sha256'):return False
        if validation.get('blank_pages')!=[] or validation.get('missing_glyph_pages')!=[]:return False
        for kind in ('html','pdf'):
            if (out/f'volume-{volume}.{kind}').stat().st_size!=manifest['files'][kind]:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def inject_publication(page,root,volume=2):
    if volume not in (2,3,4,5):raise ValueError('unsupported publication volume')
    if not fresh_publication(root,volume) or f'id="book{volume}-publication"' in page:return page
    links=f'<p id="book{volume}-publication"><b>圖文出版版：</b><a href="/book{volume}/html">HTML閱讀版</a> ｜ <a href="/book{volume}/pdf">PDF列印版</a> ｜ <a href="/book{volume}/download/html">下載HTML（離線閱讀）</a></p>'
    if '<div class="status">' in page:return page.replace('<div class="status">',links+'<div class="status">',1)
    return links+page
