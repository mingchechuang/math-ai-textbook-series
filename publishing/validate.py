"""Validate the generated PDF and write bounded publication QA metadata."""
import argparse
import re
import hashlib
import json
from pathlib import Path
import pymupdf

parser = argparse.ArgumentParser()
parser.add_argument('--volume', type=int, choices=(1, 2, 3, 4, 5), default=1)
args = parser.parse_args()
chapter_count = 20 if args.volume == 1 else 30
stem = f'volume-{args.volume}'
root = Path(__file__).resolve().parent.parent / ('books/' + {1:'linear-algebra-aquaculture',2:'modern-graphics',3:'field-simulation',4:'calculus-analysis',5:'neural-transformers'}[args.volume])
out = root / 'published'
manifest = json.loads((out/'publication.json').read_text())
assert manifest['manuscript_sha256'] == hashlib.sha256((root/'book.md').read_bytes()).hexdigest()
for file, expected in manifest.get('source_files', {}).items():
    assert expected == hashlib.sha256((root/file).read_bytes()).hexdigest(), file
for name in ['html','pdf']:
    assert manifest[f'{name}_sha256'] == hashlib.sha256((out/f'{stem}.{name}').read_bytes()).hexdigest()
assert json.loads((out/'math-errors.json').read_text()) == []
assert manifest['inspection']['missingLinks'] == []
assert manifest['inspection']['overflow'] == []
with pymupdf.open(out/f'{stem}.pdf') as pdf:
    blank = [i+1 for i,page in enumerate(pdf) if len(page.get_text().strip()) < 35]
    missing = [i+1 for i,page in enumerate(pdf) if '\0' in page.get_text()]
    bookmarks = pdf.get_toc()
    chapter_pages = {}
    for n in range(1,chapter_count+1):
        matches = [p for level,title,p in bookmarks if re.match(r'^第\s*0?'+str(n)+r'\s*章', title)]
        assert matches, f'Chapter {n} missing from PDF bookmarks'
        chapter_pages[n] = matches[-1]
    assert not blank, blank
    assert not missing, missing
    assert list(chapter_pages.values()) == sorted(set(chapter_pages.values()))
    report = {'volume':args.volume,'pages':len(pdf),'bookmarks':len(bookmarks),'chapter_pages':chapter_pages,
              'blank_pages':blank,'missing_glyph_pages':missing,
              'html_bytes':(out/f'{stem}.html').stat().st_size,
              'pdf_bytes':(out/f'{stem}.pdf').stat().st_size,
              'pdf_sha256':hashlib.sha256((out/f'{stem}.pdf').read_bytes()).hexdigest(),
              'scope':'Structure, text extraction, glyph markers and formula rendering; not mathematical validation.'}
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
