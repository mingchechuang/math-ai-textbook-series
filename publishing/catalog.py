"""Hash-audit every volume and build an offline-capable series index. No chapter code runs."""
import argparse
import base64
import subprocess
import tempfile
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

BOOKS={1:'linear-algebra-aquaculture',2:'modern-graphics',3:'field-simulation',4:'calculus-analysis',5:'neural-transformers'}
ROMAN={1:'I',2:'II',3:'III',4:'IV',5:'V'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def audit_volume(workspace,volume):
    root=Path(workspace)/'books'/BOOKS[volume];out=root/'published'
    manifest=json.loads((out/'publication.json').read_text())
    state=json.loads((root/'status.json').read_text())
    qa=json.loads((out/'validation.json').read_text())
    count=20 if volume==1 else 30
    entries=[('preface','00-preface.md')]+[(f'ch{n:02d}',f'chapters/{n:02d}.md') for n in range(1,count+1)]+[('appendix','99-appendix.md'),('references','REFERENCES.md')]
    expected={file for _,file in entries}|{str(p.relative_to(root)) for p in (root/'figures').glob('*.svg')}
    require(state['status']=='completed_model_review_pending_human','Model review incomplete')
    require(manifest['source_status']==state['status'],'Source status changed')
    require(manifest.get('volume')==volume and manifest.get('chapter_count')==count,'Wrong volume/chapter metadata')
    require(set(manifest['source_files'])==expected,'Incomplete source inventory')
    for file,digest in manifest['source_files'].items():require(sha(root/file)==digest,f'Stale source: {file}')
    require(manifest['manuscript_sha256']==sha(root/'book.md'),'Stale manuscript')
    source=[{'id':id,'file':file,'text':(root/file).read_text()} for id,file in entries]
    encoded=json.dumps({'source':source,'status':state['status'],'word_count':state['word_count']},ensure_ascii=False,separators=(',',':')).encode()
    require(hashlib.sha256(encoded).hexdigest()==manifest['source_sha256'],'Combined source digest mismatch')
    require(manifest['characters']==state['word_count']['total'],'Character count changed')
    inspected=manifest['inspection']
    require(not any(inspected[k] for k in ('missingLinks','overflow','unrenderedDollars')),'HTML layout/math/link check failed')
    require(inspected['cjkFontLoaded'] and inspected['chapterCount']==count,'Font/chapter check failed')
    require(not manifest['external_requests'],'Offline rendering used network')
    require(manifest['mobile']['bodyWidth']<=manifest['mobile']['viewport']+2,'Mobile overflow')
    require(json.loads((out/'math-errors.json').read_text())==[],'Math errors')
    artifacts={}
    for suffix in ('html','pdf'):
        path=out/f'volume-{volume}.{suffix}'
        digest=sha(path);size=path.stat().st_size
        require(digest==manifest[f'{suffix}_sha256'],f'{suffix} artifact hash mismatch')
        require(size==manifest['files'][suffix]==qa[f'{suffix}_bytes'],f'{suffix} artifact size mismatch')
        artifacts[suffix]={'path':str(path.relative_to(Path(workspace)/'books')),'bytes':size,'sha256':digest}
    require(qa['volume']==volume and qa['pdf_sha256']==artifacts['pdf']['sha256'],'Stale PDF validation')
    require(not qa['blank_pages'] and not qa['missing_glyph_pages'],'PDF text check failed')
    require(len(qa['chapter_pages'])==count,'Missing PDF chapter pages')
    config=json.loads((root/'config.json').read_text())
    text=(out/f'volume-{volume}.html').read_text()
    require(html.escape(config['title'],quote=False) in text and state['updated'] in text,'Cover title/version stale')
    return {'volume':volume,'roman':ROMAN[volume],'title':config['title'],'source_updated':state['updated'],
            'built_at':manifest['built_at'],'characters':manifest['characters'],'pages':qa['pages'],
            'chapters':count,'math_count':manifest['mathCount'],'fresh':True,'artifacts':artifacts,
            'manifest_sha256':sha(out/'publication.json'),'validation_sha256':sha(out/'validation.json'),
            'manuscript_sha256':manifest['manuscript_sha256'],'source_sha256':manifest['source_sha256'],
            'source_file_count':len(expected)}


def render(report):
    esc=html.escape;cards=[]
    for v in report['volumes']:
        links=' '.join(f'<a class="button" href="{esc(v["artifacts"][kind]["path"])}">{label}</a>' for kind,label in (('html','閱讀 HTML'),('pdf','開啟／下載 PDF')))
        title=v['title'] if v['title'].startswith('Volume '+v['roman']+'｜') else f'Volume {v["roman"]}｜{v["title"]}'
        cards.append(f'<article><h2>{esc(title)}</h2><p>{v["chapters"]}章 · {v["characters"]:,}中文字 · PDF {v["pages"]}頁</p><p class="links">{links}</p><p class="meta">原稿版本：{esc(v["source_updated"])}<br>出版時間：{esc(v["built_at"])}</p><p class="ok">檢查時與最新原稿一致 ✓</p></article>')
    return f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>數學與AI教材｜五卷出版索引</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f8;color:#16334b;font-family:system-ui,"Noto Sans CJK TC",sans-serif;line-height:1.8}}main{{max-width:1040px;margin:auto;padding:28px 20px}}h1{{line-height:1.4}}h2{{font-size:1.25rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,390px),1fr));gap:18px}}article{{background:white;border:1px solid #c8d8e5;padding:20px;border-radius:12px}}.links{{display:flex;gap:12px;flex-wrap:wrap}}a{{color:#155c98}}.button{{padding:9px 15px;background:#164e77;color:white;border-radius:6px;text-decoration:none}}.meta{{font-size:.85rem;overflow-wrap:anywhere;color:#506477}}.ok{{color:#176039}}footer{{margin-top:24px}}a:focus-visible{{outline:3px solid #dc8b00;outline-offset:3px}}
</style></head><body><main><h1>數學、AI與模擬教材</h1><p>Volumes I–V · 全文HTML與PDF</p><p>五卷均已通過逐章與跨章模型審稿，仍待人工內容覆核與全面程式驗證。出版檢查不等於數學正確性保證。</p><p>版本核對時間：<strong>{esc(report['checked_at'])}</strong>。已比對原稿、前言、附錄、參考資料、SVG及HTML／PDF的SHA-256，並檢查排版與PDF結構。下載的HTML可離線閱讀；同卷PDF放在相同資料夾即可使用內頁PDF連結。</p><div class="grid">{''.join(cards)}</div><footer><a href="publication-status.json">下載完整版本與雜湊核對紀錄</a><p>本頁為上述時間的驗證快照，原稿日後若修改，須重新出版及核對。</p></footer></main></body></html>'''


def build(workspace,embed_font=True):
    workspace=Path(workspace).resolve()
    report={'checked_at':datetime.now().astimezone().isoformat(timespec='seconds'),
            'scope':'Source/artifact SHA-256, HTML render QA and PDF structure; not human content validation.',
            'volumes':[audit_volume(workspace,v) for v in BOOKS]}
    document=render(report)
    if embed_font:
        tools=workspace/'publishing'
        license_text=(tools/'fonts/OFL.txt').read_text()
        document=document.replace('</footer>','<details><summary>內嵌字型授權（Noto Sans CJK TC／SIL OFL 1.1）</summary><pre style="white-space:pre-wrap">'+html.escape(license_text)+'</pre></details></footer>')
        with tempfile.TemporaryDirectory(prefix='catalog-font-') as temp:
            temp=Path(temp);(temp/'text.txt').write_text(document)
            subprocess.run([str(tools/'.venv/bin/python'),'-m','fontTools.subset',str(tools/'fonts/NotoSansCJKtc-Regular.otf'),f'--text-file={temp}/text.txt','--flavor=woff2',f'--output-file={temp}/index.woff2'],check=True)
            font=base64.b64encode((temp/'index.woff2').read_bytes()).decode()
        document=document.replace('<style>','<style>@font-face{font-family:CatalogCJK;src:url(data:font/woff2;base64,'+font+") format('woff2');font-display:block}")
        document=document.replace('font-family:system-ui,','font-family:CatalogCJK,system-ui,')
    # Commit only after all five pass. A failed audit never labels a partial set current.
    for name,text in (('publication-status.json',json.dumps(report,ensure_ascii=False,indent=2)),('index.html',document)):
        target=workspace/'books'/name;temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_text(text);temporary.replace(target)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    print(json.dumps(build(args.workspace),ensure_ascii=False,indent=2))
