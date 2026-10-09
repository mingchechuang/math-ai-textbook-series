"""Curated repository export. Excludes raw sessions, thinking/tool events and private settings."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

BOOKS=('linear-algebra-aquaculture','modern-graphics','field-simulation','calculus-analysis','neural-transformers')
SECRET_PATTERNS=[re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}'),re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}'),re.compile(r'\bsk-[A-Za-z0-9_-]{24,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'\bBearer [A-Za-z0-9_.-]{32,}')]
TEXT_EXT={'.md','.txt','.json','.jsonl','.py','.mjs','.js','.ts','.html','.svg','.yml','.yaml'}


def clean(text):
    for pattern in SECRET_PATTERNS:text=pattern.sub('[REDACTED_CREDENTIAL]',text)
    return text


def audit(root):
    files=[]
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file() or '.git' in p.parts:continue
        if p.is_symlink():raise ValueError('Symlink not allowed: '+str(p.relative_to(root)))
        if p.stat().st_size>=95*1024**2:raise ValueError('File too large: '+str(p.relative_to(root)))
        if p.name in ('auth.json','.env','.netrc','.git-credentials') or p.suffix in ('.pem','.key'):
            raise ValueError('Private file not allowed: '+str(p.relative_to(root)))
        raw=p.read_bytes()
        if p.suffix in TEXT_EXT:
            text=raw.decode('utf-8')
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):raise ValueError('Potential credential: '+str(p.relative_to(root)))
        files.append({'path':str(p.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    return files


def export(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Destination already exists; refusing overwrite')
    destination.mkdir(parents=True)
    def copy(p):
        target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('index.html','publication-status.json','SERIES_ROADMAP.md'):copy(source/'books'/name)
    counts={}
    for folder in BOOKS:
        root=source/'books'/folder
        for p in root.iterdir():
            if p.is_file() and p.suffix in ('.md','.json','.html'):copy(p)
        for sub in ('chapters','figures','examples','data','editorial','published'):
            for p in (root/sub).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp'):copy(p)
        # Only completed user/assistant text messages. Never copy raw session/event files.
        records=0;redactions=0
        for p in sorted((root/'agents').glob('*/pi_events/**/*.jsonl')):
            messages=[]
            for line in p.open():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')!='message_end':continue
                m=event.get('message',{})
                if m.get('role') not in ('user','assistant'):continue
                text='\n'.join(c.get('text','') for c in m.get('content',[]) if isinstance(c,dict) and c.get('type')=='text')
                if not text:continue
                safe=clean(text);redactions+=int(safe!=text)
                messages.append({'role':m['role'],'text':safe,**{k:m[k] for k in ('timestamp','provider','model','stopReason') if k in m}})
            if messages:
                target=destination/'discussions'/folder/p.relative_to(root/'agents')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in messages));records+=len(messages)
        counts[folder]={'completed_text_messages':records,'redacted_messages':redactions}
    patterns=('book*.py','continue_book*.py','recover_book*.py','test_book*.py','test_recover_book*.py','test_publication_catalog.py','test_upload_github.py')
    for pattern in patterns:
        for p in source.glob(pattern):copy(p)
    for name in ('llm_agent.py','taiwan_discussion.py','discussion_demo.py','marketsim.py','report.py'):copy(source/name)
    for sub in ('publishing','extensions'):
        for p in (source/sub).rglob('*'):
            if p.is_file() and not any(x in p.parts for x in ('node_modules','.venv','__pycache__')):copy(p)
    for name in ('MULTI_AGENT_FRAMEWORK_REVIEW.md',):copy(source/'docs'/name)
    shutil.copy2(source/'publishing/REPOSITORY_README.md',destination/'README.md')
    (destination/'.gitignore').write_text('.venv/\nnode_modules/\n__pycache__/\n*.pyc\n.env\nauth.json\n*.pem\n*.key\n.git-credentials\nruns/\n**/agents/\n**/backups/\n**/.publish-*/\n*.sqlite*\n*.lock\n')
    (destination/'discussions/README.md').write_text('''# 教材寫作與審稿討論紀錄\n\n依卷、角色／任務及原始事件檔名整理。JSONL保留已完成的user／assistant文字訊息及模型標示，包含寫作提示、草稿、審稿及修訂討論；不是原始session的完整逐位元複本。\n\n刻意排除system訊息、thinking內容、工具呼叫／回傳、原始pi_sessions、私人設定與憑證；常見token格式會遮蔽。各卷`books/*/editorial/`另保留審稿結果、修補候選及人工修復說明。未包含交易／台灣收入討論，也未包含此助理主對話；其完整逐字紀錄不在教材工作目錄中。\n''')
    summary={'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),'discussions':counts,
             'excluded':['raw agents/pi_sessions and pi_events','thinking/system/tool events','credentials and private settings','backups and failed publication staging','node_modules and virtualenvs','unrelated trading/discussion data','parent assistant chat'],
             'note':'Only this export directory is intended for upload. Review before making repository public.'}
    (destination/'EXPORT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    files=audit(destination)
    (destination/'EXPORT_MANIFEST.json').write_text(json.dumps({'files':files},ensure_ascii=False,indent=2))
    print(json.dumps({'destination':str(destination),'files':len(files),'bytes':sum(f['bytes'] for f in files),'discussions':counts},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();export(a.source,a.destination)
