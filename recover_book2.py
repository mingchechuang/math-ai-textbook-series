"""回收Volume II的拒收稿，只正規化格式；保留真錯誤供複審，不呼叫模型。"""
import fcntl
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import book2_editor as B
import book2_spec as S


def recover(root):
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((root/'config.json').read_text())
        if config['title']!=S.TITLE:raise ValueError('不是Volume II，拒絕修改')
        state=json.loads((root/'status.json').read_text())
        if state.get('active'):raise ValueError('仍有執行中標記，先核對程序狀態')
        backup=root/'backups'/('recovered-'+datetime.now().strftime('%Y%m%d-%H%M%S%f'))
        backup.mkdir(parents=True)
        shutil.copytree(root/'chapters',backup/'chapters')
        shutil.copy2(root/'status.json',backup/'status.json')
        records=[]
        for key,entry in state['chapters'].items():
            n=int(key)
            if entry.get('status')!='error':continue
            candidates=[]
            for error in (root/'editorial').glob(f'error-ch{n:02d}-draft-*.json'):
                match=re.search(r'-draft-(\d+)-\d+\.json$',error.name)
                if not match:continue
                call=int(match[1]);file=root/'editorial'/f'rejected-{call:06d}.txt'
                if not file.exists():continue
                raw=file.read_text();text=B.normalize_structure(raw,n)
                checks=B.validate(text,n)
                score=(len(B.validate(text,n,False)),len(checks),-B.prose_count(text),-call)
                candidates.append((score,file,text,checks))
            if not candidates:
                records.append({'chapter':n,'recovered':False});continue
            _,file,text,checks=min(candidates,key=lambda x:x[0])
            B.atomic(root/'chapters'/f'{n:02d}.md',text)
            state['chapters'][key]={'status':'needs_review','model_approved':False,'count':B.prose_count(text),'sha256':B.digest(text),'automatic_issues':checks,'recovered_from':str(file.relative_to(root)),'author':S.chapter_people(n)[0],'reviewer':S.chapter_people(n)[1]}
            records.append({'chapter':n,'recovered':True,'source':str(file.relative_to(root)),'characters':B.prose_count(text),'remaining_issues':checks})
        # 已存在稿件若人工修訂過，不沿用舊通過標記。
        for key,entry in state['chapters'].items():
            file=root/'chapters'/f'{int(key):02d}.md'
            if file.exists() and entry.get('sha256')!=B.digest(file.read_text()):
                entry.update(status='needs_review',model_approved=False,sha256=B.digest(file.read_text()),automatic_issues=B.validate(file.read_text(),int(key)))
        state.update(status='recovered_ready_for_review',attention=[],part_reviews={})
        B.atomic(backup/'recovery-report.json',json.dumps(records,ensure_ascii=False,indent=2))
        B.atomic(root/'editorial/recovery-report.json',json.dumps(records,ensure_ascii=False,indent=2))
        B.build(root,state)
        return records


if __name__=='__main__':
    print(json.dumps(recover(S.ROOT),ensure_ascii=False,indent=2))
