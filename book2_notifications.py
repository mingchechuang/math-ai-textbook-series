"""唯讀觀測Volume II工作；通知獨立保存，不修改稿件、status或模型程序。"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from book_editor import atomic

PANEL='''<section id="book2-notifications"><h2>進度通知</h2>
<p>階段完成、逾時及停止紀錄；<a href="/book2/notifications" target="_blank" rel="noopener">獨立開啟</a> ｜ <a href="/book2/notifications.json">最近通知JSON</a>。只在網頁內顯示，不是桌面推播。</p>
<iframe title="Volume II進度通知" src="/book2/notifications" style="width:100%;height:390px;border:1px solid #bbb;border-radius:8px"></iframe></section>'''


def inject_panel(page,base_route='/book2',label='Volume II'):
    if base_route not in ('/book2','/book3','/book4','/book5'):raise ValueError('unknown book route')
    identifier=base_route[1:]+'-notifications'
    if f'id="{identifier}"' in page:return page
    panel=PANEL.replace('/book2',base_route).replace('book2-notifications',identifier).replace('Volume II進度通知',html.escape(label)+'進度通知')
    marker='<p>初步目標'
    if marker in page:return page.replace(marker,panel+marker,1)
    if '</html>' in page:return page.replace('</html>',panel+'</html>',1)
    return page+panel


def worker_pids(root):
    """只比對這個output目錄的編輯器／續跑程序，不把通知程序算成工作。"""
    root=Path(root).resolve();found=[]
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=file.read_bytes().decode().split('\0')
            if not any(Path(a).name in ('book2_editor.py','continue_book2_review.py','book3_editor.py','continue_book3_review.py','book4_editor.py','continue_book4_review.py','book5_editor.py','continue_book5_review.py') for a in args[:3]):continue
            if '--init-only' in args:continue
            cwd=(file.parent/'cwd').resolve()
            scripts={Path(a).name for a in args[:3]}
            default=('books/neural-transformers' if scripts & {'book5_editor.py','continue_book5_review.py'} else
                     'books/calculus-analysis' if scripts & {'book4_editor.py','continue_book4_review.py'} else
                     'books/field-simulation' if scripts & {'book3_editor.py','continue_book3_review.py'} else 'books/modern-graphics')
            target=args[args.index('--output')+1] if '--output' in args else default
            if (cwd/target).resolve()==root:found.append(int(file.parent.name))
        except (OSError,UnicodeError,ValueError,IndexError):continue
    return found


def stamp(now):return datetime.fromtimestamp(now).astimezone().isoformat(timespec='seconds')


def summarize(state):
    return {'status':state.get('status'),'updated':state.get('updated'),
            'generation':state.get('repair_generation',0),'calls':state.get('calls',0),
            'chapters':{n:{'status':v.get('status'),'sha256':v.get('sha256')} for n,v in state.get('chapters',{}).items()},
            'parts':{n:{'approved':v.get('approved'),'fingerprint':v.get('fingerprint')} for n,v in state.get('part_reviews',{}).items()},
            'attention_count':len(state.get('attention',[]))}


def collect(root,*,now=None,pids=None,base_route='/book2',label='Volume II'):
    root=Path(root).resolve();now=time.time() if now is None else now
    state=json.loads((root/'status.json').read_text())
    current=summarize(state)
    pids=worker_pids(root) if pids is None else pids
    alive=bool(pids)
    with sqlite3.connect(root/'notifications.sqlite',timeout=10) as db:
        db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT UNIQUE NOT NULL, observed_at TEXT NOT NULL, kind TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL, details TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        db.commit();db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT value FROM checkpoint WHERE id=1').fetchone()
        previous=json.loads(row[0]) if row else None
        old=previous['state'] if previous else {}
        seen=dict(previous.get('errors',{})) if previous else {}

        def add(kind,message,level='info',details=None,key=None):
            details=details or {}
            key=key or hashlib.sha256(json.dumps([kind,message,details,current['updated']],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            db.execute('INSERT OR IGNORE INTO events(event_key,observed_at,kind,level,message,details) VALUES(?,?,?,?,?,?)',
                       (key,stamp(now),kind,level,message,json.dumps(details,ensure_ascii=False)))

        done=sum(v['status']=='model_reviewed' for v in current['chapters'].values())
        parts=sum(v['approved'] is True for v in current['parts'].values())
        if previous is None:
            add('monitor_started',f'網頁通知已啟用。目前逐章通過 {done}/30、跨章通過 {parts}/5。啟用前的逐項事件不回填。',key='monitor-started')
        else:
            for n,entry in current['chapters'].items():
                prior=old.get('chapters',{}).get(n,{})
                if entry==prior:continue
                if entry['status']=='model_reviewed':
                    add('chapter_approved',f'第 {int(n):02d} 章逐章模型審稿通過（非人工審定）。',details={'chapter':int(n),'sha256':entry['sha256']})
                elif prior.get('status')=='model_reviewed':
                    add('chapter_reopened',f'第 {int(n):02d} 章修訂後需再審查。','warning',{'chapter':int(n)})
                elif entry['status']=='error' and prior.get('status')!='error':
                    add('chapter_error',f'第 {int(n):02d} 章階段失敗，已留待重試或主編處理；不代表全卷停止。','warning',{'chapter':int(n)})
                elif entry['status']=='draft_ready':
                    add('draft_ready',f'第 {int(n):02d} 章初稿完成，等待審稿。',details={'chapter':int(n)})
            for n,entry in current['parts'].items():
                prior=old.get('parts',{}).get(n,{})
                if entry==prior:continue
                if entry['approved']:
                    add('part_approved',f'第 {int(n)+1} 部跨章模型審查通過。',details={'part':int(n)+1,'fingerprint':entry['fingerprint']})
                elif prior.get('approved'):
                    add('part_reopened',f'第 {int(n)+1} 部內容變更或複核退修，需要再審。','warning',{'part':int(n)+1})
            if current['generation']!=old.get('generation'):
                add('repair_started',f'進入修補世代 {current["generation"]}；續跑未完成工作。')
            if current['status']!=old.get('status'):
                if current['status']=='cross_chapter_review':add('cross_review_started','進入五部跨章審查階段。')
                elif current['status']=='needs_editorial_attention':
                    add('editorial_attention','本輪仍有編輯待辦；'+('續跑程序仍在執行。' if alive else '目前未偵測到編輯／續跑程序。'),'warning')
                elif current['status']=='interrupted':add('interrupted','工作流程記錄異常中斷，請檢查工作日誌。','error')

        if current['status']=='completed_model_review_pending_human' and old.get('status')!=current['status']:
            add('book_completed','全卷逐章及跨章模型審查已完成；仍待人工覆核與出版檢查。',details={'source_updated':current['updated'],'baseline_snapshot':previous is None})

        # 以錯誤檔案偵測timeout，不以active持續時間臆測。原始例外可能含提示／參數，不公開。
        for file in sorted((root/'editorial').glob('error-*.json')):
            try:
                stat=file.stat();version=f'{stat.st_mtime_ns}:{stat.st_size}'
                if seen.get(file.name)==version:continue
                record=json.loads(file.read_text())
                if not isinstance(record,dict):continue
            except (OSError,ValueError):continue
            seen[file.name]=version
            if previous is None:continue
            match=re.fullmatch(r'error-(.+)-(\d+|health)-(\d+)\.json',file.name)
            if not match:continue
            phase,call,attempt=match.groups();attempt=int(attempt)+1
            timeout=record.get('type') in ('TimeoutExpired','TimeoutError','ReadTimeout','ConnectTimeout') or 'timed out' in str(record.get('message','')).lower()
            details={'phase':phase,'call':int(call) if call.isdigit() else None,'attempt':attempt,'source_file':file.name}
            if timeout:
                message=f'{phase}：第 {attempt} 次嘗試逾時。'+('流程可再重試1次；目前是否仍工作請看上方狀態。' if attempt==1 else '此階段重試已耗盡，交由續跑／主編處理；不代表全卷停止。')
            else:message=f'{phase}：第 {attempt} 次嘗試失敗，等待重試或主編處理。'
            add('timeout' if timeout else 'attempt_failed',message,'warning',details,key='error:'+file.name+':'+version)

        terminal=current['status']=='completed_model_review_pending_human'
        missing_since=None if alive or terminal else (previous or {}).get('missing_since')
        if missing_since is None and not alive and not terminal:missing_since=now
        reported=(previous or {}).get('stop_reported',False) if not alive else False
        if not alive and not terminal and missing_since is not None and now-missing_since>=15 and not reported:
            add('worker_stopped','已連續15秒未偵測到本卷編輯／續跑程序，工作未完成；通知監測仍在運作。','error',{'status':current['status']},key=f'stopped:{missing_since}')
            reported=True
        if previous and not previous.get('alive') and alive:add('worker_resumed','已偵測到編輯／續跑程序恢復執行。',key=f'resumed:{now}')
        publication=(previous or {}).get('publication')
        try:
            manifest=json.loads((root/'published/publication.json').read_text())
            validation=json.loads((root/'published/validation.json').read_text())
            publication_key=manifest['html_sha256']+':'+manifest['pdf_sha256']
            if publication_key!=publication:
                from book2_publication import fresh_publication
                if fresh_publication(root,int(base_route[-1])):
                    add('publication_ready',f'HTML／PDF出版版已建立並通過排版檢查，PDF共 {int(validation["pages"])} 頁；閱讀連結已加入主頁。',details={'html':base_route+'/html','pdf':base_route+'/pdf'},key='publication:'+publication_key)
                    publication=publication_key
        except (OSError,ValueError,KeyError,TypeError):pass
        checkpoint={'publication':publication,'state':current,'errors':seen,'alive':alive,'pids':pids,'missing_since':missing_since,'stop_reported':reported,'checked_at':stamp(now)}
        db.execute('INSERT OR REPLACE INTO checkpoint(id,value) VALUES(1,?)',(json.dumps(checkpoint,ensure_ascii=False),))
        db.commit()
        rows=db.execute('SELECT id,observed_at,kind,level,message,details FROM events ORDER BY id DESC LIMIT 200').fetchall()
        events=[dict(zip(('id','observed_at','kind','level','message','details'),r)) for r in rows]
        for event in events:event['details']=json.loads(event['details'])
    result={'label':label,'base_route':base_route,'checked_at':stamp(now),'worker_alive':alive,'worker_pids':pids,'status':current['status'],'events':events}
    atomic(root/'notifications.json',json.dumps(result,ensure_ascii=False,indent=2))
    render(root,result)
    return result


def render(root,data):
    label='編輯／續跑程序執行中' if data['worker_alive'] else ('全卷模型審查完成，編輯程序已結束' if data['status']=='completed_model_review_pending_human' else '未偵測到編輯／續跑程序')
    rows=''
    for e in data['events']:
        detail=e.get('details',{})
        note=('啟用通知時已完成；原進度更新：'+str(detail.get('source_updated','未知'))) if detail.get('baseline_snapshot') else ''
        rows+=f'<li class="{html.escape(e["level"],quote=True)}"><time>{html.escape(e["observed_at"])}</time><br>{html.escape(e["message"])}'+(f'<br><small>{html.escape(note)}</small>' if note else '')+'</li>'
    page=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>{html.escape(data.get('label','Volume II'))} 進度通知</title>
<style>body{{font:15px sans-serif;line-height:1.7;margin:12px}}li{{padding:9px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}}ul{{list-style:none;padding:0}}time,small{{color:#555}}.warning{{border-left:4px solid #bb8500}}.error{{border-left:4px solid #bb2222}}.status{{background:#eaf2fa;padding:10px}}</style></head><body>
<div class="status">{html.escape(label)}<br>最近監測：{html.escape(data['checked_at'])}</div>
<small>時間為通知觀測時間；每5秒刷新。最近200則，完整紀錄保存在notifications.sqlite。關閉頁面仍記錄，但不會推播；啟用前事件不逐項回填。若最近監測時間停止更新，不能據此認定目前程序仍存活。</small><ul>{rows}</ul></body></html>'''
    atomic(Path(root)/'notifications.html',page)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='books/modern-graphics')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--base-route',choices=('/book2','/book3','/book4','/book5'),default='/book2')
    parser.add_argument('--label',default='Volume II')
    parser.add_argument('--interval',type=float,default=5)
    args=parser.parse_args();root=Path(args.output).resolve()
    if args.interval<1:parser.error('interval須至少1秒')
    with (root/'.notifications.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(f'通知監測PID {os.getpid()}；唯讀觀察 {root}',flush=True)
        while True:
            try:collect(root,base_route=args.base_route,label=args.label)
            except Exception as exc:
                print(f'通知監測本輪失敗：{type(exc).__name__}；不影響模型工作',flush=True)
                if args.once:raise
            if args.once:break
            time.sleep(args.interval)
