"""Volume II持久化工作流：40角色、30章、自動跨章定點修訂、有限預算。"""
import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import html
import json
import re
import shutil
import subprocess
from functools import lru_cache
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import llm_agent
import book2_spec as S
import book2_assets as A
from book_editor import atomic, prose_count, wait_remote_idle, active_discussions
from taiwan_discussion import usage


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=128)
def markdown_structure(text):
    parser=Path(__file__).resolve().parent/'publishing/parse-markdown.mjs'
    result=subprocess.run(['node',str(parser)],input=text,text=True,capture_output=True,check=True,timeout=20)
    return json.loads(result.stdout)


def normalize_structure(text,number):
    """只改已明示正確章號的標題、已知小節編號；不動正文或程式。"""
    lines=text.splitlines(keepends=True)
    digits='零一二三四五六七八九'
    chinese={(digits[n] if n<10 else ('十' if n<20 else digits[n//10]+'十')+(digits[n%10] if n%10 else '')):n for n in range(1,31)}
    for heading in markdown_structure(text)['headings']:
        if heading['map'][1]-heading['map'][0]!=1:continue
        title=heading['text'].strip()
        if heading['level']==1:
            match=re.match(r'^(?:第\s*)?([0-9]+|[一二三四五六七八九十]+)\s*(?:章|[.．、:：])?\s*(.*)$',title)
            if not match:continue
            value=int(match[1]) if match[1].isdigit() else chinese.get(match[1])
            if value!=number:continue
            title=f'第{number:02d}章 '+match[2].lstrip('：:．.、 ').strip()
            lines[heading['map'][0]]='# '+title+'\n'
        elif heading['level']==2:
            bare=re.sub(r'^\d+[.．、:：]\s*','',title)
            if bare in S.SECTIONS:lines[heading['map'][0]]='## '+bare+'\n'
    return ''.join(lines)


def validate(text,number,final=True):
    issues=[]
    if not re.search(r'^#\s+第\s*0?'+str(number)+r'\s*章',text):issues.append('章標題不符')
    for section in S.SECTIONS:
        if not re.search(r'^##\s+'+re.escape(section)+r'\s*$',text,re.M):issues.append('缺小節：'+section)
    if prose_count(text)<(S.MIN_CHARACTERS if final else 1200):issues.append('正文不足，需补足推導及實作說明')
    if text.count('```')%2:issues.append('程式圍欄未配對')
    blocks=[f['content'] for f in markdown_structure(text)['fences'] if f['language'] in ('python','python3')]
    if not blocks:issues.append('缺少Python實作')
    for code in blocks:
        try:ast.parse(code)
        except SyntaxError as exc:issues.append('Python語法：'+str(exc))
    if re.search(r'<(?:script|iframe|img|svg|object|style)\b',text,re.I):issues.append('禁止HTML或外部內容')
    for image in re.findall(r'!\[[^]]*\]\(([^)]+)\)',text):
        if image not in ['../figures/'+name for name in A.FIGURES]:issues.append('未知圖片：'+image)
    return issues


def checked_patches(chapters,result):
    """先驗證所有非重疊原文，再一次產生新稿，不對磁碟做部分修改。"""
    patches=result.get('patches') if isinstance(result,dict) else None
    if not isinstance(patches,list) or not 1<=len(patches)<=20:raise ValueError('需1～20個精準patch')
    edits={}
    for patch in patches:
        n=str(patch['chapter']);old=patch['old'];new=patch['new']
        if n not in chapters or not isinstance(old,str) or not isinstance(new,str) or not old or len(old)>5000:raise ValueError('patch範圍不合法')
        if chapters[n].count(old)!=1:raise ValueError('patch原文非唯一匹配')
        start=chapters[n].index(old);end=start+len(old)
        if any(start<e and end>s for s,e,_ in edits.get(n,[])):raise ValueError('patch重疊')
        edits.setdefault(n,[]).append((start,end,new))
    result={}
    for n,items in edits.items():
        text=chapters[n]
        for start,end,new in sorted(items,reverse=True):text=text[:start]+new+text[end:]
        issues=validate(text,int(n))
        if issues:raise ValueError('patch後結構無效：'+'；'.join(issues))
        result[n]=text
    return result


def build(root,state):
    base=getattr(S,'BASE_ROUTE','/book2')
    other=' ｜ <a href="/book2">Volume II</a>' if base!='/book2' else ''
    if base in ('/book4','/book5'):other+=' ｜ <a href="/book3">Volume III</a>'
    if base=='/book5':other+=' ｜ <a href="/book4">Volume IV</a>'
    counts={};pieces=['# '+S.TITLE,'> 狀態：'+state['status']+'。模型稿件不是人工審定教材。', (root/'00-preface.md').read_text()];cards=[]
    for n,(title,_,_,_) in enumerate(S.CHAPTERS,1):
        file=root/'chapters'/f'{n:02d}.md';text=file.read_text() if file.exists() else ''
        counts[str(n)]=prose_count(text)
        pieces.append(text.replace('../figures/','figures/') if text else f'# 第{n:02d}章 {title}\n\n> 尚未產生，不是完成章節。')
        entry=state['chapters'].get(str(n),{})
        cards.append(f'<li>第{n:02d}章 {html.escape(title)} — {entry.get("status","pending")}；{counts[str(n)]}字'+(f' <a href="{base}/chapter/{n:02d}">章稿</a>' if text else '')+'</li>')
    pieces.extend([(root/'99-appendix.md').read_text(),(root/'REFERENCES.md').read_text()])
    total=sum(counts.values())+prose_count(pieces[2])+prose_count(pieces[-2])
    state['word_count']={'total':total,'chapters':counts,'maximum':S.MAX_CHARACTERS,'method':'中文正文；排除公式、程式、標點、英文及參考來源'}
    state['updated']=datetime.now().isoformat()
    atomic(root/'book.md','\n\n'.join(pieces));atomic(root/'status.json',json.dumps(state,ensure_ascii=False,indent=2))
    atomic(root/'word_count.json',json.dumps(state['word_count'],ensure_ascii=False,indent=2))
    done=sum(v.get('status')=='model_reviewed' for v in state['chapters'].values())
    messages=''.join('<li>'+html.escape(x)+'</li>' for x in state.get('attention',[]))
    atomic(root/'index.html',f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>{html.escape(S.TITLE)}</title>
<style>body{{font:17px sans-serif;line-height:1.9;max-width:1050px;margin:30px auto;padding:18px}}li{{margin:8px}}img{{max-width:100%}}.status{{padding:18px;background:#eaf2fa}}</style>
<h1>{html.escape(S.TITLE)}</h1><p><a href="/book">Volume I</a>{other} ｜ <a href="{base}/toc">{getattr(S,'LABEL','Volume II')}目錄</a> ｜ <a href="{base}.md">目前整卷Markdown</a></p>
<div class="status">狀態：{html.escape(state['status'])}<br>逐章審稿 {done}/30；正文 {total:,}/200,000字；模型呼叫 {state.get('calls',0)} 次；總上限：{state.get('call_limit',240) or '無（使用者授權）'}<br>更新：{state['updated']}<br>40個獨立角色、原五模型；目前執行中呼叫：{len(state.get('active',{}))}</div>
<p>初步目標約13.5萬字，重視推導、程式、測試與除錯。模型審稿、程式已執行與人工驗證分開記錄。</p><ul>{messages}</ul><img src="{base}/figures/pipeline.svg" alt="圖學主軸"><ol>{''.join(cards)}</ol></html>''')


class Editor:
    def __init__(self,root,state,args):
        self.root=root;self.state=state;self.args=args;self.mutex=threading.RLock()
        self.providers={p:threading.Semaphore(2 if p=='openai-codex' else 1) for p,_ in S.MODELS}
        self.people={p['id']:threading.Lock() for p in S.roster()}

    def invoke(self,person,phase,context,rules,mode='markdown'):
        if rules==S.AUTHOR_RULES:
            context={**context,'heading':f'# 第{context["number"]:02d}章 {S.CHAPTERS[context["number"]-1][0]}'}
        session_id=person['id']
        if getattr(S,'TASK_SCOPED_SESSIONS',False):
            match=re.match(r'^(ch\d+|part\d+)(?:-|$)',phase)
            scope=(f'ch{int(context["number"]):02d}' if 'number' in context else
                   match[1] if match else 'preflight' if phase.startswith('preflight-') else 'task-'+digest(phase)[:12])
            session_id+='--'+scope
            context={**context,'task_session_id':session_id}
        if self.state.get('repair_generation'):
            context={**context,'repair_generation':self.state['repair_generation']}
        key=digest({'person':person,'context':context,'rules':rules,'mode':mode,'validation_version':2})
        cache=self.root/'editorial'/f'{phase}-{key[:16]}.json'
        if cache.exists():
            stored=json.loads(cache.read_text())
            if stored['key']==key:return stored['result']
        for attempt in range(2):
            call=None
            try:
                with self.people[person['id']],self.providers[person['provider']]:
                    if person['provider']=='gb10-2-vllm':wait_remote_idle()
                    with self.mutex:
                        if self.args.max_calls>0 and self.state['calls']>=self.args.max_calls:raise RuntimeError('模型呼叫上限已達，需主編檢查')
                        self.state['calls']+=1;call=self.state['calls']
                        self.state['active'][str(call)]={'person':person['id'],'phase':phase,'started':datetime.now().isoformat()};build(self.root,self.state)
                    workspace=self.root/'agents'/session_id;workspace.mkdir(parents=True,exist_ok=True)
                    prompt={**context,'round':call,'attempt':attempt+1}
                    atomic(workspace/f'input-{call:06d}.json',json.dumps(prompt,ensure_ascii=False,indent=2))
                    options=SimpleNamespace(pi_bin='pi',provider=person['provider'],model=person['model'],tools='',output_format=mode,replace_system_prompt=True,thinking='off' if person['provider']=='gb10-2-vllm' else 'low',agent_timeout=self.args.timeout)
                    result=llm_agent.decide(SimpleNamespace(id=session_id,workspace=workspace),prompt,options,instructions=rules)
                    if rules==S.REVIEW_RULES:
                        if not isinstance(result,str) or not result.strip().endswith(('VERDICT: APPROVE','VERDICT: REVISE')):
                            atomic(self.root/'editorial'/f'rejected-review-{call:06d}.txt',str(result))
                            context={**context,'review_format_error':'前次缺少有效結論；只輸出審稿，不重寫章稿。最後一行必須精確為 VERDICT: APPROVE 或 VERDICT: REVISE。','previous_response_tail':str(result)[-1500:]}
                            raise ValueError('缺少有效審稿結論')
                    elif rules==S.AUTHOR_RULES:
                        if isinstance(result,str):
                            normalized=normalize_structure(result,context['number'])
                            if normalized!=result:
                                atomic(self.root/'editorial'/'originals'/f'{call:06d}.md',result)
                                result=normalized
                        issues=validate(result,context['number'],False) if isinstance(result,str) else ['不是章稿']
                        if issues:
                            atomic(self.root/'editorial'/f'rejected-{call:06d}.txt',str(result))
                            context={**context,'previous_output':result,'output_errors':issues,'task':'修復指定輸出問題並保留正確內容，不盲目重寫。'}
                            raise ValueError('；'.join(issues))
                    atomic(cache,json.dumps({'key':key,'person':person,'call':call,'result':result},ensure_ascii=False,indent=2))
                    if isinstance(result,str):atomic(cache.with_suffix('.md'),result)
                    return result
            except Exception as exc:
                atomic(self.root/'editorial'/f'error-{phase}-{call or "health"}-{attempt}.json',json.dumps({'type':type(exc).__name__,'message':str(exc)[:1500]},ensure_ascii=False))
                if attempt==1:raise
            finally:
                if call is not None:
                    with self.mutex:self.state['active'].pop(str(call),None);build(self.root,self.state)

    def context(self,n):
        title,part,maths,lab=S.CHAPTERS[n-1]
        return {'number':n,'title':title,'part':S.PARTS[part],'outline':S.CHAPTERS,'conventions':S.CONVENTIONS,'core':maths,'lab':lab,'sections':S.SECTIONS,'sources':S.SOURCES,'source_notes':S.SOURCE_NOTES,'available_images':['../figures/'+name for name in A.FIGURES],'target_characters':S.TARGET_PER_CHAPTER}

    def save_chapter(self,n,text,status,review):
        with self.mutex:
            target=self.root/'chapters'/f'{n:02d}.md'
            if target.exists() and target.read_text()!=text:
                atomic(self.root/'editorial'/'history'/f'{n:02d}-{digest(target.read_text())[:16]}.md',target.read_text())
            atomic(target,text)
            self.state['chapters'][str(n)]={'status':status,'count':prose_count(text),'sha256':digest(text),'review':review,'automatic_issues':validate(text,n),'author':S.chapter_people(n)[0],'reviewer':S.chapter_people(n)[1]}
            build(self.root,self.state)

    def chapter(self,n,review_only=False):
        target=self.root/'chapters'/f'{n:02d}.md';entry=self.state['chapters'].get(str(n),{})
        if target.exists() and entry.get('status')=='model_reviewed' and entry.get('sha256')==digest(target.read_text()):return
        author,reviewer=S.chapter_people(n);ctx=self.context(n)
        text=target.read_text() if target.exists() else self.invoke(author,f'ch{n:02d}-draft',ctx,S.AUTHOR_RULES)
        if not target.exists():self.save_chapter(n,text,'draft_ready','初稿已產生，等待獨立審稿')
        for turn in range(1 if review_only else 3):
            checks=validate(text,n)
            review=self.invoke(reviewer,f'ch{n:02d}-review',{**ctx,'draft':text,'measured_characters':prose_count(text),'automatic_checks':checks},S.REVIEW_RULES)
            approved=review.strip().endswith('VERDICT: APPROVE') and not checks
            self.save_chapter(n,text,'model_reviewed' if approved else 'needs_revision',review)
            if approved:return
            if turn<(0 if review_only else 2):
                text=self.invoke(author,f'ch{n:02d}-revision',{**ctx,'draft':text,'review':review,'automatic_checks':checks,'task':'依具體問題修訂，保留正確內容，輸出完整章稿。'},S.AUTHOR_RULES)

    def targeted_chapter(self,n):
        """沿用已有意見，最多兩輪第三模型定點修補；不整章再生成。"""
        if not (self.root/'chapters'/f'{n:02d}.md').exists():
            self.chapter(n)
            return
        author,reviewer=S.chapter_people(n)
        # 原作者與審稿者不變；第三模型主編由既有40角色輪派。
        leads=[p for p in S.roster() if p['role']=='author' and p['provider']=='openai-codex' and p['model'] not in (author['model'],reviewer['model'])]
        if not leads:leads=[p for p in S.roster() if p['role']=='author' and p['model'] not in (author['model'],reviewer['model'])]
        for cycle in range(3):
            text=(self.root/'chapters'/f'{n:02d}.md').read_text()
            entry=self.state['chapters'].get(str(n),{})
            matching=entry.get('sha256')==digest(text)
            if matching and entry.get('status')=='model_reviewed':return
            # 手動改稿、逾時或新patch先複審；未變拒稿可直接沿用現有意見。
            if not (cycle==0 and matching and entry.get('status')=='needs_revision' and entry.get('review')):
                self.chapter(n,review_only=True)
                entry=self.state['chapters'][str(n)]
                if entry.get('status')=='model_reviewed':return
            if cycle==2:return
            chapters={str(n):text}
            lead=leads[(n+cycle)%len(leads)]
            context={**self.context(n),'chapters':chapters,'review':entry['review'],
                'measured_characters':prose_count(text),'minimum_characters':S.MIN_CHARACTERS,
                'automatic_checks':validate(text,n),
                'task':'依具體證據精準修補，不整章重寫。同一函式或段落的鄰近修改合併成單一patch，每個old相對同一原稿，不得互相包含。若正文不足，於相關教學小節補完整推導、測試與除錯解釋，確保修後中文正文至少3000字；程式、公式與英文不計字數。'}
            revised=self.patch(lead,f'ch{n:02d}-targeted-patch',context,chapters)
            for number,new in revised.items():self.save_chapter(int(number),new,'needs_review','第三模型主編定點修補後待原審稿者複審')

    def run_chapters(self,numbers,review_only=False,targeted=False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs={(pool.submit(self.targeted_chapter,n) if targeted else pool.submit(self.chapter,n,review_only)):n for n in numbers}
            for future in concurrent.futures.as_completed(jobs):
                n=jobs[future]
                try:future.result()
                except Exception as exc:
                    with self.mutex:self.state['chapters'].setdefault(str(n),{}).update(status='error',error=str(exc)[:1500]);build(self.root,self.state)
                print(f'{getattr(S,"LABEL","Volume II")} 第{n:02d}章：{self.state["chapters"].get(str(n),{}).get("status")}',flush=True)

    def patch(self,person,phase,context,chapters):
        for attempt in range(2):
            proposal=self.invoke(person,phase,context,S.PATCH_RULES,'json')
            try:return checked_patches(chapters,proposal)
            except (ValueError,KeyError,TypeError) as exc:
                if attempt==1:raise
                context={**context,'previous_invalid_proposal':proposal,'patch_error':str(exc),'task':'重新對照原稿給唯一、不重疊且保留完整結構的patch。'}

    def cross_review(self,part):
        numbers=[n for n,c in enumerate(S.CHAPTERS,1) if c[1]==part]
        reviewer=next(p for p in S.roster() if p['role']=='reviewer' and p['model']==S.MODELS[part][1])
        editor=next(p for p in S.roster() if p['role']=='author' and p['model']==S.MODELS[(part+1)%5][1])
        for cycle in range(3):
            chapters={str(n):(self.root/'chapters'/f'{n:02d}.md').read_text() for n in numbers}
            ctx={'part':S.PARTS[part],'conventions':S.CONVENTIONS,'sources':S.SOURCES,'chapters':chapters,'chapter_counts':{n:prose_count(t) for n,t in chapters.items()},'review_contract':digest({'rules':S.REVIEW_RULES,'reviewer':reviewer}),'task':'檢查跨章座標、符號、接口、依賴與推導；給可定位的實質問題。'}
            previous=self.state.get('part_reviews',{}).get(str(part),{})
            if (previous.get('approved') and previous.get('fingerprint')==digest(ctx)
                    and previous.get('review','').strip().endswith('VERDICT: APPROVE')
                    and all(self.state['chapters'][n].get('status')=='model_reviewed'
                            and self.state['chapters'][n].get('sha256')==digest(t) for n,t in chapters.items())):
                return
            review=self.invoke(reviewer,f'part{part+1}-review',ctx,S.REVIEW_RULES)
            approved=review.strip().endswith('VERDICT: APPROVE')
            self.state['part_reviews'][str(part)]={'approved':approved,'fingerprint':digest(ctx),'review':review};build(self.root,self.state)
            if approved:return
            if cycle==2:return
            revised=self.patch(editor,f'part{part+1}-patch',{**ctx,'review':review},chapters)
            for n,text in revised.items():self.save_chapter(int(n),text,'needs_review','跨章主編定點修訂，需重新複審')
            self.run_chapters([int(n) for n in revised],review_only=True)
            if any(self.state['chapters'][str(n)]['status']!='model_reviewed' for n in numbers):return


def run(args):
    if args.max_calls<0:raise ValueError('max-calls須為非負數；0表示使用者授權不設總呼叫上限')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg={'title':S.TITLE,'people':S.roster(),'chapters':S.CHAPTERS,'conventions':S.CONVENTIONS,'max_characters':S.MAX_CHARACTERS,'target_per_chapter':S.TARGET_PER_CHAPTER}
        config=root/'config.json'
        if config.exists() and json.loads(config.read_text())!=json.loads(json.dumps(cfg)):raise ValueError('章節／模型契約變更，拒絕混入舊工作')
        A.initialize(root)
        atomic(config,json.dumps(cfg,ensure_ascii=False,indent=2))
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'status':'initialized','chapters':{},'calls':0,'part_reviews':{},'active':{}}
        if args.init_only:build(root,state);return
        if active_discussions():raise RuntimeError('討論正在使用模型，請待釋放資源再啟動')
        if getattr(args,'targeted_repair',False):
            generation=state.get('repair_generation',0)+1
            backup=root/'backups'/f'targeted-generation-{generation:03d}'
            backup.mkdir(parents=True)
            shutil.copytree(root/'chapters',backup/'chapters')
            shutil.copy2(root/'status.json',backup/'status.json')
            state['repair_generation']=generation
        state.update(status='writing',active={},attention=[],call_limit=args.max_calls or None);build(root,state)
        editor=Editor(root,state,args)
        try:
            editor.run_chapters(range(1,len(S.CHAPTERS)+1),targeted=getattr(args,'targeted_repair',False))
            # 定點模式已含兩輪主編修補；一般模式才執行此額外升級。
            for n in (() if getattr(args,'targeted_repair',False) else range(1,len(S.CHAPTERS)+1)):
                entry=state['chapters'].get(str(n),{})
                if entry.get('status')!='needs_revision':continue
                lead=next(p for p in S.roster() if p['role']=='author' and p['model'] not in (entry['author']['model'],entry['reviewer']['model']))
                text=(root/'chapters'/f'{n:02d}.md').read_text()
                try:
                    revised=editor.patch(lead,f'ch{n:02d}-chief-patch',{'chapters':{str(n):text},'review':entry['review'],'conventions':S.CONVENTIONS},{str(n):text})
                    for number,new in revised.items():editor.save_chapter(int(number),new,'needs_review','主編定點修補後待審')
                    editor.run_chapters([n],review_only=True)
                except Exception as exc:state['attention'].append(f'第{n:02d}章主編修訂：{str(exc)[:500]}')
            if all(state['chapters'].get(str(n),{}).get('status')=='model_reviewed' for n in range(1,len(S.CHAPTERS)+1)):
                state['status']='cross_chapter_review';state.setdefault('part_reviews',{});build(root,state)
                for part in range(len(S.PARTS)):
                    try:editor.cross_review(part)
                    except Exception as exc:state['attention'].append(f'第{part+1}部主編／跨章階段：{type(exc).__name__}: {str(exc)[:500]}')
            complete=(len(state['chapters'])==len(S.CHAPTERS) and all(v.get('status')=='model_reviewed' for v in state['chapters'].values()) and len(state['part_reviews'])==len(S.PARTS) and all(v['approved'] for v in state['part_reviews'].values()))
            state['status']='completed_model_review_pending_human' if complete and state['word_count']['total']<=S.MAX_CHARACTERS else 'needs_editorial_attention'
            if state['status']=='needs_editorial_attention':state['attention'].append('有限修訂仍有待辦，詳見章節／跨章意見；不是等待使用者做一般編輯決定，不冒充完成。')
        except Exception as exc:
            state['status']='interrupted';state['attention'].append(f'{type(exc).__name__}: {str(exc)[:1000]}')
            raise
        finally:
            state['active']={};build(root,state)
            atomic(root/'usage.json',json.dumps(usage(root),ensure_ascii=False,indent=2))
            atomic(root/'RUN_SUMMARY.md','# '+getattr(S,'LABEL','Volume II')+'本次工作\n\n狀態：'+state['status']+'\n\n'+'\n'.join('- '+x for x in state['attention'])+'\n\n模型稿件與審稿不等於人工或程式驗證。')
            print(getattr(S,'LABEL','Volume II')+'狀態：'+state['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=S.ROOT)
    parser.add_argument('--init-only',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-calls',type=int,default=240,help='累計呼叫上限；0表示不設總上限，仍保留逾時及有限重試')
    parser.add_argument('--targeted-repair',action='store_true',help='新修補世代；只對未通過章節做第三模型精準patch與原審稿者複審')
    run(parser.parse_args())
