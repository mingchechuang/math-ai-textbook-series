"""40位編輯協作產生20章Markdown教材；可恢復、有限重試，不執行生成程式。"""
import argparse
import collections
import concurrent.futures
import fcntl
import html
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import llm_agent
from book_assets import initialize
from book_spec import TITLE, PARTS, CHAPTERS, MODELS, SOURCES, CONVENTIONS, SECTIONS, AUTHOR_RULES, REVIEW_RULES
from taiwan_discussion import usage


def atomic(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text,encoding='utf-8');temp.replace(path)


def prose_count(text):
    text=re.split(r'^##?\s+參考來源',text,flags=re.M)[0]
    text=re.sub(r'```.*?```','',text,flags=re.S)
    text=re.sub(r'\$\$.*?\$\$','',text,flags=re.S)
    text=re.sub(r'\$[^\n$]*\$','',text)
    text=re.sub(r'!\[[^\]]*\]\([^)]*\)','',text)
    return len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]',text))


def validate_chapter(text, number, final=False):
    issues=[]
    if not re.search(r'^#\s+第\s*0?'+str(number)+r'\s*章',text):issues.append('缺少正確章標題')
    for section in SECTIONS:
        if not re.search(r'^##\s+'+re.escape(section)+r'\s*$',text,re.M):issues.append('缺少小節：'+section)
    if '```python' not in text:issues.append('缺少Python範例')
    if text.count('```')%2:issues.append('程式圍欄未成對')
    if re.search(r'<(?:script|iframe|img|svg|object|style)\b',text,re.I):issues.append('含未許可HTML')
    for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
        if image not in ['../figures/'+x for x in ('roadmap.svg','shapes.svg','attention.svg','fusion.svg','agent-safety.svg')]:issues.append('未知圖片：'+image)
    count=prose_count(text)
    if final and count<2200:issues.append(f'正文中文字數{count}，至少2200；上限僅供參考')
    if not final and count<700:issues.append('內容過短，非完整初稿')
    return issues


def roster():
    people=[]
    for i in range(40):
        provider,model=MODELS[i%5]
        people.append({'id':f'editor_{i:02d}','provider':provider,'model':model,
                       'chapter':i//2+1,'role':'作者' if i%2==0 else '獨立審稿者'})
    return people


def active_discussions():
    result=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().decode().split('\0')
            if any(Path(a).name in ('taiwan_discussion.py','continue_taiwan_discussion.py') for a in args[:3]):
                result.append(int(path.parent.name))
        except (OSError,UnicodeError):pass
    return result


def wait_remote_idle(timeout=120):
    deadline=time.monotonic()+timeout
    while True:
        with urllib.request.urlopen('http://192.168.90.178:8000/health',timeout=8) as r:
            if r.status!=200:raise RuntimeError('GB10-2 API unhealthy')
        with urllib.request.urlopen('http://192.168.90.178:8000/metrics',timeout=8) as r:metrics=r.read().decode()
        values=[float(line.rsplit(' ',1)[-1]) for line in metrics.splitlines()
                if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
        if values and sum(values)==0:return
        if time.monotonic()>deadline:raise RuntimeError('GB10-2持續忙碌，暫停此章，未啟動新GPU工作')
        time.sleep(10)


def part_fingerprint(root, part):
    payload = {'conventions': CONVENTIONS, 'review_rules': REVIEW_RULES,
               'outline': CHAPTERS, 'models': MODELS, 'sources': SOURCES,
               'chapters': {n: (root/'chapters'/f'{n:02d}.md').read_text()
                            for n,c in enumerate(CHAPTERS,1) if c[1] == part}}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def reusable_part_review(root, evidence, fingerprint):
    if evidence.get('fingerprint') != fingerprint or not evidence.get('approved'):
        return False
    try:
        text = (root/evidence['review_file']).read_text()
        return (text.strip().endswith('VERDICT: APPROVE') and
                hashlib.sha256(text.encode()).hexdigest() == evidence.get('review_sha256'))
    except (OSError, KeyError):
        return False


def build(root, state):
    text=['# '+TITLE+'\n','> 編輯狀態：'+state['status']+'。模型稿件與模型審稿不是人工審定教材。\n',
          '![主軸](figures/roadmap.svg)\n','## 目錄\n','- [導讀](#preface)']
    for n,(title,part,_,_) in enumerate(CHAPTERS,1):text.append(f'- [第{n:02d}章 {title}](#ch{n:02d})')
    text+=['- [附錄](#appendix)','\n<a id="preface"></a>\n',(root/'00-preface.md').read_text()]
    counts={}; cards=[]
    for n,(title,part,_,_) in enumerate(CHAPTERS,1):
        file=root/'chapters'/f'{n:02d}.md';entry=state['chapters'].get(str(n),{})
        counts[str(n)]=prose_count(file.read_text()) if file.exists() else 0
        text+= [f'\n<a id="ch{n:02d}"></a>\n']
        if file.exists():text.append(file.read_text().replace('../figures/','figures/'))
        else:text.append(f'# 第{n:02d}章 {title}\n\n> 尚未完成，不是正式章節。\n')
        cards.append(f'<li>第{n:02d}章 {html.escape(title)} — {html.escape(entry.get("status","pending"))}；{counts[str(n)]}字'+
                     (f' <a href="/book/chapter/{n:02d}">Markdown</a>' if file.exists() else '')+'</li>')
    text+=['\n<a id="appendix"></a>\n',(root/'99-appendix.md').read_text(),(root/'REFERENCES.md').read_text()]
    front=prose_count((root/'00-preface.md').read_text())+prose_count((root/'99-appendix.md').read_text())
    state['word_count']={'method':'中文字元；不含標點、英文、公式、程式與章末參考來源','chapters':counts,
                         'front_and_appendix':front,'total':sum(counts.values())+front,'target_range':[45000,55000],
                         'upper_limit_enforced':False,'chapter_minimum':2200,
                         'length_policy':'依使用者指示允許超字；保留原建議篇幅，不以超過章節或全書上限判定失敗。'}
    atomic(root/'book.md','\n\n'.join(text))
    atomic(root/'word_count.json',json.dumps(state['word_count'],ensure_ascii=False,indent=2))
    atomic(root/'status.json',json.dumps(state,ensure_ascii=False,indent=2))
    publication_links = ''
    try:
        published = root/'published'
        publication = json.loads((published/'publication.json').read_text())
        fresh = publication.get('manuscript_sha256') == hashlib.sha256((root/'book.md').read_bytes()).hexdigest()
        if fresh and all((published/name).is_file() for name in ('volume-1.html','volume-1.pdf')):
            publication_links = '<p><b>圖文出版版：</b><a href="/book/html">HTML閱讀版（公式與圖片）</a> ｜ <a href="/book/pdf">PDF列印版</a> ｜ <a href="/book/download/html">下載HTML（離線閱讀）</a></p>'
    except (OSError, ValueError):
        pass
    page=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>{html.escape(TITLE)}</title>
    <style>body{{font:17px sans-serif;line-height:1.9;max-width:1100px;margin:25px auto;padding:16px}}li{{margin:8px}}img{{max-width:100%}}</style>
    <h1>{html.escape(TITLE)}</h1><p>40位編輯，5個模型各8位；20章，原創SVG，Markdown主稿。</p>
    <p><b>狀態：{html.escape(state['status'])}｜目前 {state['word_count']['total']:,} 中文字／目標45,000～55,000字</b></p>
    <p>計數排除公式與程式。不完整章節不冒充已完成；自動結構檢查及模型審稿不等於人工數學驗證。</p>
    <p>教材系列：<a href="/book2">Volume II 電腦圖學（製作進度）</a></p>
    {publication_links}
    <p><a href="/book.md">整本Markdown（目前進度）</a> ｜ <a href="/book/toc">詳細目錄</a></p>
    <img src="/book/figures/roadmap.svg" alt="教材主軸"><ol>{''.join(cards)}</ol></html>'''
    atomic(root/'index.html',page)


def init_book(root):
    if (root/'config.json').exists():return
    initialize(root)
    atomic(root/'config.json',json.dumps({'title':TITLE,'people':roster(),'chapters':CHAPTERS,'conventions':CONVENTIONS,
           'planned_main_calls':80,'part_consistency_reviews':5,'target_chinese_characters':50000},ensure_ascii=False,indent=2))
    atomic(root/'README.md',f'''# {TITLE}

- 主稿：`book.md`；分章：`chapters/01.md`～`20.md`。
- 目錄與章節契約：`TOC.md`；共用符號：`STYLE_GUIDE.md`。
- 原創SVG：`figures/`；合成資料：`data/`；基礎驗證程式：`examples/`。
- 40位模型編輯的session及事件：`agents/`；審稿與修訂：`editorial/`。
- 完成20章後做5部跨章一致性審查，仍需人工審定。
- 字數規則：正文中文字元，不含標點、英文、公式、程式及章末參考來源；目標45,000～55,000。
- 合成資料不能作真實養殖閾值；範例不連接現場設備。
- 執行：`python3 book_editor.py --output {root} --wait-discussion`；已存的成功階段不重跑。
- `python3 book_editor.py --output {root} --build-only`只重組既有稿件，不呼叫模型。
- 失敗可用相同命令恢復；每次重試保留之前事件與token紀錄，不自行換模型。
''')


def run(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        init_book(root)
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'status':'initialized','chapters':{}}
        if args.build_only:
            build(root,state);return
        people=roster()
        if json.loads((root/'config.json').read_text())['people']!=people:raise ValueError('模型配置已變更，拒絕混寫')
        while args.wait_discussion and active_discussions():
            state['status']='queued_waiting_for_discussion';state['waiting_for_pids']=active_discussions();build(root,state)
            print('教材排隊中，等待現有40人討論釋放模型服務',flush=True);time.sleep(30)
        state.pop('waiting_for_pids',None)
        repair_mode = getattr(args, 'repair', False)
        review_only = getattr(args, 'review_only', False)
        if review_only and not repair_mode:
            raise ValueError('--review-only必須搭配--repair，以新代次審查目前稿件')
        if repair_mode:
            generation = state.get('repair_generation', 0) + 1
            backup = root / 'backups' / f'repair-{generation:03d}'
            backup.mkdir(parents=True, exist_ok=False)
            for name in ('chapters','editorial'):
                shutil.copytree(root/name, backup/name)
            for name in ('status.json','book.md','word_count.json'):
                if (root/name).exists(): shutil.copy2(root/name, backup/name)
            state['repair_generation'] = generation
        else:
            generation = state.get('repair_generation', 0)
        state['status']='repairing' if repair_mode else 'writing';build(root,state)
        limits={p:threading.Semaphore(2 if p=='openai-codex' else 1) for p,_ in MODELS}
        def invoke(person,number,tag,context,rules):
            target=root/'editorial'/f'{number:02d}-{tag}.md'
            if target.exists():return target.read_text()
            workspace=root/'agents'/person['id'];workspace.mkdir(parents=True,exist_ok=True)
            (workspace/'pi_events').mkdir(exist_ok=True)
            # 每階段以數字保存，失敗嘗試移至attempts以免覆寫。
            if tag.startswith('repair-'):
                # 唯一回合編號：保留原session與旧事件，不把修訂當成最初稿。
                _, gen, step = tag.split('-')
                stage = 1000 + int(gen)*20 + int(step)
            else:
                stage={'draft':1,'review':2,'revision':3,'length-fix':4,'final-review':5,'quality-fix':6,'quality-review':7,'part-review':8}[tag]
            is_review = rules == REVIEW_RULES
            if is_review and isinstance(context.get('draft'), str):
                measured = prose_count(context['draft'])
                context = {**context, 'measured_chinese_characters': measured,
                           'length_requirement_passed': measured >= 2200,
                           'length_policy': '使用者已允許超字，原2600字上限不再是拒稿理由；歷史審稿與舊session若提及上限，均由本政策取代。',
                           'count_evidence': '以上為編輯程式對本次draft直接計數；automatic_checks=[]代表未檢出問題，不是沒檢查。',
                           'notation_ruling': '全書以R^d表示column vector，shape d×1；R^d與R^(d×1)在此約定相容，不僅因兩種記號並存就判錯。',
                           'review_boundary': '請以實際錯誤或必要內容缺漏判定。原創推導可自成證明，毋須把每個式子都強制掛外部來源；未執行程式需如實標示，不能反過來要求作者偽稱跑過。'}
            for attempt in range(2):
                options=SimpleNamespace(pi_bin='pi',provider=person['provider'],model=person['model'],tools='',
                                        output_format='markdown',replace_system_prompt=True,thinking='off' if person['provider']=='gb10-2-vllm' else 'low',agent_timeout=args.timeout)
                prompt={**context,'round':stage,'attempt':attempt+1}
                atomic(workspace/f'input-{number:02d}-{tag}-{attempt}.json',json.dumps(prompt,ensure_ascii=False,indent=2))
                logdir=workspace/'pi_events'
                for old in list(logdir.glob(f'round_{stage:06d}.*')):
                    archive=logdir/'attempts';archive.mkdir(exist_ok=True)
                    old.rename(archive/(datetime.now().strftime('%Y%m%d%H%M%S%f')+'_'+old.name))
                try:
                    with limits[person['provider']]:
                        if person['provider']=='gb10-2-vllm':wait_remote_idle()
                        result=llm_agent.decide(SimpleNamespace(id=person['id'],workspace=workspace),prompt,options,instructions=rules)
                    if not isinstance(result,str) or len(result)<100:raise ValueError('回覆過短')
                    output_issues = ([] if result.strip().endswith(('VERDICT: APPROVE','VERDICT: REVISE')) else ['缺少審稿結論']) if is_review else validate_chapter(result, number, final=False)
                    if output_issues:
                        rejected = root/'editorial'/'rejected'/f'{number:02d}-{tag}-{datetime.now().strftime("%Y%m%d%H%M%S%f")}.md'
                        atomic(rejected,result)
                        context = {**context, 'previous_invalid_output':result[:1000],
                                   'output_correction':'不要檢查檔案或說你打算做什麼；直接輸出完整繁體中文章稿或指定審稿內容。',
                                   'output_errors':output_issues}
                        raise ValueError('無效內容：'+'；'.join(output_issues))
                    atomic(target,result);return result
                except (ValueError,OSError,TypeError,TimeoutError,subprocess.TimeoutExpired) as exc:
                    atomic(root/'editorial'/f'{number:02d}-{tag}-error.json',json.dumps({'type':type(exc).__name__,'error':str(exc)[:1000]},ensure_ascii=False))
                    if attempt==1:raise
            raise RuntimeError('階段未完成')
        def chapter(number):
            title,part,maths,lab=CHAPTERS[number-1]
            author,reviewer=people[(number-1)*2:(number-1)*2+2]
            existing=state['chapters'].get(str(number),{})
            if existing.get('status')=='model_reviewed' and (root/'chapters'/f'{number:02d}.md').exists():return existing
            context={'chapter':number,'title':title,'part':PARTS[part],'core_math':maths,'lab':lab,
                     'conventions':CONVENTIONS,'outline':CHAPTERS,'sources':SOURCES,
                     'sections':SECTIONS,'heading':f'# 第{number:02d}章 {title}',
                     'available_images':['../figures/'+x for x in ('roadmap.svg','shapes.svg','attention.svg','fusion.svg','agent-safety.svg')],
                     'target_chinese_characters':2400}
            if repair_mode and existing.get('status') != 'model_reviewed':
                source = root/'chapters'/f'{number:02d}.md'
                draft = source.read_text() if source.exists() else ''
                if validate_chapter(draft,number,final=False):
                    candidates = sorted((root/'editorial').glob(f'{number:02d}-*.md'),key=lambda f:f.stat().st_mtime,reverse=True)
                    draft = next((f.read_text() for f in candidates if not validate_chapter(f.read_text(),number,final=False)), '')
                old_reviews = [f.read_text() for f in (root/'editorial').glob(f'{number:02d}-*review.md')]
                tag = lambda step: f'repair-{generation}-{step}'
                if not draft:
                    draft = invoke(author,number,tag(0),{**context,'task':'重建完整章節；不可輸出工作計畫。'},AUTHOR_RULES)
                review = invoke(reviewer,number,tag(1),{**context,'draft':draft,'prior_reviews':old_reviews,
                                'automatic_checks':validate_chapter(draft,number,True)},REVIEW_RULES)
                approved = review.strip().endswith('VERDICT: APPROVE')
                for turn in range(0 if review_only else 2):
                    if approved and not validate_chapter(draft,number,True): break
                    draft = invoke(author,number,tag(2+turn*2),{**context,'draft':draft,'review':review,
                                   'automatic_checks':validate_chapter(draft,number,True),
                                   'measured_chinese_characters':prose_count(draft),
                                   'task':'修正具體審稿問題，保留正確內容。直接輸出完整修訂章稿，不是計畫、摘要或差異。'},AUTHOR_RULES)
                    review = invoke(reviewer,number,tag(3+turn*2),{**context,'draft':draft,'prior_review':review,
                                    'automatic_checks':validate_chapter(draft,number,True)},REVIEW_RULES)
                    approved = review.strip().endswith('VERDICT: APPROVE')
                checks = validate_chapter(draft,number,True)
                atomic(root/'chapters'/f'{number:02d}.md',draft)
                return {'status':'model_reviewed' if approved and not checks else 'needs_revision',
                        'count':prose_count(draft),'author':author,'reviewer':reviewer,
                        'automatic_issues':checks,'model_approved':approved,'repair_generation':generation}
            draft=invoke(author,number,'draft',context,AUTHOR_RULES)
            checks=validate_chapter(draft,number,final=True)
            review=invoke(reviewer,number,'review',{**context,'draft':draft,'automatic_checks':checks},REVIEW_RULES)
            revision=invoke(author,number,'revision',{**context,'draft':draft,'review':review,'automatic_checks':checks,
                            'task':'依審稿修訂，輸出完整章節，不是修訂清單。'},AUTHOR_RULES)
            checks=validate_chapter(revision,number,final=True)
            if checks:
                revision=invoke(author,number,'length-fix',{**context,'draft':revision,'automatic_checks':checks,
                                'task':'修正格式與漏節；不足要補充真正推導與解題。使用者允許超字，不為符合舊上限刪減正確內容。輸出完整章。'},AUTHOR_RULES)
            review=invoke(reviewer,number,'final-review',{**context,'draft':revision,'automatic_checks':validate_chapter(revision,number,True),
                           'earlier_review':review,'task':'複審修订後的整章，確認先前問題是否解決。'},REVIEW_RULES)
            approved=review.strip().endswith('VERDICT: APPROVE')
            if not approved:
                revision=invoke(author,number,'quality-fix',{**context,'draft':revision,'review':review,
                                'automatic_checks':validate_chapter(revision,number,True),'task':'最後一輪有限修補，輸出完整章。'},AUTHOR_RULES)
                review=invoke(reviewer,number,'quality-review',{**context,'draft':revision,'automatic_checks':validate_chapter(revision,number,True)},REVIEW_RULES)
                approved=review.strip().endswith('VERDICT: APPROVE')
            checks=validate_chapter(revision,number,True)
            atomic(root/'chapters'/f'{number:02d}.md',revision)
            return {'status':'model_reviewed' if approved and not checks else 'needs_revision','count':prose_count(revision),
                    'author':author,'reviewer':reviewer,'automatic_issues':checks,'model_approved':approved}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs={pool.submit(chapter,n):n for n in range(1,21)}
                for future in concurrent.futures.as_completed(jobs):
                    number=jobs[future]
                    try:result=future.result()
                    except Exception as exc:result={'status':'error','type':type(exc).__name__,'error':str(exc)[:1000]}
                    state['chapters'][str(number)]=result
                    print(f'第{number:02d}章：{result["status"]}',flush=True);build(root,state)
            all_ok=all(state['chapters'].get(str(n),{}).get('status')=='model_reviewed' for n in range(1,21))
            state['part_reviews']={}
            if all_ok:
                state['status']='cross_chapter_review';build(root,state)
                for part in range(5):
                    members=[n for n,c in enumerate(CHAPTERS,1) if c[1]==part]
                    fingerprint = part_fingerprint(root, part)
                    evidence = state.setdefault('part_review_evidence', {}).get(str(part), {})
                    if reusable_part_review(root, evidence, fingerprint):
                        state['part_reviews'][str(part)] = True
                        print(f'第{part+1}部：內容與契約雜湊未變，沿用已通過審稿',flush=True)
                        build(root,state)
                        continue
                    # 五部交給五個不同模型的既有審稿者，不新增人物。
                    reviewer=next(p for p in people if p['role']=='獨立審稿者' and p['model']==MODELS[part][1])
                    part_tag = f'repair-{generation}-{10+part}' if repair_mode else 'part-review'
                    review_path = root/'editorial'/f'{90+part:02d}-{part_tag}.md'
                    if review_path.exists():
                        # 無有效證據時，不讓invoke沿用與新內容不符的舊快取。
                        review_path.rename(review_path.with_name(review_path.stem+'-obsolete-'+datetime.now().strftime('%Y%m%d%H%M%S%f')+'.md'))
                    review=invoke(reviewer,90+part,part_tag,{'part':PARTS[part],'conventions':CONVENTIONS,
                                  'chapter_character_counts':{n:prose_count((root/'chapters'/f'{n:02d}.md').read_text()) for n in members},
                                  'chapters':{n:(root/'chapters'/f'{n:02d}.md').read_text() for n in members},
                                  'task':'檢查實際章節間的符號一致性、難度遞進、重複矛盾及共同養殖主軸。'},REVIEW_RULES)
                    approved = review.strip().endswith('VERDICT: APPROVE')
                    state['part_reviews'][str(part)] = approved
                    state['part_review_evidence'][str(part)] = {
                        'fingerprint':fingerprint, 'approved':approved,
                        'review_file':str(review_path.relative_to(root)),
                        'review_sha256':hashlib.sha256(review.encode()).hexdigest()}
                    build(root,state)
            all_parts=all_ok and len(state['part_reviews'])==5 and all(state['part_reviews'].values())
            state['status']='completed_model_review_pending_human' if all_parts and state['word_count']['total']>=45000 else 'needs_editorial_review'
        finally:
            if state['status'] in ('writing','repairing','cross_chapter_review'):state['status']='interrupted'
            state['updated']=datetime.now().isoformat();build(root,state)
            atomic(root/'usage.json',json.dumps(usage(root),ensure_ascii=False,indent=2))
        print(f'教材狀態：{state["status"]}；{root}/book.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='books/linear-algebra-aquaculture')
    p.add_argument('--wait-discussion',action='store_true')
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--repair',action='store_true',help='備份既有稿件，只對未通過章節新開有限修補與複審')
    p.add_argument('--review-only',action='store_true',help='搭配--repair，只複審現有定點修訂，不讓作者自動重寫')
    run(p.parse_args())
