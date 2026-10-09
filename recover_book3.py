"""第三卷停滯修復：分段Markdown候選、原審稿者複審；不執行生成程式。"""
import fcntl
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from book3_editor import ENGINE as E

SECTION_RULES='''你是Volume III的第三模型修復主編。只回傳指定的連續小節，使用原始Markdown，不用JSON、不包整篇程式圍欄、不輸出一級章標題。二級標題及順序必須完全等於requested_sections；可用三級標題。
依證據修復原稿，保留有效解釋及手算例，不保留自問自答、錯式再撤回、未完成程式或不相關金融類比。這是修復第19章，不改其他章。遵守共同契約、量綱、符號和輸出範圍；未執行結果寫預期。不使用工具、不執行程式。每批補足必要的中文教學，兩批合計至少3000中文字，公式／程式不充字數。
注意JSON傳輸的\\n是換行，不是LaTeX的\\nu；判斷實際解碼文字，不把換行後的u誤看成希臘nu。'''
REPAIR_CONTRACT='''第19章主題是不可壓流體與動量模型，核心驗收是完整二維製造流場的散度及動量殘差。第20章才是MAC壓力投影實作。因此不要試圖修補成龐大的時間積分器；移除原稿不完整的IncompressibleSolver，換成自足、雙分量、NumPy CPU的製造解驗證程式。這仍須完整實作，不能省略v或使用未定義函式。
保留並修正一維MAC投影手算例；清楚說明這是投影代數，不是黏性時間演化。週期壓力矩陣A=-L有常數零空間，僅在零均值子空間正定，右端b=-(rho/dt)D u_star。四格例最終零均值p=[-1.875,1.875,0.625,-0.625]及校正速度[1.5]*4可保留，錯誤中間代數須刪除重推。N=2周期鄰居重合，A=[[2,-2],[-2,2]]，rho=dt=dx=1時初速[1,-1]給b=[2,-2]，零均值p=[.5,-.5]，校正後速度全零。
製造場可選k=2*pi/L、a(t)=U*exp(-alpha*t)，u=a*sin(kx)*cos(ky)，v=-a*cos(kx)*sin(ky)，p=rho*a^2*(cos(2kx)+cos(2ky))/4；外力加速度f=(2*nu*k^2-alpha)*(u,v)。散度解析為零，壓力平衡對流，兩動量分量都要核對。alpha=2*nu*k^2時為無外力Taylor–Green；Stokes省略平流時不可沿用這個非恆定壓力而不修改外力。
程式在均勻正方形週期cell中心取樣速度和壓力，明示這不是MAC配置及不是時間積分器。中央差分空間導數、解析時間導數，分別計算散度與u/v動量殘差。16/32/64格細化對照解析場，觀測二階空間誤差；不宣稱做過時間收斂。核對兩分量、shape、有限輸入、非法L/rho/nu/grid及非零外力例；只做真有執行程式對應的預期診斷。速度非零不能依低Re一概保證穩定，無通用層流Re閾值。壓力不是抵抗密度堆積的可壓模型，不可壓壓力是約束反應。
若讨论能量，包含u和v，二維積分明示每單位厚度。顯式Stokes黏性穩定條件只作理論延伸，不假稱本章殘差程式有時間更新。無因次外力f*=L*f/U^2。養殖案例只能是合成流場與溶質傳輸接口；相場耦合需額外體力和模型，溶氧閾值不是相變。習題四類都須完整解答，包含兩格壓力、雙分量製造殘差及有限差分誤差／假反例。'''


def split_sections(text):
    headings=E.markdown_structure(text)['headings']
    if any(h['level']==1 for h in headings):raise ValueError('分段候選不得含一級標題')
    headings=[h for h in headings if h['level']==2]
    lines=text.splitlines(keepends=True)
    if not headings or ''.join(lines[:headings[0]['map'][0]]).strip():raise ValueError('二級標題前有額外內容')
    names=[h['text'] for h in headings]
    if len(set(names))!=len(names):raise ValueError('重複小節')
    result={}
    for i,h in enumerate(headings):
        end=headings[i+1]['map'][0] if i+1<len(headings) else len(lines)
        result[h['text']]=''.join(lines[h['map'][0]:end]).strip()+'\n'
    return result


def group(editor,person,number,names,context):
    ctx={**context,'requested_sections':names,'minimum_chinese_prose_for_this_batch':1800}
    for attempt in range(2):
        text=editor.invoke(person,f'ch19-recovery-sections-{number}',ctx,SECTION_RULES)
        try:
            sections=split_sections(text)
            if list(sections)!=names:raise ValueError('小節名稱／順序不符')
            if E.prose_count(text)<1800:raise ValueError('分段正文未達1800中文字，須補教學與解答')
            return sections
        except ValueError as exc:
            if attempt==1:raise
            ctx={**ctx,'previous_output':text,'format_error':str(exc)}


def review(editor,n,evidence):
    text=(editor.root/'chapters'/f'{n:02d}.md').read_text()
    checks=E.validate(text,n)
    response=editor.invoke(E.S.chapter_people(n)[1],f'ch{n:02d}-recovery-review',
        {**editor.context(n),'draft':text,'measured_characters':E.prose_count(text),
         'automatic_checks':checks,'independent_editorial_evidence':evidence},E.S.REVIEW_RULES)
    approved=response.strip().endswith('VERDICT: APPROVE') and not checks
    editor.save_chapter(n,text,'model_reviewed' if approved else 'needs_revision',response)
    print(f'Volume III recovery chapter {n}: {editor.state["chapters"][str(n)]["status"]}',flush=True)


def recover(root,timeout=1200):
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'status.json').read_text())
        if json.loads((root/'config.json').read_text()).get('title')!=E.S.TITLE:raise ValueError('卷別不符')
        if E.active_discussions():raise RuntimeError('其他討論正在使用模型')
        preserved={p.name:E.digest(p.read_text()) for p in (root/'chapters').glob('*.md') if p.stem not in ('17','19')}
        backup=root/'backups'/('segmented-recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        backup.mkdir(parents=True)
        for name in ('status.json','book.md','chapters/17.md','chapters/19.md'):
            target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
        state.update(status='writing',active={},attention=[],call_limit=None,repair_generation=state.get('repair_generation',0)+1)
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0))
        E.build(root,state)
        try:
            if state['chapters']['17']['status']!='model_reviewed':
                review(editor,17,'主編檢查實際UTF-8原稿：先前被指出的三式實際皆為Latin u，不是LaTeX命令\\nu。先前JSON中的\\n是换行。現僅把這三式排成同一行$$ u... $$以消除辨讀歧義，數學未變。請以當前稿件重算，不重播舊意見。')
            if state['chapters']['19']['status']!='model_reviewed':
                original=(root/'chapters/19.md').read_text()
                lead=next(p for p in E.S.roster() if p['role']=='author' and p['model']=='gpt-6-sol')
                ctx={**editor.context(19),'original_draft':original,'previous_review':state['chapters']['19'].get('review',''),
                     'repair_contract':REPAIR_CONTRACT,'task':'分段修復，保留正確教學，移除失敗草稿痕跡；只產生指定小節。'}
                first=group(editor,lead,1,E.S.SECTIONS[:4],ctx)
                second=group(editor,lead,2,E.S.SECTIONS[4:],{**ctx,'accepted_first_sections':first})
                candidate='# 第19章 '+E.S.CHAPTERS[18][0]+'\n\n'+'\n'.join((first|second)[name] for name in E.S.SECTIONS)
                E.atomic(root/'editorial/ch19-segmented-candidate.md',candidate)
                issues=E.validate(candidate,19)
                if issues:raise ValueError('候選未過靜態檢查：'+'；'.join(issues))
                editor.save_chapter(19,candidate,'needs_review','第三模型分段Markdown修復；未執行生成程式，待原審稿者複審')
                review(editor,19,REPAIR_CONTRACT+' 此次改用完整二維製造流場驗證取代壞掉且越過本章scope的時間積分器，並未聲稱完成CFD或執行過本章程式。請核對新稿，不要求恢復原稿無效介面。')
            assert all(E.digest((root/'chapters'/name).read_text())==value for name,value in preserved.items())
            state['status']='needs_editorial_attention' if any(c['status']!='model_reviewed' for c in state['chapters'].values()) else 'ready_for_cross_review'
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['分段修復中止，保留候選及錯誤：'+str(exc)[:1200]]
            raise
        finally:
            state['active']={};E.build(root,state)
