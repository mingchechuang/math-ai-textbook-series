"""第四卷停滯修復：第09章證據複審、第19／29章分段候選；不執行生成程式。"""
import fcntl
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from book4_editor import ENGINE as E
from recover_book3 import split_sections

SECTION_RULES=r'''你是Volume IV的第三模型修復主編。本次不用OLD/NEW patch格式，也不用JSON。只回傳requested_sections指定的連續小節，原始Markdown，不含一級章標題、不包整篇程式圍欄、不加前後說明。二級標題名稱及順序必須完全一致，可用三級標題。
保留原稿中正確的定義、證明與手算結果，只修有證據的錯誤及補足教學缺漏。刪除自問自答、未完成式、pass/TODO/omitted及沒有定義的接口。每批至少1600中文字，公式、英文、程式不充字數；整章至少3000字，不強求4500字。
遵守本卷共同契約，完整列出定理假設、函數空間／範數、正常／邊界／故障測試及習題解答。不使用工具、不執行程式、不捏造測試紀錄。JSON傳輸的換行不等於LaTeX的nu；以解碼後原文核對。'''
CONTRACTS={
19:r'''第19章是多重積分、Riemann和與Fubini。保留正確的Darboux上下和、緊矩形上連續函數的可積性及Fubini證明；若引述進階可測／零測度判準，標作未證明延伸或刪除。本章核心使用緊矩形上連續函數版本，不混淆充分與必要條件。
原稿「逐步手算例題」已獨立核對：3x²+2y在[0,1]×[0,2]積分為6；x+y在0<=y<=x<=1積分為1/2。這個小節將原樣保留，輸出不重寫它。
原點瑕積分反例(x²-y²)/(x²+y²)²應限[0,1]²：先對x再y為-pi/4，反序為pi/4。刪除無界象限的错误複述。習題(x-y)/(x+y)²則絕對可積：|f|<=1/(x+y)，後者在單位正方形積分2log2，且反對稱使總積分0。可將習題改成證明它的兩種瑕迭代積分皆0，不能要求讀者證明假的不等式；邊界截面瑕積分與連續矩形定理需分開。
補零標記Riemann和用完整cell面積乘補零後取樣值；交集面積加權是另一求積策略，不能說前者必須算交集。Jordan邊界結論列条件，不宣稱任意區域的誤差必定O(h)。
程式只需NumPy二維中點及梯形，統一method='midpoint'/'trapezoidal'。使用q[j,i]對應(y,x)，拒絕布林／非整數／非正nx,ny、非有限端點、反向或退化矩形、非法method及不合輸出shape／非有限值。常數函數可以明確支援broadcast或要求同shape，二擇一且測試一致。刪佔位函式，提供正常、極小非零矩形、非法方法、nx=0等斷言。範例3x²+2y中，中點值6-1/(2n²)，梯形值6+1/n²；n=10的中點誤差0.005，n=200為1.25e-5，二階不是四階或指數。
養殖模型C=5-0.1x²-0.05y²在[0,10]×[0,5]的面積平均為1.25 mg/L，角點為-6.25 mg/L；係數單位mg L^-1 m^-2。若保留此例，明示是局部模型失效反例，不把負濃度裁零冒充修正，不給曝氣等操作建議或安全閾值。圓域C0-k r²平均C0-k R²/2是面積平均濃度，不是總氧量。
A6是scipy.optimize.minimize，不是積分文件，刪錯配引用；保留A1/A3及實際可追溯來源。所有程式結果僅預期。''',
29:r'''第29章是變分法、第一變分與能量泛函。原稿不足3000字且程式與習題有實質錯誤，必須補推導、域／範數、兩個完整手算及解答，不只縮句。
不要預設已教完整Sobolev理論。可先在一維C1([0,1])及其C1範數推導光滑L的第一變分，再用H1/H1_0列清定義與引用的嵌入／Poincare條件。二階充分條件為駐點加相對所選範數的一致強制性及有o(||h||²)餘項；這是充分條件，不能說所有局部極小必須強制性。給完整小命題證明。
第一變分的交換導數與積分須有支配或緊域連續條件，弱EL需容許擾動；經典式需合成場L_(grad u)(x,u,grad u)有相应可微性，不能只說u∈C2就足夠。混合邊界自然條件僅在自由端／Gamma_N，不是全邊界。
保留正確手算：J=1/2∫u'²-∫fu，f=pi²sin(pi x)，零Dirichlet下u=sin(pi x)，J=-pi²/4。用能量差為1/2∫v'²說明全域最小，不把局部Hessian定理當全域證明。混合端點例J=∫[u'²/2+u^4/4-gu]可取u=x(2-x)、g=2+[x(2-x)]³，則u(0)=0、u'(1)=0並滿足-u''+u³-g=0，提供完整代回。
N明確表示分段數，零Dirichlet自由度N-1；所有方向V與U同shape，以固定seed或確定方向。J_h=∑(U_(i+1)-U_i)²/(2h)-h∑f_i U_i，Euclidean梯度(2U_j-U_(j-1)-U_(j+1))/h-h f_j。DJ_h[ V ]=gradient_E^T V，不能多乘h。若用M=hI定義離散L2梯度g_M，則M g_M=gradient_E。連續梯度也取決於泛函及內積，不是普遍等於u。
實作需要驗證shape、h>0、有限值，拒絕非法維度及NaN；小型NumPyCPU，中央方向差分至少一個非駐點（不可只在梯度零的解上測）。加本卷要求的週期相場離散能量F_h=h∑(phi²-1)²/4+kappa/(2h)∑(phi_(i+1)-phi_i)²，其Euclidean梯度h*(phi³-phi-kappa*L_h phi)，核對有限差分；這不是執行AC/CH時間模擬。
習題1 J=∫(u'²-u)給-2u''-1=0；若零端點則u=x(1-x)/4。Neumann習題不能加epsilon I當原方程：可用cell中心有限體積零通量的Laplacian與mean-zero增廣系統，列完整矩陣、相容性及非法常數右端測試，cell與node配置不可混用。最小N若不支援明確拒絕，不留空解。
鞍點反例可用J[u]=1/2∫(u'²-2pi²u²)於H1_0(0,1)，u=0為駐點，方向sin(pi x)與sin(2pi x)的第二變分分別-pi²/2及pi²，一負一正。不要把W''(±1)>0假說成鞍點。H1與L2梯度比較需固定泛函、內積、邊界，再給Riesz表示，不泛稱L2梯度為自身。
養殖應用只用明確合成無因次平滑泛函或完整定義的有界線性算子G，不冒充控制設備。刪除已通過數值測試等無證據句子；未執行的生成程式均標預期。
新增可追溯參考：Daniel Liberzon, Calculus of Variations and Optimal Control Theory, 1.3.2 First variation（https://liberzon.csl.illinois.edu/teaching/cvoc/node15.html）、2.3.1 Euler–Lagrange（https://liberzon.csl.illinois.edu/teaching/cvoc/node28.html）、2.3.5 Variable-endpoint problems（https://liberzon.csl.illinois.edu/teaching/cvoc/node32.html）。主編已取得這些HTML並局部核對自由端點的容許擾動與自然條件，未逐條查核全書，不複製原文。'''}


def sections_of_chapter(text):
    lines=text.splitlines(keepends=True)
    headings=[h for h in E.markdown_structure(text)['headings'] if h['level']==2]
    if not headings:raise ValueError('缺小節')
    return split_sections(''.join(lines[headings[0]['map'][0]:]))


def group(editor,person,n,batch,names,ctx):
    context={**ctx,'requested_sections':names,'minimum_chinese_prose_for_batch':1600}
    for attempt in range(2):
        result=editor.invoke(person,f'ch{n:02d}-recovery-sections-{batch}',context,SECTION_RULES)
        try:
            sections=split_sections(result)
            if list(sections)!=names:raise ValueError('小節名稱／順序不符')
            if E.prose_count(result)<1600:raise ValueError('分段正文未達1600中文字，補推導與教學而非重複句子')
            return sections
        except ValueError as exc:
            if attempt==1:raise
            context={**context,'previous_output':result,'format_error':str(exc)}


def review(editor,n,evidence):
    text=(editor.root/'chapters'/f'{n:02d}.md').read_text();checks=E.validate(text,n)
    result=editor.invoke(E.S.chapter_people(n)[1],f'ch{n:02d}-recovery-review',
        {**editor.context(n),'draft':text,'measured_characters':E.prose_count(text),'automatic_checks':checks,
         'editorial_evidence':evidence,'review_note':'核對當前稿件；3000是最低字數，4500只是目標。依事實重算，不重播已解決的舊意見。'},E.S.REVIEW_RULES)
    approved=result.strip().endswith('VERDICT: APPROVE') and not checks
    editor.save_chapter(n,text,'model_reviewed' if approved else 'needs_revision',result)
    print(f'Volume IV recovery chapter {n}: {editor.state["chapters"][str(n)]["status"]}',flush=True)


def recover(root,timeout=1200):
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'status.json').read_text())
        if json.loads((root/'config.json').read_text()).get('title')!=E.S.TITLE:raise ValueError('卷別不符')
        if E.active_discussions():raise RuntimeError('其他討論正在使用模型')
        preserved={p.name:E.digest(p.read_text()) for p in (root/'chapters').glob('*.md') if p.stem not in ('09','19','29')}
        backup=root/'backups'/('segmented-recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
        for name in ('status.json','book.md','chapters/09.md','chapters/19.md','chapters/29.md'):
            target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
        state.update(status='writing',active={},attention=[],call_limit=None,repair_generation=state.get('repair_generation',0)+1)
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0));E.build(root,state)
        try:
            if state['chapters']['9']['status']!='model_reviewed':
                review(editor,9,'主編已定點修正範數、column協向量、鏈式shape、平方根跳脫、VJP斷言、精確邊界測試及習題2完整程式，補充數值驗證的界線。另經完整閱讀後，僅對SHA256=b59aee5ccca996598beb04449723ad9cf817db9f42697b923019c93a5285d68c快照的兩段Python做CPU執行，主程式及習題斷言通過，主JVP/VJP誤差為1.08e-9/8.80e-10，紀錄runs/book4-recovered09-tests.log。此實驗不等於定理證明或所有版本／章稿已執行；仍需獨立核對當前數學及內容。')
            for n in (19,29):
                if state['chapters'][str(n)]['status']=='model_reviewed':continue
                original=(root/'chapters'/f'{n:02d}.md').read_text();old_sections=sections_of_chapter(original)
                lead=next(p for p in E.S.roster() if p['role']=='author' and p['model']=='gpt-6-sol')
                ctx={**editor.context(n),'original_draft':original,'previous_review':state['chapters'][str(n)].get('review',''),
                     'repair_contract':CONTRACTS[n],'task':'分段修復當前章；保留正確內容，補足實質缺漏，不作全文patch。'}
                first_names=E.S.SECTIONS[:3] if n==19 else E.S.SECTIONS[:4]
                first=group(editor,lead,n,1,first_names,ctx)
                if n==19:first[E.S.SECTIONS[3]]=old_sections[E.S.SECTIONS[3]]
                second=group(editor,lead,n,2,E.S.SECTIONS[4:],{**ctx,'accepted_first_sections':first})
                candidate=f'# 第{n:02d}章 '+E.S.CHAPTERS[n-1][0]+'\n\n'+'\n'.join((first|second)[name] for name in E.S.SECTIONS)
                E.atomic(root/'editorial'/f'ch{n:02d}-segmented-candidate.md',candidate)
                issues=E.validate(candidate,n)
                if issues:raise ValueError('候選靜態檢查失敗：'+'；'.join(issues))
                editor.save_chapter(n,candidate,'needs_review','第三模型分段修復，尚未執行生成程式，待原審稿者複審')
                review(editor,n,CONTRACTS[n])
            assert all(E.digest((root/'chapters'/name).read_text())==digest for name,digest in preserved.items())
            state['status']='ready_for_cross_review' if all(c['status']=='model_reviewed' for c in state['chapters'].values()) else 'needs_editorial_attention'
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['分段修復中止，候選及事件已保留：'+str(exc)[:1200]]
            raise
        finally:
            state['active']={};E.build(root,state)
