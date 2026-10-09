"""Volume V跨章定點修訂後的原審稿者複審；不執行新生成程式。"""
import fcntl
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from book5_editor import ENGINE as E

EVIDENCE={
15:'已核對當前稿件，保留既有絕對位置causal規則。allowed原已有嚴格布林驗證；本次补key_is_valid/query_is_valid/valid_targets的dtype檢查，不再靜默將數值轉bool，補四入口故障斷言。只對完整閱讀、SHA固定快照的NumPy主程式做CPU測試，見runs/book5-cross-recovery-tests.log，不代表後續版本已執行。',
16:r'''舊跨章審稿中章號、mask前導軸與一般參數數量公式已在前輪修好，本次保留；未盲從過時引句。新增完整backward及forward快照：輸入三路梯度相加、四個權重及所有bias梯度，無多除平均。固定mask的softmax VJP在遮罩位置為0。支持每頭dv，V總寬H*dv與WO第一維相同，Q/K縮放仍dh。
修正維度非bool正整數、非有限輸入及參數、非有限scores保守拒絕、例外TypeError敘述、H1與單頭同參數等價。刪除固定身份矩陣手算『學到了』與『頭塌縮』之過度聲稱，columns與投影後座標分塊明確。PyTorch選用答案改.double().cpu()、四矩陣轉置、bias分段、布林mask反轉及完整比較；未執行PyTorch，不宣稱已驗證其環境。
僅完整閱讀後執行SHA256=62565894565056ac6bf70042506d5c08811eb47e27ae6213b10ac7af947da8ee快照的NumPy片段與獨立核對。CPU Python3.10.12 NumPy2.2.6，輸入／全部W／bias、無mask／causal／dv!=dh的有限差分最大誤差1.365083046600546e-10；另測未來token不洩漏、單頭對照、snapshot、mask、量化與視窗，8項核對通過。紀錄runs/book5-cross-recovery-tests.log。此證據不是全書程式執行或定理證明。''',
17:'只將來源尾註改成入口及按實際版本核對，不把題目提供的URL或來源附註當成作者親自查閱／執行證據；保留RoPE推導與程式，本次未執行本章程式。',
26:'只修正文「量化必然引入誤差」為「可能引入誤差；格點值可精確還原」。原零輸入與0.5格點例、s/2上界與蒸餾推導都保留；未執行本章程式。',
30:'只同步資料切分段：步長1才高度重疊；本程式步長context，輸入窗口不重疊，位移target可共用邊界token。無論重疊與否，仍先按文件group切分再建窗。保留完整模型及訓練loop；本次未執行PyTorch或訓練。'}


def recover(root,timeout=1200,chapters=None):
    selected=tuple(EVIDENCE) if chapters is None else tuple(chapters)
    if not selected or len(set(selected))!=len(selected) or any(n not in EVIDENCE for n in selected):
        raise ValueError('複審章號須唯一且屬已修訂章節')
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if json.loads((root/'config.json').read_text()).get('title')!=E.S.TITLE:raise ValueError('卷別不符')
        if E.active_discussions():raise RuntimeError('其他討論正在使用模型')
        for n in selected:
            checks=E.validate((root/'chapters'/f'{n:02d}.md').read_text(),n)
            if checks:raise ValueError(f'第{n}章靜態檢查失敗：{checks}')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'chapters').glob('*.md')}
        state=json.loads((root/'status.json').read_text())
        backup=root/'backups'/('cross-review-'+datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
        shutil.copy2(root/'status.json',backup/'status.json')
        state.update(status='writing',active={},attention=[],call_limit=None,repair_generation=state.get('repair_generation',0)+1)
        for n in selected:state['chapters'][str(n)]['status']='needs_review'
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0));E.build(root,state)
        try:
            for n in selected:
                evidence=EVIDENCE[n]
                if n==16:
                    evidence+='\n本輪再補清：基本shape與3D²+3D+D*Dout+Dout參數式預設dv=dh；一般dv總数為2D²+2D+(D+1)H*dv+(H*dv+1)Dout。只改正文兩段，所有Python片段保持原快照內容；不沿用未註明範圍的旧引句。'
                text=(root/'chapters'/f'{n:02d}.md').read_text();checks=E.validate(text,n)
                result=editor.invoke(E.S.chapter_people(n)[1],f'ch{n:02d}-cross-recovery-review',
                    {**editor.context(n),'draft':text,'measured_characters':E.prose_count(text),
                     'automatic_checks':checks,'editorial_evidence':evidence},E.S.REVIEW_RULES)
                approved=result.strip().endswith('VERDICT: APPROVE') and not checks
                editor.save_chapter(n,text,'model_reviewed' if approved else 'needs_revision',result)
                print(f'Volume V chapter {n}: {state["chapters"][str(n)]["status"]}',flush=True)
            assert all(hashlib.sha256((root/'chapters'/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
            state['status']='ready_for_cross_review' if len(state['chapters'])==30 and all(v['status']=='model_reviewed' for v in state['chapters'].values()) else 'needs_editorial_attention'
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['定點修訂複審停止：'+str(exc)[:1000]]
            raise
        finally:
            state['active']={};E.build(root,state)
