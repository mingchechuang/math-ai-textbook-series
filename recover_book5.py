"""第29章定點修正後由原審稿者複審；不再生成巨型patch，不執行新稿程式。"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime
from types import SimpleNamespace
from book5_editor import ENGINE as E

PIN='a72da16d0169b3c8337e2e8f21fe71558499288f51c01c823afbafef1d4e1c1d'
EVIDENCE='''主編已核對原稿及原審稿，保留定義／證明／手算，定點修正：每次暫時性失敗都有ATTEMPT_FAILED、attempt與will_retry；所有FAILED有reason_code；註冊期檢查界限型別、有限值、enum型別、角色、未知設定、pattern及必填鍵，所有字串有max_length。註冊後深拷貝Schema。
AuditLogger以allow_nan=False序列化，先成功append才提交seq及tip；儲存details獨立快照；verify核對seq、長度、tip，格式／序列化錯誤回False。未授權直接輸入的keys／retry值不記錄原始值。参数及結果摘要分別對原資料重算，不互相比較。
額外發現mock結果直接別名引用受保護資料，已在結果編碼後反序列化回傳，使呼叫者不能經返回值改寫mock。這不證明任意handler唯讀；可信初始化、固定allowlist、無外部能力、單執行緒等假設已明列，計數器只是測試觀測狀態。
新增不可編碼輸入／輸出、控制鍵注入、重複註冊、不合法Schema、修改外部Schema、結果別名、日誌提交失敗、結構破壞與序號重算等測試，補權限矩陣及重試完整序列。養殖案例完全合成，不設操作門檻或給設備操作建議。
已完整閱讀後只執行SHA固定快照的22段mock Python與8項獨立核對，紀錄runs/book5-recovered29-tests.log；未載入任何後續模型新稿。此證據不是全書程式測試或人工審定。雜湊鏈只查內部一致性：獨立測試確認攻擊者整鏈重算後仍可通過，故不宣稱抗整鏈改写、尾端截斷或保密。請獨立核對當前稿件，不能以安全工具名稱當作實際能力隔離證明。'''


def recover(root,timeout=1200):
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if json.loads((root/'config.json').read_text()).get('title')!=E.S.TITLE:raise ValueError('卷別不符')
        if E.active_discussions():raise RuntimeError('其他討論正在使用模型')
        target=root/'chapters/29.md';text=target.read_text();checks=E.validate(text,29)
        if checks:raise ValueError('修訂稿未通過靜態檢查：'+str(checks))
        original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'chapters').glob('*.md')}
        state=json.loads((root/'status.json').read_text())
        backup=root/'backups'/('recovery-review-'+datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
        shutil.copy2(root/'status.json',backup/'status.json');shutil.copy2(target,backup/'29.md')
        state.update(status='writing',active={},attention=[],call_limit=None,repair_generation=state.get('repair_generation',0)+1)
        state['chapters']['29']['status']='needs_review'
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0));E.build(root,state)
        try:
            result=editor.invoke(E.S.chapter_people(29)[1],'ch29-manual-recovery-review',
                {**editor.context(29),'draft':text,'measured_characters':E.prose_count(text),'automatic_checks':checks,
                 'editorial_evidence':EVIDENCE,'current_matches_tested_snapshot':original['29.md']==PIN},E.S.REVIEW_RULES)
            approved=result.strip().endswith('VERDICT: APPROVE') and not checks
            editor.save_chapter(29,text,'model_reviewed' if approved else 'needs_revision',result)
            assert all(hashlib.sha256((root/'chapters'/name).read_bytes()).hexdigest()==sha for name,sha in original.items())
            state['status']='ready_for_cross_review' if len(state['chapters'])==30 and all(v['status']=='model_reviewed' for v in state['chapters'].values()) else 'needs_editorial_attention'
            print('Volume V chapter 29 recovery:',state['chapters']['29']['status'],flush=True)
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['第29章複審停止，證據保留：'+str(exc)[:1000]]
            raise
        finally:
            state['active']={};E.build(root,state)
