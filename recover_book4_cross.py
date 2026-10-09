"""複審主編定點修正的16／25／26章；不生成新稿，不執行章內程式。"""
import fcntl
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from book4_editor import ENGINE as E
from recover_book4 import review

EVIDENCE={
16:r'''主編已獨立重算並定點修正，請依當前稿件核對，而非重播舊引句。
例16.3：H=diag(-1,2)、g=(-1.5,1)，d=(-1.5,-0.5)，g^T d=1.75>0；從(0,.5)出發則一步到鞍點，不是報非下降。馬鞍x²-y²的Newton步為(-x,-y)，預設線搜尋從(1,.1)接受完整一步至原點。近最優邊界epsilon改1e-12。
固定alpha=.01的GD估計log(1e-8/9)/log(.99)=2051.4643，首個整數2052；先前跨章審稿稱2053也是誤差。此估計不能套在alpha0=1的回溯版本。
已完整閱讀16章兩段Python，僅執行SHA固定快照。原未平移目標GD於NumPy2.2.6在10000更新達max_iter，函數求值317446次，不能捏造達標。修後練習保留這個失敗狀態，並加穩定的平移目標作診斷；後者與原函數差50.5，梯度Hessian相同。主程式、習題與六項獨立CPU核對見examples/check_cross_recovery_20261006.py及runs/book4-cross-recovery-tests.log；原失敗紀錄另保留。測試證據只及於固定快照，不代表全部版本／全書程式驗證。
亦修正章號、Wolfe及Newton全域收斂的過度聲稱、純量縮放不改變條件數、線性最小平方Hessian無殘餘二階項等。不因風格偏好拒稿。''',
25:r'''原矩陣外已乘小時^-1；括號內K必須無因次，實際右上元素為K 小時^-1。已只修正此單位句，保留其餘推導；未執行本章生成程式。''',
26:r'''請重新獨立計算，不盲從先前第五部審稿的Picard指控。對y'=cos(t)+y、y(0)=0、y0=0，原稿y3=t+1-cos(t)、y4=t+t²/2本來正確；先前審稿要求改成y3=t、y4=sin(t)+t²/2是錯誤，因為漏掉∫sin=1-cos。
保留正確原式，補完整n5,n10及有理係數遞推程式；y(1)=1.5097252536994008，n3誤差.05002755956754，n5誤差.00158760222484，n10誤差2.71406e-8。以e_(n+1)=∫e_n及非負遞增性說明sup誤差在t1；整段[0,1]上一致範數q=1不是嚴格壓縮，但階乘誤差界證明一致收斂。
例26.9指定a=1，半徑min(1,1/17,1/(2sqrt21))=1/17，不將0.059當保證半徑。
已完整閱讀並僅執行SHA固定快照的新增習題程式，獨立核對係數導數與上一個Picard迭代的恆等式，六項CPU核對通過，見runs/book4-cross-recovery-tests.log。未執行原章RK4等程式，不把預期當實測。亦限定唯一性比較解取值於所述閉球、有限時間爆破需終點仍在時間定義域內。'''}


def recover(root,timeout=1200):
    root=Path(root).resolve()
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if json.loads((root/'config.json').read_text()).get('title')!=E.S.TITLE:raise ValueError('卷別不符')
        if E.active_discussions():raise RuntimeError('其他討論正在使用模型')
        for n in EVIDENCE:
            checks=E.validate((root/'chapters'/f'{n:02d}.md').read_text(),n)
            if checks:raise ValueError(f'第{n}章靜態檢查未過：{checks}')
        state=json.loads((root/'status.json').read_text())
        backup=root/'backups'/('cross-review-'+datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
        shutil.copy2(root/'status.json',backup/'status.json')
        original={p.name:E.digest(p.read_text()) for p in (root/'chapters').glob('*.md')}
        state.update(status='writing',active={},attention=[],call_limit=None,repair_generation=state.get('repair_generation',0)+1)
        for n in EVIDENCE:state['chapters'][str(n)]['status']='needs_review'
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0));E.build(root,state)
        try:
            for n,evidence in EVIDENCE.items():review(editor,n,evidence)
            assert all(E.digest((root/'chapters'/name).read_text())==value for name,value in original.items()),'複審階段不應修改稿件'
            state['status']='ready_for_cross_review' if all(v['status']=='model_reviewed' for v in state['chapters'].values()) else 'needs_editorial_attention'
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['主編修訂複審停止：'+str(exc)[:1000]]
            raise
        finally:
            state['active']={};E.build(root,state)
