"""Volume II有限回合主編續跑：使用者授權不設呼叫總額上限，仍防止無效循環。"""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import book2_editor as B


def signature(state):
    return B.digest({'chapters':{k:(v.get('status'),v.get('sha256')) for k,v in state['chapters'].items()},
                     'parts':{k:(v.get('approved'),v.get('fingerprint')) for k,v in state.get('part_reviews',{}).items()}})


def continue_review(root,max_passes=12,timeout=1200,engine=None):
    workflow=engine or B
    label=getattr(workflow.S,'LABEL','Volume II')
    root=Path(root).resolve()
    if max_passes<1:raise ValueError('max_passes必須大於0')
    with (root/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stalled=0
        for cycle in range(1,max_passes+1):
            before=json.loads((root/'status.json').read_text())
            print(f'續跑回合{cycle}/{max_passes}，累計{before["calls"]}次；不設總呼叫上限',flush=True)
            workflow.run(SimpleNamespace(output=str(root),init_only=False,timeout=timeout,max_calls=0,targeted_repair=True))
            after=json.loads((root/'status.json').read_text())
            if after['status']=='completed_model_review_pending_human':
                print('逐章及五部跨章模型審查完成；仍待人工驗證。',flush=True);return
            if after['status']=='interrupted':return
            stalled=stalled+1 if signature(before)==signature(after) else 0
            if stalled>=3:break
        # 不把無預算上限誤當無限循環；保留明確停因。
        with (root/'.run.lock').open('a') as run_lock:
            fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=json.loads((root/'status.json').read_text())
            message='續跑保護停止：連續3輪無內容／審查進展。' if stalled>=3 else f'續跑保護停止：已完成{max_passes}輪，需檢視剩餘問題；不是預算停止。'
            state.setdefault('attention',[]).append(message)
            workflow.build(root,state)
            workflow.atomic(root/'RUN_SUMMARY.md','# '+label+'續跑結果\n\n'+message+'\n\n'+'\n'.join('- '+x for x in state['attention']))
            print(message,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='books/modern-graphics')
    p.add_argument('--max-passes',type=int,default=12)
    p.add_argument('--timeout',type=int,default=1200)
    args=p.parse_args()
    continue_review(args.output,args.max_passes,args.timeout)
