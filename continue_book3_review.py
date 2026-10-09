"""Volume III初稿、審稿與有限回合續跑；無總呼叫額度上限。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from book3_editor import ENGINE
from continue_book2_review import continue_review


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=ENGINE.S.ROOT)
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-passes',type=int,default=12)
    parser.add_argument('--recover-stalled',action='store_true',help='先做第17／19章證據導向分段修復，再接續原審稿流程')
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if args.recover_stalled:
        from recover_book3 import recover
        recover(root,args.timeout)
    status=root/'status.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    if not state.get('chapters'):
        ENGINE.run(SimpleNamespace(output=str(root),init_only=False,timeout=args.timeout,max_calls=0,targeted_repair=False))
        state=json.loads(status.read_text())
    if state.get('status')!='completed_model_review_pending_human':
        continue_review(root,args.max_passes,args.timeout,engine=ENGINE)
