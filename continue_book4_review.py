"""第四卷初稿、逐章與跨章續跑；有限重試，不設總呼叫額度上限。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from book4_editor import ENGINE
from continue_book2_review import continue_review


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=ENGINE.S.ROOT)
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-passes',type=int,default=12)
    parser.add_argument('--recover-stalled',action='store_true',help='先複審第09章，分段修復第19／29章，再接續原審稿流程')
    parser.add_argument('--recover-cross-stalled',action='store_true',help='複審主編修正的16／25／26章，再續跑跨章審稿')
    args=parser.parse_args();root=Path(args.output).resolve()
    if args.recover_stalled and args.recover_cross_stalled:
        parser.error('兩種修復入口不可同時使用')
    if args.recover_cross_stalled:
        from recover_book4_cross import recover
        recover(root,args.timeout)
    if args.recover_stalled:
        from recover_book4 import recover
        recover(root,args.timeout)
    status=root/'status.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    if not state.get('chapters'):
        ENGINE.run(SimpleNamespace(output=str(root),init_only=False,timeout=args.timeout,max_calls=0,targeted_repair=False))
        state=json.loads(status.read_text())
    if state.get('status')!='completed_model_review_pending_human':
        continue_review(root,args.max_passes,args.timeout,engine=ENGINE)
