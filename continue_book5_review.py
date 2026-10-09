"""Volume V：先驗證五個原模型，再初稿、逐章及跨章有限續跑。"""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from book5_editor import ENGINE as E
from continue_book2_review import continue_review


def preflight(root,timeout):
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'status.json').read_text())
        editor=E.Editor(root,state,SimpleNamespace(timeout=timeout,max_calls=0))
        results=[]
        try:
            for person in E.S.roster()[:5]:
                result=editor.invoke(person,'preflight-'+person['id'],
                    {'task':'確認能以指定原模型撰寫繁體中文數學教材；只回READY。','volume':E.S.TITLE},
                    '本次只測API及實際模型識別；不用工具。只回READY。')
                if result.strip()!='READY':raise ValueError('preflight未回READY：'+person['id'])
                results.append({'person':person,'result':result,'identity_verification':'llm_agent事件中的provider/model驗證，原始事件保留'})
                E.atomic(root/'editorial/preflight.json',json.dumps(results,ensure_ascii=False,indent=2))
        except Exception as exc:
            state['status']='needs_editorial_attention';state['attention']=['五模型preflight失敗：'+str(exc)[:600]]
            raise
        finally:
            state['active']={};E.build(root,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=E.S.ROOT)
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-passes',type=int,default=12)
    parser.add_argument('--recover-stalled',action='store_true',help='先複審已定點修正的第29章，再自動續跑跨章流程')
    parser.add_argument('--recover-cross-stalled',action='store_true',help='複審已定點修正的15／16／17／26／30章，再續跑')
    parser.add_argument('--review-chapters',type=int,nargs='+',choices=(15,16,17,26,30),help='限縮跨章修復的原審稿者複審章號')
    args=parser.parse_args();root=Path(args.output).resolve()
    if args.review_chapters and not args.recover_cross_stalled:
        parser.error('--review-chapters須與--recover-cross-stalled一起使用')
    if args.recover_stalled and args.recover_cross_stalled:
        parser.error('兩種修復入口不能同時使用')
    E.run(SimpleNamespace(output=str(root),init_only=True,timeout=args.timeout,max_calls=0,targeted_repair=False))
    if not (root/'editorial/preflight.json').exists() or len(json.loads((root/'editorial/preflight.json').read_text()))!=5:
        preflight(root,args.timeout)
    if args.recover_cross_stalled:
        from recover_book5_cross import recover
        recover(root,args.timeout,chapters=args.review_chapters)
    if args.recover_stalled:
        from recover_book5 import recover
        recover(root,args.timeout)
    state=json.loads((root/'status.json').read_text())
    if not state.get('chapters'):
        E.run(SimpleNamespace(output=str(root),init_only=False,timeout=args.timeout,max_calls=0,targeted_repair=False))
    if json.loads((root/'status.json').read_text()).get('status')!='completed_model_review_pending_human':
        continue_review(root,args.max_passes,args.timeout,engine=E)
