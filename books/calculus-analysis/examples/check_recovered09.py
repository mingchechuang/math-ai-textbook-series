"""只執行主編已完整閱讀、SHA256固定的第09章兩段CPU程式。
不是通用Markdown執行器；不載入日後改動的章稿。"""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from book4_editor import ENGINE as E


def main():
    snapshot=ROOT/'books/calculus-analysis/editorial/recovery-20261006/09-reviewed.md'
    raw=snapshot.read_bytes()
    expected='b59aee5ccca996598beb04449723ad9cf817db9f42697b923019c93a5285d68c'
    if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('reviewed snapshot changed; refuse execution')
    blocks=[f['content'] for f in E.markdown_structure(raw.decode())['fences'] if f['language']=='python']
    if len(blocks)!=2:raise RuntimeError('unexpected code block count')
    namespace={'__name__':'__main__'}
    for i,code in enumerate(blocks,1):exec(compile(code,f'reviewed09-block-{i}','exec'),namespace)
    print('Reviewed chapter 09 main and exercise assertions passed; snapshot SHA256:',expected)


if __name__=='__main__':main()
