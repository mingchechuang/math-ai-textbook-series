"""等待目前教材工作，必要時再做一代有限修復；不無限重跑。"""
import fcntl
import json
import subprocess
import sys
from pathlib import Path

root = Path('books/linear-algebra-aquaculture').resolve()
with (root / '.run.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads((root / 'status.json').read_text())
if state['status'] in ('needs_editorial_review', 'interrupted'):
    print('使用更新的相容性修正再做一代有限修復，保留已通過章節。', flush=True)
    subprocess.run([sys.executable, '-u', 'book_editor.py', '--output', str(root),
                    '--repair', '--wait-discussion'], check=False)
else:
    print('不啟動額外修復；教材狀態：' + state['status'], flush=True)
