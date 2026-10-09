"""本機期貨模擬器：僅使用 Python 標準函式庫。"""
import argparse
import csv
import fcntl
import shutil
import json
import random
import shlex
import sqlite3
import subprocess

import llm_agent
import report
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@dataclass
class Bar:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


def load_bars(files, start=None, end=None):
    """來源時間為 K 棒起始時間；只採用完整且非合成的五分鐘。"""
    minutes = {}
    for file in files:
        with open(file, encoding='utf-8') as stream:
            in_copy = False
            for line in stream:
                if line.startswith('COPY futures_1min '):
                    in_copy = True
                    continue
                if line.startswith('\\.'):
                    in_copy = False
                if not in_copy:
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 9 or parts[1] != 'TXFR1':
                    continue
                ts = datetime.fromisoformat(parts[0])
                if start and ts.date().isoformat() < start:
                    continue
                if end and ts.date().isoformat() > end:
                    continue
                minute = ts.hour * 60 + ts.minute
                if not (525 <= minute < 825 or minute >= 900 or minute < 300):
                    continue
                if parts[8] != 'f':
                    continue
                value = tuple(parts[2:7])
                if ts in minutes and minutes[ts] != value:
                    raise ValueError(f'重複時間資料衝突：{ts}')
                minutes[ts] = value
    buckets = {}
    for ts in sorted(minutes):
        key = ts.replace(minute=ts.minute // 5 * 5, second=0)
        buckets.setdefault(key, []).append((ts, minutes[ts]))
    bars = []
    for ts, rows in sorted(buckets.items()):
        if [r[0] for r in rows] != [ts + timedelta(minutes=i) for i in range(5)]:
            continue
        values = [r[1] for r in rows]
        bars.append(Bar(ts.isoformat(' '), float(values[0][0]),
                        max(float(v[1]) for v in values), min(float(v[2]) for v in values),
                        float(values[-1][3]), sum(int(v[4]) for v in values)))
    return bars


@dataclass
class Account:
    cash: float = 1_000_000
    position: int = 0
    average: float = 0
    realized: float = 0
    fees: float = 0
    bankrupt: bool = False

    def equity(self, price):
        return self.cash + self.position * (price - self.average) * 10

    def trade(self, target, price):
        old = self.position
        delta = target - old
        if not delta:
            return 0.0, 0.0
        closed = min(abs(old), abs(delta)) if old * delta < 0 else 0
        pnl = closed * (price - self.average) * (1 if old > 0 else -1) * 10
        fee = abs(delta) * 15
        self.cash += pnl - fee
        self.realized += pnl
        self.fees += fee
        if target == 0:
            self.average = 0
        elif old == 0 or old * target < 0:
            self.average = price
        elif abs(target) > abs(old):
            self.average = (abs(old) * self.average + abs(delta) * price) / abs(target)
        self.position = target
        return pnl, fee


class Trader:
    def __init__(self, number, root, seed, engine='builtin', provider=None, model=None, resume=False):
        self.id = f'trader_{number:03d}'
        self.workspace = root / 'traders' / self.id
        self.workspace.mkdir(parents=True, exist_ok=resume)
        self.rng = random.Random(seed + number)
        self.strategy = ['動能', '均值回歸', '突破', '隨機基準'][number % 4]
        self.window = self.rng.randint(6, 36)
        self.size = self.rng.randint(1, 5)
        self.account = Account()
        if resume:
            profile = json.loads((self.workspace / 'profile.json').read_text())
            self.strategy, self.window, self.size = profile['strategy'], profile['window'], profile['size']
            return
        (self.workspace / 'profile.json').write_text(json.dumps({
            'id': self.id, 'strategy': self.strategy, 'window': self.window,
            'size': self.size, 'seed': seed + number, 'engine': engine,
            'provider': provider, 'model': model, 'internet_enabled': engine == 'llm'}, ensure_ascii=False, indent=2))

    def log(self, event):
        with (self.workspace / 'session.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    def decide(self, history, forum):
        closes = [b.close for b in history[-self.window:]]
        target = 0
        if len(closes) >= self.window:
            difference = closes[-1] - sum(closes) / len(closes)
            if self.strategy == '均值回歸':
                difference *= -1
            elif self.strategy == '突破':
                difference = 1 if closes[-1] > max(closes[:-1]) else -1 if closes[-1] < min(closes[:-1]) else 0
            elif self.strategy == '隨機基準':
                difference = self.rng.choice([-1, 0, 1])
            target = self.size * (1 if difference > 0 else -1 if difference < 0 else 0)
        read = self.rng.random() < .5
        post = None
        if self.rng.random() < .12:
            post = {'body': f'{self.strategy}：本輪目標倉位 {target} 口。',
                    'parent_id': self.rng.choice(forum)['id'] if read and forum and self.rng.random() < .5 else None}
        return {'target': target, 'strategy': self.strategy,
                'reason': f'視窗 {self.window}；僅使用已完成 K 棒', 'read_forum': read, 'post': post}


def external_decision(command, trader, context):
    # 由使用者明確指定的可信任程式；不是安全沙箱。
    result = subprocess.run(shlex.split(command), input=json.dumps(context, ensure_ascii=False),
                            text=True, capture_output=True, cwd=trader.workspace, timeout=120)
    trader.log({'type': 'external_output', 'stdout': result.stdout, 'stderr': result.stderr,
                'returncode': result.returncode})
    if result.returncode:
        raise ValueError(f'外部決策失敗：exit {result.returncode}')
    return json.loads(result.stdout)


def validate_decision(d, limit, forum):
    if not isinstance(d, dict):
        raise ValueError('決策必須是 JSON object')
    if type(d.get('target')) is not int or abs(d['target']) > limit:
        raise ValueError('目標倉位必須為限制範圍內整數')
    if not isinstance(d.get('strategy'), str) or not d['strategy'].strip():
        raise ValueError('必須記錄策略')
    if type(d.get('read_forum', False)) is not bool:
        raise ValueError('read_forum 必須為布林值')
    post = d.get('post')
    if post is not None:
        if not isinstance(post, dict) or not isinstance(post.get('body'), str) or not 0 < len(post['body']) <= 2000:
            raise ValueError('貼文需為 1～2000 字')
        parent = post.get('parent_id')
        if parent is not None and (not d.get('read_forum') or parent not in {p['id'] for p in forum}):
            raise ValueError('回覆必須指向已讀討論區中的貼文')
    return d


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args)


def resume_run(request):
    root = Path(request.output).resolve()
    if request.additional_rounds < 1:
        raise ValueError('additional-rounds 必須為正')
    with (root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((root / 'config.json').read_text())
        if config.get('engine') != 'llm' and not config.get('agent_command'):
            raise ValueError('目前僅支援 LLM 或自訂外部決策的已完成批次續跑')
        with connect(root / 'simulation.sqlite') as db:
            last = db.execute('SELECT MAX(round) FROM snapshots').fetchone()[0]
            count = db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0]
            summary = json.loads((root / 'summary.json').read_text())
            if (not last or count != last * config['traders'] or len(summary) != config['traders']
                    or any(row['round'] != last for row in summary)
                    or db.execute('SELECT COUNT(*) FROM decisions WHERE round=?', (last,)).fetchone()[0]):
                raise ValueError('只能續跑已完整完成的批次，不能從部分輪次恢復')
            if config.get('rounds') and last != config['rounds']:
                raise ValueError('原批次未完成，拒絕續跑')
        # 在修改原資料前保存全部帳戶、session 和資料庫。
        backup = root.parent / (root.name + f'-checkpoint-r{last}')
        if backup.exists():
            raise ValueError(f'備份已存在，請先檢查：{backup}')
        args = argparse.Namespace(**config)
        args.output = str(root)
        args.rounds = last + request.additional_rounds
        args._resume_round = last
        # 先驗證歷史行情一致且未來資料足夠，再保存備份並續跑。
        available = load_bars(args.data, args.start, args.end)[getattr(args, 'warmup_bars', 0):]
        if len(available) < args.rounds:
            raise ValueError('行情不足以延續指定輪數')
        with connect(root / 'simulation.sqlite') as db:
            saved = db.execute('SELECT * FROM quotes ORDER BY round').fetchall()
        if len(saved) != last or any(tuple(r)[1:] != tuple(asdict(b).values()) for r,b in zip(saved, available)):
            raise ValueError('原始行情已改變，拒絕續跑')
        shutil.copytree(root, backup, ignore=shutil.ignore_patterns('.run.lock'))
        print(f'原批次完整備份：{backup}', flush=True)
        _run(args)


def _run(args):
    root = Path(args.output).resolve()
    resume_round = getattr(args, '_resume_round', 0)
    bars = load_bars(args.data, args.start, args.end)
    warmup_count = getattr(args, 'warmup_bars', 0)
    warmup = bars[:warmup_count]
    bars = bars[warmup_count:]
    if args.rounds:
        bars = bars[:args.rounds]
    if len(bars) < 2:
        raise ValueError('至少需要兩根完整五分 K')
    if args.traders < 1 or args.max_position < 5:
        raise ValueError('交易員數需大於 0；倉位上限至少 5')
    db = connect(root / 'simulation.sqlite')
    schema = '''
    CREATE TABLE quotes(round INTEGER PRIMARY KEY,time TEXT,open REAL,high REAL,low REAL,close REAL,volume INTEGER);
    CREATE TABLE snapshots(round INTEGER,trader TEXT,position INTEGER,average REAL,cash REAL,realized REAL,unrealized REAL,fees REAL,equity REAL,pnl REAL,strategy TEXT,bankrupt INTEGER,PRIMARY KEY(round,trader));
    CREATE TABLE decisions(round INTEGER,trader TEXT,target INTEGER,strategy TEXT,reason TEXT,read_forum INTEGER,error TEXT);
    CREATE TABLE trades(round INTEGER,trader TEXT,old_position INTEGER,target INTEGER,price REAL,realized REAL,fee REAL,reason TEXT);
    CREATE TABLE posts(id INTEGER PRIMARY KEY,round INTEGER,trader TEXT,parent_id INTEGER,body TEXT);
    CREATE TABLE forum_reads(round INTEGER,trader TEXT,post_ids TEXT);
    '''
    if not resume_round:
        db.executescript(schema)
    (root / 'config.json').write_text(json.dumps({k: v for k, v in vars(args).items() if k != 'func' and not k.startswith('_')}, ensure_ascii=False, indent=2))
    engine = getattr(args, 'engine', 'builtin')
    traders = [Trader(i, root, args.seed, engine, getattr(args, 'provider', None),
                      getattr(args, 'model', None), resume=bool(resume_round)) for i in range(args.traders)]
    pending = {}
    strategies = {t.id: '尚未決策' for t in traders}
    history = list(warmup) + bars[:resume_round]
    if resume_round:
        for t in traders:
            row = dict(db.execute('SELECT * FROM snapshots WHERE round=? AND trader=?', (resume_round, t.id)).fetchone())
            t.account = Account(**{k: row[k] for k in Account.__dataclass_fields__})
            strategies[t.id] = row['strategy']
            t.log({'type': 'resume', 'after_round': resume_round, 'target_round': len(bars),
                   'discussion_prompt': '可向其他人提問、徵求反方觀點、分享策略假設；自願參與'})
    consecutive_errors = 0
    with (root / 'quotes_5min.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(bars[0])))
        writer.writeheader()
        writer.writerows(asdict(b) for b in bars)

    def execute(t, target, price, reason, round_no):
        old = t.account.position
        if old == target:
            return
        pnl, fee = t.account.trade(target, price)
        db.execute('INSERT INTO trades VALUES(?,?,?,?,?,?,?,?)',
                   (round_no, t.id, old, target, price, pnl, fee, reason))
        t.log({'type': 'fill', 'round': round_no, 'old': old, 'target': target,
               'price': price, 'realized': pnl, 'fee': fee, 'reason': reason})

    def risk(t, price, round_no):
        if t.account.equity(price) <= 0:
            execute(t, 0, price, '破產強制平倉', round_no)
            t.account.bankrupt = True
            pending.pop(t.id, None)

    def record_round(bar, round_no):
        db.execute('INSERT INTO quotes VALUES(?,?,?,?,?,?,?)', (round_no, *asdict(bar).values()))
        history.append(bar)
        for t in traders:
            risk(t, bar.open, round_no)
            if not t.account.bankrupt and t.id in pending:
                execute(t, pending.pop(t.id), bar.open, '前輪決策', round_no)
                risk(t, bar.open, round_no)
            risk(t, bar.close, round_no)
            a = t.account
            equity = a.equity(bar.close)
            snapshot = {'round': round_no, 'trader': t.id, 'position': a.position,
                        'average': a.average, 'cash': a.cash, 'realized': a.realized,
                        'unrealized': equity - a.cash, 'fees': a.fees, 'equity': equity,
                        'pnl': equity - 1_000_000, 'strategy': strategies[t.id], 'bankrupt': int(a.bankrupt)}
            db.execute('INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', tuple(snapshot.values()))
            t.log({'type': 'round', 'quote': asdict(bar), 'account': snapshot})
            (t.workspace / 'account.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        db.commit()

    for round_no, bar in enumerate(bars, 1):
        if resume_round and round_no < resume_round:
            continue
        if round_no != resume_round:
            record_round(bar, round_no)
        # 已完成批次最後一輪原本沒有決策；在同一收盤補決策，下一輪才成交。
        # 所有人只看同一份前輪討論區，避免執行先後順序優勢。
        forum = [dict(row) for row in db.execute('SELECT * FROM posts ORDER BY id DESC LIMIT 100')][::-1]
        for t in traders:
            if t.account.bankrupt or round_no == len(bars):
                continue
            error = None
            try:
                if args.agent_command or engine == 'llm':
                    context = {'round': round_no, 'trader': t.id, 'session_id': root.name + ':' + t.id,
                               'account': asdict(t.account), 'equity': t.account.equity(bar.close),
                               'history': [asdict(b) for b in history[-args.history_window:]],
                               'forum': forum, 'max_position': args.max_position,
                               'objective': '最大化扣除全部手續費後的最終帳戶權益',
                               'discussion_guidance': '可向其他人提問、徵求反方觀點、分享策略假設；可以先發言，也可以不參與。',
                               'initial_style': t.strategy, 'personality_seed': args.seed + int(t.id.split('_')[1]),
                               'instructions': '請以中文決策，只能使用截至本輪的資訊。輸出單一 JSON。'}
                    t.log({'type': 'agent_input', 'context': context})
                    d = (external_decision(args.agent_command, t, context) if args.agent_command
                         else llm_agent.decide(t, context, args))
                else:
                    d = t.decide(history, forum)
                d = validate_decision(d, args.max_position, forum)
            except (ValueError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
                d = {'target': t.account.position, 'strategy': strategies[t.id],
                     'reason': '決策失敗，維持原倉位', 'read_forum': False}
            strategies[t.id] = d['strategy']
            pending[t.id] = d['target']
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?,?)',
                       (round_no, t.id, d['target'], d['strategy'], str(d.get('reason', '')), bool(d.get('read_forum')), error))
            if d.get('read_forum'):
                db.execute('INSERT INTO forum_reads VALUES(?,?,?)', (round_no, t.id, json.dumps([p['id'] for p in forum])))
            if d.get('post'):
                post = d['post']
                db.execute('INSERT INTO posts(round,trader,parent_id,body) VALUES(?,?,?,?)',
                           (round_no, t.id, post.get('parent_id'), post['body']))
            t.log({'type': 'decision', 'round': round_no, 'decision': d, 'error': error})
            db.commit()
            consecutive_errors = consecutive_errors + 1 if error else 0
            print(f'第 {round_no}/{len(bars)} 輪 {t.id}：目標 {d["target"]}；' +
                  ('錯誤 ' + error if error else d['strategy']), flush=True)
            if engine == 'llm' and consecutive_errors >= 3:
                db.close()
                report.generate(root)
                raise RuntimeError('連續三次 LLM 決策失敗，停止批次以避免無效消耗；請檢查事件日誌。')
        db.commit()
    summary = [dict(r) for r in db.execute('SELECT * FROM snapshots WHERE round=? ORDER BY equity DESC', (len(bars),))]
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    db.close()
    report.generate(root)
    print(f'完成：{len(traders)} 位交易員 × {len(bars)} 輪；輸出 {root}')


PAGE = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>交易模擬討論區</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd}article{padding:12px;border:1px solid #ddd;margin:8px}</style>
<h1>交易模擬系統</h1><p><a href="/books/">五卷教材 HTML／PDF 總索引</a> ｜ <a href="/report">全體總損益與台指期趨勢報表</a> ｜ <a href="/discussion">黃金議題：3人AI討論示範</a> ｜ <a href="/taiwan-discussion">40人：在台灣如何賺錢</a> ｜ <a href="/book">Volume I 線性代數</a> ｜ <a href="/book2">Volume II 電腦圖學</a> ｜ <a href="/book3">Volume III 場模擬</a> ｜ <a href="/book4">Volume IV 微積分與分析</a> ｜ <a href="/book5">Volume V 神經網路與Transformer</a></p><iframe title="績效報表" src="/report" style="width:100%;height:850px;border:0"></iframe><p>每 5 秒更新；討論區由交易員決策介面發文／回覆。</p>
<h2>最新倉位與損益</h2><table id="accounts"></table><h2>討論區（最新 100 則）</h2><section id="forum"></section>
<script>
async function refresh(){const r=await fetch('/api/state');const s=await r.json();
const table=document.getElementById('accounts');table.replaceChildren();
const head=document.createElement('tr');for(const text of ['交易員','輪次','倉位','權益','損益','策略','破產']){const th=document.createElement('th');th.textContent=text;head.append(th)}table.append(head);
for(const a of s.accounts){const row=document.createElement('tr');for(const k of ['trader','round','position','equity','pnl','strategy','bankrupt']){const cell=document.createElement('td');cell.textContent=a[k];row.append(cell)}table.append(row)}
const forum=document.getElementById('forum');forum.replaceChildren();for(const p of s.posts){const e=document.createElement('article');e.textContent=`#${p.id} 第${p.round}輪 ${p.trader} ${p.parent_id?'回覆 #'+p.parent_id:''}：${p.body}`;forum.append(e)}}
refresh();setInterval(refresh,5000);
setInterval(()=>{document.querySelector('iframe').src='/report';},30000);
</script></html>'''


def serve(args):
    path = Path(args.output).resolve() / 'simulation.sqlite'
    if not path.is_file():
        raise ValueError('找不到 simulation.sqlite')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            book_root = Path(__file__).resolve().parent / 'books/linear-algebra-aquaculture'
            book_files = {'/book': ('index.html', 'text/html; charset=utf-8'),
                          '/book/html': ('published/volume-1.html', 'text/html; charset=utf-8'),
                          '/book/volume-1.html': ('published/volume-1.html', 'text/html; charset=utf-8'),
                          '/book/download/html': ('published/volume-1.html', 'text/html; charset=utf-8'),
                          '/book/pdf': ('published/volume-1.pdf', 'application/pdf'),
                          '/book/volume-1.pdf': ('published/volume-1.pdf', 'application/pdf'),
                          '/book.md': ('book.md', 'text/plain; charset=utf-8'),
                          '/book/toc': ('TOC.md', 'text/plain; charset=utf-8')}
            book_files.update({f'/book/chapter/{n:02d}': (f'chapters/{n:02d}.md', 'text/plain; charset=utf-8') for n in range(1, 21)})
            book_files.update({f'/book/figures/{name}': (f'figures/{name}', 'image/svg+xml') for name in
                               ('roadmap.svg','shapes.svg','attention.svg','fusion.svg','agent-safety.svg')})
            if self.path in ('/book2','/book2.md') or self.path.startswith('/book2/'):
                book_root = Path(__file__).resolve().parent / 'books/modern-graphics'
                book_files = {'/book2': ('index.html','text/html; charset=utf-8'),
                              '/book2.md': ('book.md','text/plain; charset=utf-8'),
                              '/book2/html': ('published/volume-2.html','text/html; charset=utf-8'),
                              '/book2/volume-2.html': ('published/volume-2.html','text/html; charset=utf-8'),
                              '/book2/download/html': ('published/volume-2.html','text/html; charset=utf-8'),
                              '/book2/pdf': ('published/volume-2.pdf','application/pdf'),
                              '/book2/volume-2.pdf': ('published/volume-2.pdf','application/pdf'),
                              '/book2/toc': ('TOC.md','text/plain; charset=utf-8'),
                              '/book2/status': ('status.json','application/json; charset=utf-8'),
                              '/book2/notifications': ('notifications.html','text/html; charset=utf-8'),
                              '/book2/notifications.json': ('notifications.json','application/json; charset=utf-8'),
                              '/book2/examples': ('examples/graphics_lab.py','text/plain; charset=utf-8')}
                book_files.update({f'/book2/chapter/{n:02d}':(f'chapters/{n:02d}.md','text/plain; charset=utf-8') for n in range(1,31)})
                book_files.update({f'/book2/figures/{name}':(f'figures/{name}','image/svg+xml') for name in ('pipeline.svg','coordinates.svg','surface.svg','light.svg','animation.svg')})
            if self.path in ('/book3','/book3.md','/book4','/book4.md','/book5','/book5.md') or self.path.startswith(('/book3/','/book4/','/book5/')):
                base='/'+self.path.split('/')[1].removesuffix('.md')
                folder,example,figures={
                    '/book3':('field-simulation','field_lab.py',('pipeline.svg','operators.svg','conservation.svg','solvers.svg','phase-field.svg')),
                    '/book4':('calculus-analysis','analysis_lab.py',('pipeline.svg','derivative.svg','chain.svg','geometry.svg','conditions.svg')),
                    '/book5':('neural-transformers','neural_lab.py',('pipeline.svg','shapes.svg','backprop.svg','mask.svg','evidence.svg'))}[base]
                book_root = Path(__file__).resolve().parent / 'books' / folder
                book_files = {base:('index.html','text/html; charset=utf-8'),
                              base+'.md':('book.md','text/plain; charset=utf-8'),
                              base+'/toc':('TOC.md','text/plain; charset=utf-8'),
                              base+'/status':('status.json','application/json; charset=utf-8'),
                              base+'/notifications':('notifications.html','text/html; charset=utf-8'),
                              base+'/notifications.json':('notifications.json','application/json; charset=utf-8'),
                              base+'/examples':('examples/'+example,'text/plain; charset=utf-8')}
                volume=int(base[-1])
                for suffix in ('html','pdf'):
                    mime='text/html; charset=utf-8' if suffix=='html' else 'application/pdf'
                    for route in (f'{base}/{suffix}',f'{base}/volume-{volume}.{suffix}'):
                        book_files[route]=(f'published/volume-{volume}.{suffix}',mime)
                book_files[base+'/download/html']=(f'published/volume-{volume}.html','text/html; charset=utf-8')
                book_files.update({f'{base}/chapter/{n:02d}':(f'chapters/{n:02d}.md','text/plain; charset=utf-8') for n in range(1,31)})
                book_files.update({f'{base}/figures/{name}':(f'figures/{name}','image/svg+xml') for name in figures})
            if self.path == '/books':
                self.send_response(302)
                self.send_header('Location','/books/')
                self.send_header('Content-Length','0')
                self.end_headers()
                return
            if self.path.startswith('/books/'):
                from publishing.catalog import BOOKS
                book_root = Path(__file__).resolve().parent / 'books'
                book_files = {'/books/':('index.html','text/html; charset=utf-8'),
                              '/books/index.html':('index.html','text/html; charset=utf-8'),
                              '/books/publication-status.json':('publication-status.json','application/json; charset=utf-8')}
                for v,folder in BOOKS.items():
                    for suffix,mime in (('html','text/html; charset=utf-8'),('pdf','application/pdf')):
                        relative=f'{folder}/published/volume-{v}.{suffix}'
                        book_files['/books/'+relative]=(relative,mime)
            if self.path in book_files:
                relative, mime = book_files[self.path]
                file = book_root / relative
                if not file.is_file():
                    self.send_error(404, 'Chapter not ready')
                    return
                content = file.read_bytes()
                if self.path == '/book2':
                    from book2_notifications import inject_panel
                    from book2_publication import inject_publication
                    content = inject_panel(inject_publication(content.decode('utf-8'),book_root)).encode('utf-8')
                elif self.path in ('/book3','/book4','/book5'):
                    from book2_notifications import inject_panel
                    from book2_publication import inject_publication
                    content = inject_panel(inject_publication(content.decode('utf-8'),book_root,int(self.path[-1])),self.path,{'/book3':'Volume III','/book4':'Volume IV','/book5':'Volume V'}[self.path]).encode('utf-8')
            elif self.path == '/':
                content, mime = PAGE.encode(), 'text/html; charset=utf-8'
            elif self.path in ('/discussion', '/taiwan-discussion'):
                relative = ('runs/gold-discussion/index.html' if self.path == '/discussion'
                            else 'runs/taiwan-money-40/index.html')
                demo = Path(__file__).resolve().parent / relative
                if not demo.is_file():
                    self.send_error(404, 'Discussion demo not generated')
                    return
                content, mime = demo.read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/report':
                content, mime = report.render(path.parent).encode(), 'text/html; charset=utf-8'
            elif self.path == '/api/state':
                with connect(path) as db:
                    accounts = [dict(r) for r in db.execute('SELECT * FROM snapshots WHERE round=(SELECT MAX(round) FROM snapshots) ORDER BY equity DESC')]
                    posts = [dict(r) for r in db.execute('SELECT * FROM posts ORDER BY id DESC LIMIT 100')]
                content, mime = json.dumps({'accounts': accounts, 'posts': posts}, ensure_ascii=False).encode(), 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            if self.path.startswith('/books/') or (self.path in book_files and book_files[self.path][0].startswith('published/')):
                self.send_header('Cache-Control', 'no-store')
            if self.path in ('/book2', '/book2/status', '/book2/notifications', '/book2/notifications.json', '/book3', '/book3/status', '/book3/notifications', '/book3/notifications.json', '/book4', '/book4/status', '/book4/notifications', '/book4/notifications.json', '/book5', '/book5/status', '/book5/notifications', '/book5/notifications.json'):
                self.send_header('Cache-Control', 'no-store')
            if self.path in ('/book2/notifications','/book3/notifications','/book4/notifications','/book5/notifications'):
                self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'self'; form-action 'none'")
            if self.path in ('/book/download/html', '/book2/download/html', '/book3/download/html', '/book4/download/html', '/book5/download/html'):
                v=self.path.split('/')[1].removeprefix('book') or '1'
                name=f'volume-{v}.html'
                self.send_header('Content-Disposition', f'attachment; filename="{name}"')
            if self.path in ('/book/html', '/book/volume-1.html', '/book/download/html', '/book2/html', '/book2/volume-2.html', '/book2/download/html', '/book3/html', '/book3/volume-3.html', '/book3/download/html', '/book4/html', '/book4/volume-4.html', '/book4/download/html', '/book5/html', '/book5/volume-5.html', '/book5/download/html'):
                self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; font-src data:; img-src data:; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)
    print(f'開啟 http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description='80 人期貨模擬交易系統')
    commands = parser.add_subparsers(dest='action', required=True)
    sim = commands.add_parser('run')
    sim.add_argument('--data', nargs='+', default=['data/txfr1_sql/data_TXFR1_2025.sql', 'data/txfr1_sql/data_TXFR1_2026.sql'])
    sim.add_argument('--output', required=True)
    sim.add_argument('--start')
    sim.add_argument('--end')
    sim.add_argument('--rounds', type=int, default=100)
    sim.add_argument('--warmup-bars', type=int, default=0, help='先提供已發生歷史，不交易；再開始正式輪次')
    sim.add_argument('--traders', type=int, default=80)
    sim.add_argument('--seed', type=int, default=42)
    sim.add_argument('--max-position', type=int, default=10)
    sim.add_argument('--history-window', type=int, default=300)
    sim.add_argument('--engine', choices=['llm', 'builtin'], default='llm', help='預設真實 LLM；builtin 為離線程式策略')
    sim.add_argument('--provider', default=llm_agent.DEFAULT_PROVIDER)
    sim.add_argument('--model', default=llm_agent.DEFAULT_MODEL)
    sim.add_argument('--pi-bin', default='pi')
    sim.add_argument('--agent-timeout', type=int, default=180)
    sim.add_argument('--agent-command', help='可信任外部決策程式（建議絕對路徑），JSON stdin/stdout')
    sim.set_defaults(func=run)
    continuation = commands.add_parser('resume')
    continuation.add_argument('--output', required=True)
    continuation.add_argument('--additional-rounds', type=int, default=100)
    continuation.set_defaults(func=resume_run)
    reports = commands.add_parser('report')
    reports.add_argument('--output', required=True)
    reports.set_defaults(func=lambda args: report.generate(args.output))
    web = commands.add_parser('serve')
    web.add_argument('--output', required=True)
    web.add_argument('--port', type=int, default=8080)
    web.set_defaults(func=serve)
    args = parser.parse_args()
    if getattr(args, 'warmup_bars', 0) < 0:
        parser.error('warmup-bars 不可為負')
    if getattr(args, 'rounds', 0) < 0 or getattr(args, 'history_window', 1) < 1:
        parser.error('rounds 不可為負；history-window 必須為正')
    args.func(args)


if __name__ == '__main__':
    main()
