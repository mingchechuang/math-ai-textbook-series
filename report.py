"""無外部 JS/CDN 的可攜式 HTML/SVG 績效報表。"""
import csv
import html
import json
import sqlite3
from pathlib import Path


def collect(root):
    root = Path(root)
    with sqlite3.connect(root / 'simulation.sqlite') as db:
        db.row_factory = sqlite3.Row
        series = [dict(r) for r in db.execute('''
            SELECT s.round,q.time,q.close,SUM(s.pnl) total_pnl,SUM(s.equity) total_equity,
                   SUM(s.fees) fees,COUNT(*) traders
            FROM snapshots s JOIN quotes q ON s.round=q.round
            GROUP BY s.round ORDER BY s.round''')]
        accounts = [dict(r) for r in db.execute('SELECT * FROM snapshots WHERE round=(SELECT MAX(round) FROM snapshots) ORDER BY equity DESC')]
        errors = db.execute('SELECT COUNT(*) FROM decisions WHERE error IS NOT NULL').fetchone()[0]
    config = json.loads((root / 'config.json').read_text())
    return series, accounts, config, errors


def chart(series, field, title, color, zero=False):
    values = [s[field] for s in series]
    low, high = min(values + ([0] if zero else [])), max(values + ([0] if zero else []))
    if low == high:
        low, high = low - 1, high + 1
    pad = (high-low)*.05
    low, high = low-pad, high+pad
    def y(v):
        return 225 - (v-low)/(high-low)*185
    points = ' '.join(f'{100+i/max(1,len(values)-1)*850:.2f},{y(v):.2f}' for i,v in enumerate(values))
    labels = ''.join(f'<text x="90" y="{y(v)+4:.2f}" text-anchor="end">{v:,.0f}</text><line x1="100" x2="950" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#ddd"/>' for v in (low,(low+high)/2,high))
    baseline = f'<line x1="100" x2="950" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="#777" stroke-dasharray="5 3"/>' if zero else ''
    return f'''<h2>{html.escape(title)}</h2><svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 1000 270">
    {labels}{baseline}<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>
    <text x="100" y="255">{html.escape(series[0]['time'])}</text><text x="950" y="255" text-anchor="end">{html.escape(series[-1]['time'])}</text></svg>'''


def render(root):
    series, accounts, config, errors = collect(root)
    if not series:
        return '<meta charset="utf-8"><p>尚無完成輪次。</p>'
    last = series[-1]
    with sqlite3.connect(Path(root) / 'simulation.sqlite') as db:
        decisions = db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
    planned = config.get('rounds', 0)
    progress = f'行情 {last["round"]}/{planned or "全部"} 輪；已記錄 {decisions} 次決策（含失敗）'
    initial = last['traders'] * 1_000_000
    peak = initial
    drawdown = 0
    for row in series:
        peak = max(peak, row['total_equity'])
        drawdown = max(drawdown, peak-row['total_equity'])
    engine = config.get('engine', 'builtin')
    label = (config.get('provider', '') + '/' + config.get('model', '')) if engine == 'llm' else '程式策略（非 LLM）'
    if config.get('agent_command'):
        label = '自訂外部決策程式'
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(a[k]))}</td>' for k in ('trader','position','realized','unrealized','fees','pnl','strategy')) + '</tr>' for a in accounts)
    return f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>全體交易員績效報表</title>
    <style>body{{font:16px sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;color:#223}}svg{{width:100%;background:#fafafa}}svg text{{font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.cards{{background:#eef;padding:20px;line-height:2}}</style>
    <h1>全體交易員績效報表</h1><p>{html.escape(progress)}</p><p>決策模式：{html.escape(label)} ｜ {last['traders']} 人 ｜ {len(series)} 輪</p>
    <div class="cards">總損益（含未實現、已扣手續費）：<b>{last['total_pnl']:,.2f}</b> 元<br>
    總權益：{last['total_equity']:,.2f} 元 ｜ 初始資金：{initial:,.0f} 元<br>
    報酬率：{last['total_pnl']/initial:.4%} ｜ 累計手續費：{last['fees']:,.2f} 元<br>
    全體權益最大回撤：{drawdown:,.2f} 元 ｜ 決策錯誤：{errors}</div>
    {chart(series,'total_pnl','所有交易員總損益走勢（元）','#1769aa',True)}
    {chart(series,'close','台指期 TXFR1 近月連續價格趨勢（點；非現貨加權指數）','#d65a18')}
    <p>橫軸為有效回放輪次（交易空檔壓縮），兩圖使用相同時間範圍。價格時間為 K 棒起始標籤；損益為該棒收盤估值。資料為未調整換月價差的台指期，不是現貨指數。網路研究可能引入前視偏誤。</p>
    <h2>所有交易員明細</h2><table><tr><th>交易員</th><th>倉位</th><th>已實現毛損益</th><th>未實現</th><th>費用</th><th>淨損益</th><th>策略</th></tr>{rows}</table></html>'''


def generate(root):
    root = Path(root)
    (root / 'report.html').write_text(render(root), encoding='utf-8')
    series, accounts, _, _ = collect(root)
    for name, rows in [('portfolio_timeseries.csv', series), ('trader_summary.csv', accounts)]:
        if rows:
            with (root / name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f'報表：{root / "report.html"}')
