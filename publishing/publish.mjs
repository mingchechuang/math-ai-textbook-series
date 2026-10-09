// Offline HTML + Chromium PDF; never executes manuscript Python.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';
import {chromium} from 'playwright-core';

const here=path.dirname(fileURLToPath(import.meta.url));
const args=process.argv.slice(2);
if(args.length && !(args.length===2 && args[0]==='--volume' && ['1','2','3','4','5'].includes(args[1])))throw new Error('Usage: node publishing/publish.mjs [--volume 1|2|3|4|5]');
const volume=args.length?Number(args[1]):1;
const roman=['','I','II','III','IV','V'][volume], chapterCount=volume===1?20:30;
const stem=`volume-${volume}`;
const root=path.resolve(here,'../books/'+({1:'linear-algebra-aquaculture',2:'modern-graphics',3:'field-simulation',4:'calculus-analysis',5:'neural-transformers'}[volume]));
const destination=path.join(root,'published');fs.mkdirSync(destination,{recursive:true});
const out=fs.mkdtempSync(path.join(root,'.publish-'));
const state=JSON.parse(fs.readFileSync(path.join(root,'status.json'),'utf8'));
const config=JSON.parse(fs.readFileSync(path.join(root,'config.json'),'utf8'));
if(state.status!=='completed_model_review_pending_human')throw new Error('Model review must be complete before publication');
const title=volume===1?'Volume I｜從生成式 AI 到智慧水產養殖':volume===2?'Volume II｜現代電腦圖學':`Volume ${roman}｜${config.title}`;
const subtitle={1:'線性代數 × 生成式 AI × 多模態 × Agent × 水產養殖',2:'幾何與成像 × 建模與材質 × 光線追蹤 × 動畫 × 養殖數位分身',3:'場論 × 偏微分方程 × 守恆離散 × 流體 × 相場',4:'極限 × 線性映射 × 最佳化 × 積分 × 微分方程與變分',5:'張量與梯度 × 神經網路 × 注意力 × Transformer × 唯讀Agent'}[volume];
const entries=[['preface','00-preface.md'],...Array.from({length:chapterCount},(_,i)=>[`ch${String(i+1).padStart(2,'0')}`,`chapters/${String(i+1).padStart(2,'0')}.md`]),['appendix','99-appendix.md'],['references','REFERENCES.md']];
const source=entries.map(([id,file])=>({id,file,text:fs.readFileSync(path.join(root,file),'utf8')}));
const hash=crypto.createHash('sha256').update(JSON.stringify({source,status:state.status,word_count:state.word_count})).digest('hex');
const figures=fs.readdirSync(path.join(root,'figures')).filter(x=>x.endsWith('.svg'));
const manuscriptHash=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'book.md'))).digest('hex');
const sourceFiles=Object.fromEntries([...entries.map(x=>x[1]),...figures.map(x=>'figures/'+x)].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,file))).digest('hex')]));
const ui='全文閱讀 章節導覽 下載PDF 列印 搜尋請用瀏覽器 全部章節 模型審稿完成 待人工審定 前言 附錄 參考資料 圖文版 使用方法 字型授權';
fs.writeFileSync(path.join(out,'font-characters.txt'),source.map(x=>x.text).join('\n')+ui+title+fs.readFileSync(fileURLToPath(import.meta.url),'utf8')+figures.map(x=>fs.readFileSync(path.join(root,'figures',x),'utf8')).join(''));
execFileSync(path.join(here,'.venv/bin/python'),['-m','fontTools.subset',path.join(here,'fonts/NotoSansCJKtc-Regular.otf'),`--text-file=${path.join(out,'font-characters.txt')}`,'--flavor=woff2',`--output-file=${path.join(out,'book-cjk.woff2')}`],{stdio:'inherit'});
const data=(file,mime)=>`data:${mime};base64,${fs.readFileSync(file).toString('base64')}`;
let mathCss=fs.readFileSync(path.join(here,'node_modules/katex/dist/katex.min.css'),'utf8');
mathCss=mathCss.replace(/url\(([^)]+)\)/g,(_,url)=>{
  const name=url.replace(/["']/g,'');const file=path.join(here,'node_modules/katex/dist',name);
  return `url(${data(file,name.endsWith('woff2')?'font/woff2':name.endsWith('woff')?'font/woff':'font/ttf')})`;
});
const errors=[];let mathCount=0;let current='';
const engine={renderToString(tex,options){
  mathCount++;
  // Chapter 4 uses Python's @ operator inside a formula. Typeset it literally.
  tex=tex.replaceAll('\\@','\\mathbin{@}');
  if(volume===5){
   const original=tex;
   if(current==='ch11')tex=tex.replaceAll(String.raw`g^\*`,String.raw`g^{*}`);
   if(current==='ch14')tex=tex.replaceAll('G_A_{','{G_A}_{');
   if(current==='ch26')tex=tex.replaceAll(String.raw`\logp_0`,String.raw`\log p_0`).replaceAll(String.raw`\logq_0`,String.raw`\log q_0`);
   if(tex!==original)rendererNormalizations.push({chapter:current,rule:'Repair TeX spelling/grouping only: escaped star, nested subscript, log spacing',original,rendered:tex});
  }
  try{return katex.renderToString(tex,{...options,throwOnError:true,strict:'ignore',trust:false,output:'htmlAndMathml'});}
  catch(e){errors.push({chapter:current,tex,error:e.message});return '<span class="math-error">公式轉換失敗</span>';}
}};
const mathDelimiters=volume>=2?['dollars','brackets']:['dollars'];
// Unique rule names let both delimiter families interrupt paragraphs correctly.
for(const family of mathDelimiters){
 for(const kind of ['inline','block'])texmath.rules[family][kind]=texmath.rules[family][kind].map(rule=>({...rule,name:`${family}_${rule.name}`}));
}
const md=new MarkdownIt({html:false,linkify:true,typographer:false}).use(texmath,{engine,delimiters:mathDelimiters});
// Block equations must interrupt paragraphs, or '=' lines become setext headings.
for(const family of mathDelimiters)for(const rule of texmath.rules[family].block)md.block.ruler.at(rule.name,texmath.block(rule),{alt:['paragraph','blockquote','list']});
const esc=s=>md.utils.escapeHtml(s);
md.renderer.rules.image=(tokens,index)=>{
 const token=tokens[index];const name=path.basename(token.attrGet('src'));
 if(!figures.includes(name))throw new Error(`Unknown figure: ${name}`);
 let svg=fs.readFileSync(path.join(root,'figures',name),'utf8');
 if(/<script|<foreignObject|\bon\w+\s*=|(?:href|src)\s*=/i.test(svg))throw new Error(`Unsafe SVG: ${name}`);
 return `<span class="figure" role="group">${svg}<span class="caption">${md.renderInline(token.content)}</span></span>`;
};
const defaultLink=md.renderer.rules.link_open||((tokens,i,options,env,self)=>self.renderToken(tokens,i,options));
md.renderer.rules.link_open=(tokens,i,options,env,self)=>{
 const token=tokens[i];const href=token.attrGet('href')||'';
 const found=source.find(x=>href===x.file||href===`../${x.file}`);
 if(found)token.attrSet('href',`#${found.id}`);
 if(/^https?:/.test(href))token.attrSet('rel','noopener noreferrer');
 return defaultLink(tokens,i,options,env,self);
};
const rendererNormalizations=[];
// Later volumes contain spaced and wrapped inline TeX. Normalize only inline
// Markdown tokens, excluding code spans; fenced code is never an inline token.
if(volume>=3)md.core.ruler.before('inline','publication_math_whitespace',state=>{
 for(const token of state.tokens){
  if(token.type!=='inline')continue;
  token.content=token.content.split(/(`+[^`]*`+)/g).map(part=>part.startsWith('`')?part:part.replace(/(?<![\\$])\$(?!\$)([^$]+?)\$(?!\$)/g,(whole,body)=>{
   const normalized='$'+body.trim().replace(/\n\s*/g,' ')+'$';
   if(normalized!==whole)rendererNormalizations.push({chapter:current,rule:'Normalize inline TeX whitespace',original:whole,rendered:normalized});
   return normalized;
  })).join('');
 }
});
let headingIndex=0;
md.renderer.rules.heading_open=(tokens,i,options,env,self)=>{
 tokens[i].attrSet('id',`${current}-heading-${++headingIndex}`);
 return self.renderToken(tokens,i,options);
};
const chapters=source.map(item=>{
 current=item.id;headingIndex=0;
 let renderText=item.text;
 if(volume===5 && item.id==='ch11' && renderText.includes(String.raw`\tag{$\ast$}`)){
  renderText=renderText.replaceAll(String.raw`\tag{$\ast$}`,String.raw`\tag{*}`);
  rendererNormalizations.push({chapter:item.id,rule:'Render star equation label without nested dollar delimiters',original:String.raw`\tag{$\ast$}`,rendered:String.raw`\tag{*}`});
 }
 if(volume===3 && item.id==='ch26' && renderText.includes('\boldsymbol J')){
  renderText=renderText.replaceAll('\boldsymbol J',String.raw`\boldsymbol J`);
  rendererNormalizations.push({chapter:item.id,rule:'Restore backspace-corrupted boldsymbol command; same expression intact in chapter summary'});
 }
 if(volume===3 && item.id==='ch19'){
  const broken='$u_y[j_{\\rm face},i]`';
  if(renderText.includes(broken)){
   renderText=renderText.replace(broken,'$u_y[j_{\\rm face},i]$');
   rendererNormalizations.push({chapter:item.id,rule:'Close MAC index inline formula with dollar rather than stray backtick'});
  }
 }
 if(volume===4 && item.id==='ch26'){
  renderText=renderText.replace("26.1$'$",'26.1′');
  rendererNormalizations.push({chapter:item.id,rule:'Render definition-label prime as Unicode prime (dollar after digit rejected by parser)'});
 }
 // A wrapped inline formula in Volume II ch03: join whitespace only; keep manuscript intact.
 if(volume===2 && item.id==='ch03'){
  const wrapped=String.raw`$(M_{\mathrm{parent}}M_{\mathrm{local}})^{-1}
   =M_{\mathrm{local}}^{-1}M_{\mathrm{parent}}^{-1}$`;
  if(renderText.includes(wrapped)){
   renderText=renderText.replace(wrapped,wrapped.replace(/\n\s*/g,' '));
   rendererNormalizations.push({chapter:item.id,rule:'Join wrapped inline formula whitespace'});
  }
 }
 const rendered=md.render(renderText);
 return `<section class="chapter" id="${item.id}">${rendered}</section>`;
}).join('\n');
fs.writeFileSync(path.join(out,'math-errors.json'),JSON.stringify(errors,null,2));
if(errors.length)throw new Error(`${errors.length} invalid math expressions; see math-errors.json`);
const toc=source.map(item=>`<li><a href="#${item.id}">${esc(item.text.match(/^#\s+(.+)$/m)?.[1]||item.id)}</a></li>`).join('');
const css=`
@font-face{font-family:BookCJK;src:url(${data(path.join(out,'book-cjk.woff2'),'font/woff2')}) format('woff2');font-display:block}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#182d40;background:#f0f4f8;font-family:BookCJK,sans-serif;line-height:1.85}
a{color:#14579a;text-decoration:none;overflow-wrap:anywhere}a:hover{text-decoration:underline}.sidebar{position:fixed;inset:0 auto 0 0;width:280px;background:#e7eef5;padding:24px 18px;overflow:auto;font-size:13px}.sidebar ol{padding-left:22px}.sidebar li{margin:10px 0}.document{margin:0 28px 0 304px;padding:48px 56px;background:white;max-width:1100px;min-height:100vh}.cover h1{font-size:34px;line-height:1.45}.cover{padding:30px 0 50px;border-bottom:3px solid #1a6385}.badge{color:#53677b}.tools{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.tools a{border:1px solid #aac2d6;padding:7px 12px;border-radius:6px;background:white}.chapter{padding-top:32px;margin-top:32px;border-top:1px solid #ccd9e4;scroll-margin-top:20px}.chapter h1{font-size:28px;line-height:1.5;color:#124d70}.chapter h2{font-size:21px;margin-top:1.7em;color:#1e5b7c}.chapter h3{font-size:18px}p{overflow-wrap:break-word}blockquote{margin:1em 0;padding:10px 18px;border-left:4px solid #7caac5;background:#f4f8fc}pre{font:13px/1.6 'DejaVu Sans Mono',BookCJK,monospace;padding:16px;background:#f2f5f8;border:1px solid #dce4eb;border-radius:4px;white-space:pre-wrap;overflow-wrap:anywhere;tab-size:4}code{font-family:'DejaVu Sans Mono',BookCJK,monospace;font-size:.9em}p code,li code{background:#eef3f7;padding:0 3px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd9e4;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#eaf2f8}.figure{display:block;text-align:center;margin:24px auto;break-inside:avoid}.figure svg{display:block;width:100%;height:auto;max-height:600px}svg text{font-family:BookCJK,sans-serif!important}.caption{display:block;font-size:13px;color:#52677b}.katex{font-size:1.08em}.katex .cjk_fallback{font-family:BookCJK,sans-serif}.katex-display{padding:8px 0;overflow-x:auto;overflow-y:hidden;max-width:100%}.katex-display>.katex{white-space:nowrap}eq,eqn{max-width:100%}eqn{display:block}.print-toc{display:none}.licenses{font-size:12px;color:#52677b;overflow-wrap:anywhere}
@media screen and (max-width:1000px){.document{overflow-x:clip}eq{display:inline-block;max-width:100%;overflow-x:auto;overflow-y:hidden;vertical-align:middle;padding:3px 0}.sidebar{position:static;width:auto;max-height:300px}.document{margin:0;padding:24px;max-width:none}.cover h1{font-size:27px}.chapter h1{font-size:24px}}
@page{size:A4;margin:18mm 18mm 20mm}
@media print{html{scroll-behavior:auto}body{background:white;color:#111;font-size:10.5pt;line-height:1.7}.sidebar,.tools,.screen-only{display:none!important}.document{width:174mm;max-width:none;margin:0;padding:0;min-height:0}.cover{padding:28mm 0 10mm;border:0;break-after:page}.cover h1{font-size:28pt}.cover .figure svg{max-height:140mm}.print-toc{display:block;break-after:page}.print-toc li{margin:3mm 0}.chapter{break-before:page;margin:0;padding:0;border:0}.chapter h1{font-size:20pt}.chapter h2{font-size:14pt}h1,h2,h3{break-after:avoid}p{orphans:3;widows:3}pre{font-size:8pt;line-height:1.55;padding:9px;border-radius:0;break-inside:auto}table{font-size:9pt}tr,figure,.figure,.katex-display{break-inside:avoid}.figure svg{max-height:160mm}.katex-display{overflow:visible}.katex-display>.katex{white-space:nowrap}a{color:inherit;text-decoration:none}.licenses{font-size:8pt}}
`;
const roadmap=fs.readFileSync(path.join(root,'figures',volume===1?'roadmap.svg':'pipeline.svg'),'utf8');
const html=`<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)}</title><meta name="description" content="${esc(title)}：圖文、公式與程式；模型審稿完成，仍待人工審定。"><style>${mathCss}\n${css}</style></head><body>
<aside class="sidebar"><strong>Volume ${roman} · 章節導覽</strong><div class="tools"><a href="${stem}.pdf">下載 PDF</a></div><p>搜尋請用瀏覽器 Ctrl／⌘＋F。</p><ol>${toc}</ol></aside>
<main class="document"><header class="cover"><p class="badge">線性代數與應用教材系列 · Volume ${roman}</p><h1>${esc(config.title)}</h1><p>${esc(subtitle)}</p><p>${chapterCount}章 · ${state.word_count.total.toLocaleString('en-US')} 中文字</p><p><strong>模型審稿完成，仍待人工審定。</strong><br>模型審稿不等於人工數學驗證；所有養殖案例為合成教學資料。</p><p class="badge">內容版本：${esc(state.updated)}<br>HTML／PDF 圖文出版版</p><div class="tools"><a href="${stem}.pdf">下載 PDF</a><a href="#ch01">開始閱讀</a></div><span class="figure">${roadmap}</span></header><nav class="print-toc"><h1>目錄</h1><ol>${toc}</ol></nav>${chapters}<section class="licenses"><h2>出版與字型授權</h2><p>本版由既有Markdown排版，未執行章內程式碼。字型：Noto Sans CJK TC（SIL Open Font License 1.1）；數學排版：KaTeX（MIT）。全文來源SHA-256：${hash}。</p><details><summary>字型與數學排版授權全文</summary><pre>${esc(fs.readFileSync(path.join(here,'fonts/OFL.txt'),'utf8'))}\n${esc(fs.readFileSync(path.join(here,'node_modules/katex/LICENSE'),'utf8'))}</pre></details></section></main></body></html>`;
const htmlPath=path.join(out,`${stem}.html`);fs.writeFileSync(htmlPath,html);
const browser=await chromium.launch({executablePath:process.env.BOOK_CHROMIUM||'/snap/bin/chromium',headless:true,args:['--disable-gpu']});
let inspection;
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 const requests=[];await page.route(/^https?:/,route=>{requests.push(route.request().url());return route.abort();});
 await page.goto(pathToFileURL(htmlPath).href,{waitUntil:'load'});
 await page.evaluate(()=>document.fonts.ready);
 await page.screenshot({path:path.join(out,'preview-cover.png')});
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>({viewport:innerWidth,bodyWidth:document.body.scrollWidth}));
 if(mobile.bodyWidth>mobile.viewport+2)throw new Error('Mobile overflow: '+JSON.stringify(mobile));
 await page.locator('#ch13').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(out,'preview-mobile.png')});
 await page.setViewportSize({width:1440,height:1000});
 await page.emulateMedia({media:'print'});
 // Fit oversized display/inline equations to the printable text column.
 const scaled=await page.evaluate(()=>{
  const changed=[];const width=document.querySelector('.document').getBoundingClientRect().width;
  for(const element of document.querySelectorAll('.katex-display > .katex, eq > .katex')){
   const boxes=[...(element.querySelector('.katex-html')?.children||[])].map(x=>x.getBoundingClientRect());
   const measured=boxes.length?Math.max(...boxes.map(x=>x.right))-Math.min(...boxes.map(x=>x.left)):element.getBoundingClientRect().width;
   if(measured>width-8){const scale=(width-8)/measured;const size=parseFloat(getComputedStyle(element).fontSize);element.style.fontSize=`${size*scale}px`;changed.push({text:element.textContent.slice(0,100),scale});}
  }
  return changed;
 });
 inspection=await page.evaluate(()=>({
  math:document.querySelectorAll('.katex').length,figures:document.querySelectorAll('.figure > svg').length,
  missingLinks:[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href').slice(1)).filter(id=>!document.getElementById(id)),
  chapterCount:document.querySelectorAll('section.chapter[id^="ch"]').length,
  unrenderedDollars:(()=>{const walker=document.createTreeWalker(document.querySelector('main'),NodeFilter.SHOW_TEXT),raw=[];let node;while(node=walker.nextNode())if((node.textContent.includes('$')||/\\(?:frac|mathbf|mathrm|begin)\b/.test(node.textContent))&&!node.parentElement.closest('pre,code,.katex'))raw.push(node.textContent.slice(0,100));return raw;})(),
  cjkFontLoaded:document.fonts.check('16px BookCJK'),
  overflow:[...document.querySelectorAll('pre,table,.katex-display')].filter(e=>e.scrollWidth>e.clientWidth+4).map(e=>({tag:e.tagName,width:e.clientWidth,scroll:e.scrollWidth,text:e.textContent.slice(0,100)}))
 }));
 if(requests.length||inspection.missingLinks.length||inspection.unrenderedDollars.length||inspection.overflow.length||inspection.chapterCount!==chapterCount||!inspection.cjkFontLoaded)throw new Error('Publication validation failed: '+JSON.stringify({requests,inspection}));
 await page.pdf({path:path.join(out,`${stem}.pdf`),format:'A4',printBackground:true,preferCSSPageSize:true,tagged:true,outline:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:`<div style="width:100%;text-align:center;font:9px sans-serif;color:#64748b">Volume ${roman} &nbsp; · &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>`});
 if(manuscriptHash!==crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'book.md'))).digest('hex') || Object.entries(sourceFiles).some(([file,sha])=>sha!==crypto.createHash('sha256').update(fs.readFileSync(path.join(root,file))).digest('hex')))throw new Error('Sources changed during publication; refusing to publish');
 fs.writeFileSync(path.join(out,'publication.json'),JSON.stringify({volume,chapter_count:chapterCount,built_at:new Date().toISOString(),source_sha256:hash,source_files:sourceFiles,renderer_normalizations:rendererNormalizations,manuscript_sha256:manuscriptHash,source_status:state.status,characters:state.word_count.total,mathCount,inspection,mobile,scaled_math:scaled,external_requests:requests,files:{html:fs.statSync(htmlPath).size,pdf:fs.statSync(path.join(out,`${stem}.pdf`)).size},html_sha256:crypto.createHash('sha256').update(fs.readFileSync(htmlPath)).digest('hex'),pdf_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(out,`${stem}.pdf`))).digest('hex')},null,2));
 for(const file of fs.readdirSync(out).filter(x=>x!=='publication.json'))fs.renameSync(path.join(out,file),path.join(destination,file));
 fs.renameSync(path.join(out,'publication.json'),path.join(destination,'publication.json'));
 fs.rmdirSync(out);
 console.log(JSON.stringify({mathCount,inspection,mobile,scaled_math:scaled.length,output:destination},null,2));
}finally{await browser.close();}
