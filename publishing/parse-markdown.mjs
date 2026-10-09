// Parse Markdown structure only. Never executes code from the manuscript.
import MarkdownIt from 'markdown-it';
let input='';for await(const chunk of process.stdin)input+=chunk;
const tokens=new MarkdownIt({html:false}).parse(input,{});
const headings=[],fences=[];
for(let i=0;i<tokens.length;i++){
 const t=tokens[i];
 if(t.type==='heading_open'&&t.map)headings.push({level:Number(t.tag.slice(1)),text:tokens[i+1]?.content||'',map:t.map});
 if(t.type==='fence')fences.push({language:t.info.trim().split(/\s+/)[0].toLowerCase(),content:t.content,map:t.map});
}
process.stdout.write(JSON.stringify({headings,fences}));
