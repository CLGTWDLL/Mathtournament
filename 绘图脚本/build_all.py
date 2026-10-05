"""Read-only data inputs; write figures and captions. Does not run experiments."""
from common import *
import question1,question23,question4,tables,question3_results

def main():
 question1.run();question23.run();question4.run();tables.run();extra=question3_results.run()
 html=OUT/'表格排版预览.html';html.write_text(html.read_text().replace('</html>',extra+'</html>'))
 (OUT/'图表清单.json').write_text(json.dumps({'figures':manifest,'input_sha256':sources},ensure_ascii=False,indent=2))
 for q in range(1,5):
  rows=[r for r in manifest if r['name'].startswith(f'question{q}/')]
  text=f'# 第{q}问图表与独立图注\n\n'
  for r in sorted(rows,key=lambda r:int(r['name'].split('/图')[1].split('_')[0])):
   stem=r['name'].split('/')[1];text+=f'![{stem}]({stem}.png)\n\n**{stem.replace("_"," ")}。** {r["caption"]}\n\n'
  (OUT/f'question{q}/图注.md').write_text(text)
 gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><title>论文图表总览</title><style>body{font-family:serif;color:black;max-width:1050px;margin:40px auto;padding:0 24px}img{display:block;max-width:100%;height:auto;margin:auto}figure{margin:40px 0 70px}figcaption{line-height:1.8;margin-top:16px}h2{font-size:20px}</style><h1>论文图表重构版</h1><p>图片与图注分离；全部文字为黑色。<a href="表格排版预览.html">查看三线表</a></p>'
 for r in sorted(manifest,key=lambda x:(x['name'].split('/')[0],int(x['name'].split('/图')[1].split('_')[0]))):
  gallery+=f'<figure><h2>{r["name"].replace("_"," ")}</h2><a href="{r["name"]}.svg"><img src="{r["name"]}.png"></a><figcaption>{r["caption"]}</figcaption></figure>'
 (OUT/'全部图表预览.html').write_text(gallery+'</html>')
 print(f'Generated {len(manifest)} figures as PNG + SVG')
if __name__=='__main__':main()
