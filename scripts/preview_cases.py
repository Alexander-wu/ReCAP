"""Build a standalone local HTML gallery from packaged qualitative GIFs."""
import html
from pathlib import Path
root=Path(__file__).resolve().parents[1]
items=[]
for p in sorted((root/'artifacts/cases').glob('*.gif')):
    items.append(f'<figure><img loading="lazy" src="{p.relative_to(root).as_posix()}" alt="{html.escape(p.stem)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>')
(root/'gallery.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReCAP qualitative cases</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:18px Georgia,serif;background:#faf9f6;color:#242424}img{width:100%;height:auto}figure{margin:32px 0;padding:20px;background:white;border:1px solid #ddd}figcaption{text-align:center;margin-top:10px}</style><h1>ReCAP qualitative cases</h1><p>Ground truth / Full context / ReCAP. These are selected illustrations, not a random evaluation sample. See artifacts/cases and docs/CASES.md for provenance and limitations.</p>'''+''.join(items)+'</html>')
print('gallery.html')
