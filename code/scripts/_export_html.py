#!/usr/bin/env python3
"""Export executed .ipynb -> standalone HTML (tanpa nbconvert). ml-env punya `markdown`."""
import sys, json, html
import markdown as md

NB  = sys.argv[1]
OUT = sys.argv[2]
MDEXT = ['tables', 'fenced_code', 'sane_lists']

nb = json.load(open(NB))

def render_md(src):
    return md.markdown(src, extensions=MDEXT)

parts = []
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        parts.append(f'<div class="md">{render_md(src)}</div>')
    else:
        parts.append(f'<div class="code"><pre class="src"><code>{html.escape(src)}</code></pre>')
        for o in cell.get('outputs', []):
            t = o.get('output_type')
            if t == 'stream':
                parts.append(f'<pre class="stream">{html.escape("".join(o["text"]))}</pre>')
            elif t in ('display_data', 'execute_result'):
                data = o.get('data', {})
                if 'image/png' in data:
                    img = data['image/png']
                    if isinstance(img, list): img = ''.join(img)
                    parts.append(f'<img src="data:image/png;base64,{img}"/>')
                elif 'text/html' in data:
                    h = data['text/html']
                    parts.append(h if isinstance(h, str) else ''.join(h))
                elif 'text/plain' in data:
                    parts.append(f'<pre class="stream">{html.escape("".join(data["text/plain"]) if isinstance(data["text/plain"],list) else data["text/plain"])}</pre>')
            elif t == 'error':
                parts.append(f'<pre class="err">{html.escape(chr(10).join(o.get("traceback",[])))}</pre>')
        parts.append('</div>')

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;margin:0 auto;
padding:28px 38px;color:#1a1a1a;line-height:1.55;background:#fff}
h1{border-bottom:3px solid #00897B;padding-bottom:6px;margin-top:1.4em}
h2{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:1.3em;color:#00695c}
h3{color:#2e7d32;margin-top:1.2em}
.md{margin:8px 0 14px}
.code{margin:10px 0 18px;border-left:3px solid #bbdefb;padding-left:12px}
pre.src{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:6px;padding:10px 12px;overflow-x:auto;font-size:12.5px}
pre.stream{background:#fbfbfb;border:1px solid #eee;border-radius:4px;padding:8px 10px;overflow-x:auto;
font-size:12px;white-space:pre-wrap;color:#222}
pre.err{background:#fff5f5;border:1px solid #ffd0d0;color:#b00;padding:8px;overflow-x:auto;font-size:12px}
img{max-width:100%;height:auto;display:block;margin:10px 0;border:1px solid #eee;border-radius:4px}
table{border-collapse:collapse;margin:10px 0;font-size:12.5px}
table,th,td{border:1px solid #ccc}th,td{padding:4px 8px}th{background:#f0f4f8}
code{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12.5px}
blockquote{border-left:4px solid #ffb74d;background:#fff8e1;margin:10px 0;padding:6px 14px;color:#5d4037}
"""
doc = f"""<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laporan Evaluasi Lengkap — MaskDINO Tree Classification</title>
<style>{CSS}</style></head><body>
{''.join(parts)}
</body></html>"""
open(OUT, 'w').write(doc)
print('HTML →', OUT, f'({len(doc)/1e6:.1f} MB)')