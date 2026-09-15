#!/usr/bin/env python
"""Konverter terfokus thesis_full.tex -> thesis_full.html (self-contained).
Menangani konstruk yang dipakai dokumen ini saja. Math dibiarkan verbatim untuk
MathJax (CDN). Figur PNG di-embed base64. Bukan konverter LaTeX umum."""
import re, os, base64

PAPER = "/scratch2/pr65/anur0018/tree_classification/paper"
FIGDIR = "/scratch2/pr65/anur0018/tree_classification/reports/figures"
TEX = f"{PAPER}/thesis_full.tex"
OUT = f"{PAPER}/thesis_full.html"

def oneline(s):
    return re.sub(r"[ \t]*\n[ \t]*", " ", s)

src = open(TEX).read()
pre, body = src.split(r"\begin{document}", 1)
body = body.split(r"\end{document}")[0]

title = re.search(r"\\title\{(.+?)\}", pre, re.S).group(1)
title = re.sub(r"\\textbf\{(.+?)\}", r"\1", title).replace("\\\\", " ").strip()
author_m = re.search(r"\\author\{(.+?)\}", pre, re.S)
author = author_m.group(1) if author_m else ""
author_lines = [re.sub(r"\\[a-z]+|[{}]|\\\\", "", l).strip()
                for l in author.split("\\\\") if l.strip()]

# ---- pre-scan: bib keys -> number, figure/table labels -> number ----
bibkeys = re.findall(r"\\bibitem\{([^}]+)\}", body)
citenum = {k: i + 1 for i, k in enumerate(bibkeys)}

fig_labels, tab_labels = [], []
for env, store in (("figure", fig_labels), ("table", tab_labels)):
    for m in re.finditer(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", body, re.S):
        lab = re.search(r"\\label\{([^}]+)\}", m.group(0))
        if lab:
            store.append(lab.group(1))
refnum = {}
for i, l in enumerate(fig_labels):
    refnum[l] = ("Figure", i + 1)
for i, l in enumerate(tab_labels):
    refnum[l] = ("Table", i + 1)


def inline(t):
    t = re.sub(r"\\noindent\s*|\\small\s*|\\centering\s*", "", t)
    t = t.replace("\\&", "&amp;").replace("\\%", "%").replace("\\_", "_")
    t = re.sub(r"\\cite\{([^}]+)\}",
               lambda m: "".join(f'<a class="cite" href="#ref-{k.strip()}">[{citenum.get(k.strip(),"?")}]</a>'
                                 for k in m.group(1).split(",")), t)
    t = re.sub(r"\\ref\{([^}]+)\}",
               lambda m: f'{refnum.get(m.group(1),("?",0))[1]}' , t)
    t = re.sub(r"\\textbf\{(.+?)\}", r"<strong>\1</strong>", t)
    t = re.sub(r"\\textit\{(.+?)\}", r"<em>\1</em>", t)
    t = re.sub(r"\\emph\{(.+?)\}", r"<em>\1</em>", t)
    t = t.replace("~", "&nbsp;")
    t = re.sub(r"\{,\}", ",", t)          # 75{,}000 -> 75,000
    t = t.replace("\\,", " ")
    return t


def conv_table(block):
    cap = re.search(r"\\caption\{(.+?)\}\s*\n", block, re.S)
    cap = inline(re.sub(r"\s+", " ", cap.group(1)).strip()) if cap else ""
    lab = re.search(r"\\label\{([^}]+)\}", block)
    num = refnum.get(lab.group(1), ("Table", "?"))[1] if lab else "?"
    tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S).group(1)
    rows = []
    for raw in tab.split("\\\\"):
        raw = re.sub(r"\\(top|mid|bottom)rule", "", raw).strip()
        if not raw:
            continue
        cells = [inline(c.strip()) for c in raw.split("&")]
        rows.append(cells)
    html = [f'<figure class="tbl"><table>']
    if rows:
        html.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr></thead>")
        html.append("<tbody>")
        for r in rows[1:]:
            html.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        html.append("</tbody>")
    html.append("</table>")
    html.append(f'<figcaption><b>Table {num}.</b> {cap}</figcaption></figure>')
    return "\n".join(html)


def conv_figure(block):
    img = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", block).group(1)
    png = os.path.join(FIGDIR, os.path.basename(img).replace(".pdf", ".png"))
    data = base64.b64encode(open(png, "rb").read()).decode()
    cap = re.search(r"\\caption\{(.+?)\}\s*\\label", block, re.S)
    cap = inline(re.sub(r"\s+", " ", cap.group(1)).strip()) if cap else ""
    lab = re.search(r"\\label\{([^}]+)\}", block)
    num = refnum.get(lab.group(1), ("Figure", "?"))[1] if lab else "?"
    return (f'<figure class="fig"><img src="data:image/png;base64,{data}" alt="figure {num}">'
            f'<figcaption><b>Figure {num}.</b> {cap}</figcaption></figure>')


out = []
# split body into top-level environment blocks and text
pattern = re.compile(
    r"\\begin\{(abstract|table|figure|equation|thebibliography|description|enumerate|itemize)\}"
    r"(?:\{[^}]*\})?(.*?)\\end\{\1\}", re.S)
pos = 0
segments = []
for m in pattern.finditer(body):
    if m.start() > pos:
        segments.append(("text", body[pos:m.start()]))
    segments.append((m.group(1), m.group(2)))
    pos = m.end()
segments.append(("text", body[pos:]))

toc = []


def emit_text(txt):
    for m in re.finditer(r"\\(chapter|section|subsection)\{([^}]+)\}", txt):
        pass
    # process line by line, splitting on structural commands and blank lines
    txt = re.sub(r"\\maketitle|\\tableofcontents", "", txt)
    # headings
    parts = re.split(r"(\\chapter\{[^}]+\}|\\section\{[^}]+\}|\\subsection\{[^}]+\})", txt)
    for part in parts:
        h = re.match(r"\\(chapter|section|subsection)\{([^}]+)\}", part)
        if h:
            lvl = {"chapter": 2, "section": 3, "subsection": 4}[h.group(1)]
            tid = re.sub(r"[^a-z0-9]+", "-", h.group(2).lower()).strip("-")
            out.append(f'<h{lvl} id="{tid}">{inline(h.group(2))}</h{lvl}>')
            if h.group(1) in ("chapter", "section"):
                toc.append((h.group(1), tid, h.group(2)))
            continue
        # paragraphs separated by blank lines
        for para in re.split(r"\n\s*\n", part):
            para = para.strip()
            if not para:
                continue
            pg = re.match(r"\\paragraph\{([^}]+)\}(.*)", para, re.S)
            if pg:
                rest = oneline(pg.group(2).strip())
                out.append(f'<p><strong>{inline(pg.group(1))}.</strong> {inline(rest)}</p>')
            else:
                out.append(f"<p>{inline(oneline(para))}</p>")


for kind, content in segments:
    if kind == "text":
        emit_text(content)
    elif kind == "abstract":
        out.append('<div class="abstract"><h2>Abstract</h2>' +
                   "".join(f"<p>{inline(oneline(p.strip()))}</p>"
                           for p in re.split(r"\n\s*\n", content) if p.strip()) + "</div>")
    elif kind == "table":
        out.append(conv_table("\\begin{table}" + content + "\\end{table}"))
    elif kind == "figure":
        out.append(conv_figure(content))
    elif kind == "equation":
        out.append(r'<div class="eq">\[' + content.strip() + r'\]</div>')
    elif kind in ("description", "enumerate", "itemize"):
        tag = "ol" if kind == "enumerate" else "ul"
        items = re.split(r"\\item", content)[1:]
        lis = []
        for it in items:
            it = it.strip()
            lead = re.match(r"\[([^\]]+)\]\s*(.*)", it, re.S)
            if lead:
                lis.append(f"<li><strong>{inline(lead.group(1))}</strong> "
                           f"{inline(oneline(lead.group(2).strip()))}</li>")
            else:
                lis.append(f"<li>{inline(oneline(it))}</li>")
        out.append(f"<{tag} class='lst'>" + "".join(lis) + f"</{tag}>")
    elif kind == "thebibliography":
        refs = re.split(r"\\bibitem\{([^}]+)\}", content)
        html = ['<h2 id="references">References</h2><ol class="refs">']
        for i in range(1, len(refs), 2):
            key, txt = refs[i], refs[i + 1].strip()
            html.append(f'<li id="ref-{key}">{inline(oneline(txt))}</li>')
        html.append("</ol>")
        out.append("\n".join(html))

toc_html = '<nav class="toc"><h2>Contents</h2><ul>' + "".join(
    f'<li class="toc-{k}"><a href="#{tid}">{t}</a></li>' for k, tid, t in toc) + "</ul></nav>"

authors_html = "<br>".join(author_lines)
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']]}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root{{--ink:#1a1a1a;--muted:#555;--rule:#ddd;--accent:#0072B2;--bg:#fff;}}
@media(prefers-color-scheme:dark){{:root{{--ink:#e6e6e6;--muted:#a0a0a0;--rule:#333;--accent:#4aa3df;--bg:#161616;}}}}
html{{background:var(--bg)}}
body{{color:var(--ink);background:var(--bg);max-width:820px;margin:0 auto;padding:2.5rem 1.3rem 5rem;
font:16px/1.65 Georgia,'Times New Roman',serif;}}
h1.title{{font-size:1.9rem;line-height:1.25;margin:.2em 0}}
.byline{{color:var(--muted);font-size:.95rem;margin-bottom:2rem}}
h2{{font-size:1.45rem;border-bottom:2px solid var(--rule);padding-bottom:.25em;margin-top:2.4em}}
h3{{font-size:1.18rem;margin-top:1.8em}} h4{{font-size:1.02rem;color:var(--muted);margin-top:1.4em}}
p{{margin:.7em 0;text-align:justify;hyphens:auto}}
a{{color:var(--accent)}} a.cite{{text-decoration:none;font-size:.85em;vertical-align:super}}
.abstract{{background:rgba(0,114,178,.06);border-left:3px solid var(--accent);padding:.6rem 1.2rem;
border-radius:4px;margin:1.5rem 0}} .abstract h2{{border:0;font-size:1.1rem;margin:.4em 0}}
.abstract p{{font-size:.95rem}}
nav.toc{{background:rgba(128,128,128,.06);border:1px solid var(--rule);border-radius:6px;padding:.6rem 1.4rem;margin:1.5rem 0}}
nav.toc ul{{list-style:none;padding-left:0;margin:.3em 0}} nav.toc h2{{border:0;font-size:1.1rem}}
.toc-section a{{padding-left:1.4em;color:var(--muted);font-size:.92em}}
.toc-chapter a{{font-weight:bold}}
figure.fig,figure.tbl{{margin:1.8em 0;text-align:center}}
figure.fig img{{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:4px}}
figcaption{{font-size:.86rem;color:var(--muted);text-align:left;margin-top:.5em}}
table{{border-collapse:collapse;margin:0 auto;font-size:.88rem;font-family:Arial,sans-serif}}
th,td{{padding:.32em .7em;text-align:center}} th{{border-bottom:2px solid var(--ink)}}
thead tr{{border-top:2px solid var(--ink)}} tbody tr:last-child td{{border-bottom:2px solid var(--ink)}}
td:first-child,th:first-child{{text-align:left}}
.tbl{{overflow-x:auto}}
ol.refs{{font-size:.9rem;line-height:1.5}} ol.refs li{{margin:.5em 0}}
.eq{{overflow-x:auto;margin:1em 0}}
ul.lst,ol.lst{{padding-left:1.3em}} ul.lst li,ol.lst li{{margin:.5em 0;text-align:justify}}
</style></head><body>
<h1 class="title">{title}</h1>
<div class="byline">{authors_html}</div>
{toc_html}
{''.join(out)}
</body></html>"""

open(OUT, "w").write(html)
print(f"wrote {OUT} ({len(html)//1024} KB), figures embedded, {len(bibkeys)} refs, {len(toc)} toc entries")
# leftover latex commands check
leftover = re.findall(r"\\[a-zA-Z]+", re.sub(r"\\\[.*?\\\]", "", "".join(out), flags=re.S))
from collections import Counter
print("leftover TeX cmds (should be math-only):", Counter(leftover).most_common(12))
