#!/usr/bin/env python3
"""
build_report_slides.py — Ubah laporan HTML jadi dek slide HTML mandiri.

Sumber : reports/stratified_v4/lr_restart_comparison_interactive.html
Keluaran: reports/stratified_v4/lr_restart_comparison_slides.html

Pendekatan: CSS dan JS laporan dipakai ulang apa adanya, jadi seluruh grafik tetap
hidup dan interaktif. Yang ditambahkan hanya pembungkus slide, navigasi, dan CSS cetak.
Isi laporan tidak diubah, sehingga kalau laporannya diperbarui cukup jalankan ulang skrip ini.

Grafik dibangun lewat pencarian elemen berdasarkan id, bukan lewat pengukuran tata letak,
sehingga tetap ter-render walau slide-nya sedang tersembunyi.

Jalankan:
    python scripts/build_report_slides.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "stratified_v4" / "lr_restart_comparison_interactive.html"
DST = ROOT / "reports" / "stratified_v4" / "lr_restart_comparison_slides.html"

# Satu entri = satu slide. Angka merujuk indeks elemen tingkat atas di dalam .page.
# Section-block besar (26, 39, 40, 41) dibongkar dulu jadi anak-anaknya, lihat split_blocks.
SLIDES = [
    ("judul",            [0, 1, 2, 3]),
    ("ringkasan",        [4]),
    ("isi",              [5]),
    ("bag1-chart",       [6, 7, 8, 9]),
    ("bag1-temuan",      [10, 11]),
    ("bag2-diagram",     [12, 13, 14, 15]),
    ("bag3-komposisi",   [16, 17, 18, 19]),
    ("bag3-literatur",   [20]),
    ("bag3-seed",        [21, 22, 23]),
    ("bag3-perspesies",  [24, 25]),
    ("bag4",             ["26:*"]),
    ("bag5-basis",       [27, 28, 29, 30, 31, 32, 33]),
    ("bag5-metrik",      [34]),
    ("bag5-baca",        [35]),
    ("bag5-live",        [36]),
    ("bag5-ringkas",     [37, 38]),
    ("bag5-tradeoff",    [39]),
    ("bag5-11sp",        ["40:*"]),
    ("bag5-7sp",         ["41:*"]),
    ("bag5-periter",     ["42:*"]),
    ("bag6",             ["43:*"]),
    ("bag7",             ["44:*"]),
    ("bag8",             ["45:*"]),
    ("lampiran",         ["46:*"]),
]

NAV_CSS = """
  /* ---------- kerangka slide ---------- */
  html, body { height: 100%; overflow: hidden; }
  body { background: var(--surface-2); }

  .deck { height: 100vh; position: relative; }

  .slide {
    position: absolute; inset: 0;
    display: none;
    overflow-y: auto;
    background: var(--surface);
    padding: 46px 56px 78px;
  }
  .slide.active { display: block; }
  .slide-inner { max-width: 1180px; margin: 0 auto; }

  /* laporan memakai .page; di slide perannya diambil .slide-inner */
  .page { max-width: none; margin: 0; padding: 0; }

  /* rapatkan jarak antar-elemen supaya lebih banyak yang muat per slide */
  .slide .card { margin-bottom: 18px; }
  .slide .section-block { margin-top: 0; }
  .slide h1 { font-size: 25px; margin-bottom: 6px; }
  .slide .subtitle { margin-bottom: 20px; max-width: 78ch; }

  /* slide judul */
  .slide[data-name="judul"] { display: none; }
  .slide[data-name="judul"].active {
    display: flex; align-items: center;
  }
  .slide[data-name="judul"] .slide-inner { width: 100%; }
  .slide[data-name="judul"] h1 { font-size: 40px; line-height: 1.15; }

  /* ---------- navigasi ---------- */
  .navbar {
    position: fixed; left: 0; right: 0; bottom: 0; height: 46px;
    display: flex; align-items: center; gap: 14px;
    padding: 0 20px;
    background: var(--surface);
    border-top: 1px solid var(--line);
    font-size: 12.5px; color: var(--ink-2);
    z-index: 40;
  }
  .navbar button {
    font: inherit; color: var(--ink-2);
    background: none; border: 1px solid var(--line-strong);
    border-radius: 6px; padding: 4px 11px; cursor: pointer;
  }
  .navbar button:hover { background: var(--surface-2); color: var(--ink); }
  .navbar button:disabled { opacity: 0.35; cursor: default; }
  .nav-title { font-weight: 600; color: var(--ink); }
  .nav-count { margin-left: auto; font-variant-numeric: tabular-nums; }
  .nav-hint { color: var(--ink-3); font-size: 11.5px; }

  .progress {
    position: fixed; left: 0; top: 0; height: 3px;
    background: var(--blue); z-index: 41;
    transition: width 0.18s ease;
  }

  /* ---------- cetak ke PDF ---------- */
  @media print {
    html, body { height: auto; overflow: visible; background: #fff; }
    .navbar, .progress { display: none !important; }
    .deck { height: auto; }
    .slide {
      position: static; display: block !important;
      page-break-after: always; break-after: page;
      min-height: auto; padding: 24px 30px;
      overflow: visible;
    }
    .slide[data-name="judul"] { display: block !important; }
  }
"""

NAV_JS = """
(function () {
  "use strict";
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var idx = 0;
  var elTitle = document.getElementById("navTitle");
  var elCount = document.getElementById("navCount");
  var elProg = document.getElementById("progress");
  var btnPrev = document.getElementById("navPrev");
  var btnNext = document.getElementById("navNext");

  function titleOf(s) {
    var h = s.querySelector("h1");
    if (h) return h.textContent.trim();
    var t = s.querySelector(".chart-title");
    return t ? t.textContent.trim() : "";
  }

  function show(i) {
    idx = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach(function (s, k) { s.classList.toggle("active", k === idx); });
    slides[idx].scrollTop = 0;
    elTitle.textContent = titleOf(slides[idx]);
    elCount.textContent = (idx + 1) + " / " + slides.length;
    elProg.style.width = ((idx + 1) / slides.length * 100) + "%";
    btnPrev.disabled = idx === 0;
    btnNext.disabled = idx === slides.length - 1;
    if (history.replaceState) history.replaceState(null, "", "#" + (idx + 1));
  }

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    var k = e.key;
    if (k === "ArrowRight" || k === "PageDown" || k === " ") { e.preventDefault(); show(idx + 1); }
    else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); show(idx - 1); }
    else if (k === "Home") { e.preventDefault(); show(0); }
    else if (k === "End") { e.preventDefault(); show(slides.length - 1); }
  });
  btnPrev.addEventListener("click", function () { show(idx - 1); });
  btnNext.addEventListener("click", function () { show(idx + 1); });

  var start = parseInt((location.hash || "#1").slice(1), 10);
  show(isNaN(start) ? 0 : start - 1);
})();
"""


def top_level_children(body: str):
    """Kembalikan daftar potongan HTML anak langsung dari .page."""
    start = body.index(">") + 1
    inner = body[start:]
    kids, depth, open_at = [], 0, 0
    for m in re.finditer(r"<(/?)(div|p|h1|dl|ol|ul)\b[^>]*>", inner):
        if m.group(1) != "/":
            if depth == 0:
                open_at = m.start()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                kids.append(inner[open_at:m.end()])
    return kids


def split_block(frag: str):
    """Bongkar satu <div class="section-block"> jadi daftar anak-anaknya."""
    start = frag.index(">") + 1
    end = frag.rindex("</div>")
    return top_level_children('<div>' + frag[start:end] + "</div>")



def _find_child(frag: str, opener_re: str):
    """Cari blok anak berimbang yang diawali pola opener_re. Kembalikan (awal, akhir) atau None."""
    m = re.search(opener_re, frag)
    if not m:
        return None
    depth, i = 0, m.start()
    for t in re.finditer(r"<(/?)div\b[^>]*>", frag[m.start():]):
        depth += 1 if t.group(1) != "/" else -1
        if depth == 0:
            return (m.start(), m.start() + t.end())
    return None


def split_oversized(slides):
    """Pecah slide yang terlalu penuh.

    Dua pola yang sering bikin meluber:
      1. satu kartu berisi grafik/tabel PLUS blok .findings -> temuan dipindah ke slide sendiri
      2. kartu diagram yang juga memuat tabel glosarium -> glosarium dipindah
    """
    out = []
    for name, content in slides:
        # 0. kartu lampiran berisi beberapa .param-group -> satu grup per slide
        groups = []
        rest = content
        while True:
            g = _find_child(rest, r'<div class="param-group"[^>]*>')
            if not g:
                break
            groups.append(rest[g[0]:g[1]])
            rest = rest[:g[0]] + rest[g[1]:]
        if len(groups) >= 2:
            head = re.search(r'<div class="chart-head">.*?</div>\s*</div>', rest, re.S)
            head = head.group(0)[:head.group(0).rindex("</div>")] if head else ""
            for k, grp in enumerate(groups, 1):
                title = re.search(r"<h3[^>]*>(.*?)</h3>", grp, re.S)
                out.append((f"{name}-g{k}",
                            '<div class="card">' + (head if k == 1 or not title else "")
                            + grp + "</div>"))
            continue

        # 1. glosarium di dalam kartu diagram
        g = _find_child(content, r'<div style="margin-top:22px;padding-top:18px;border-top')
        if g:
            gl = content[g[0]:g[1]]
            content = content[:g[0]] + content[g[1]:]
            out.append((name, content))
            out.append((name + "-glosarium", '<div class="card">' + gl + "</div>"))
            continue

        # 2. blok .findings di dalam kartu
        f = _find_child(content, r'<div class="findings[^"]*"[^>]*>')
        if f and content.count("<table") + content.count("<svg") >= 1:
            fnd = content[f[0]:f[1]]
            rest = content[:f[0]] + content[f[1]:]
            out.append((name, rest))
            out.append((name + "-temuan", fnd))
            continue

        out.append((name, content))
    return out


def main():
    src = SRC.read_text()
    style = re.search(r"<style>.*?</style>", src, re.S).group(0)
    scripts = re.findall(r"<script>.*?</script>", src, re.S)
    body = src[src.index('<div class="page">'):src.index("<script>")]
    kids = top_level_children(body)
    print(f"elemen tingkat atas: {len(kids)}")

    out = []
    for name, spec in SLIDES:
        parts = []
        for item in spec:
            if isinstance(item, str) and item.endswith(":*"):
                i = int(item.split(":")[0])
                sub = split_block(kids[i])
                # anak pertama biasanya eyebrow + h1 + subtitle -> gabung jadi 1 slide,
                # sisanya masing-masing jadi slide sendiri
                head, rest = [], []
                for c in sub:
                    if re.match(r'\s*<(p class="eyebrow|h1|p class="subtitle)', c) and not rest:
                        head.append(c)
                    else:
                        rest.append(c)
                parts.append("\n".join(head))
                for r in rest:
                    parts.append(r)
            else:
                parts.append(kids[item])
        if len(parts) == 1 or not any(isinstance(x, str) and x.endswith(":*") for x in spec):
            out.append((name, "\n".join(parts)))
        else:
            # spec ber-":*" menghasilkan banyak slide
            head = parts[0]
            out.append((name, head))
            for k, extra in enumerate(parts[1:], 2):
                out.append((f"{name}-{k}", extra))

    out = split_oversized(out)

    slides_html = []
    for name, content in out:
        if not content.strip():
            continue
        slides_html.append(
            f'<section class="slide" data-name="{name}">\n'
            f'  <div class="slide-inner"><div class="page">\n{content}\n</div></div>\n'
            f"</section>"
        )
    print(f"slide dibuat: {len(slides_html)}")

    html = [
        # Charset WAJIB paling depan dan harus berada dalam 1024 byte pertama. Tanpa ini
        # browser menebak encoding dan karakter seperti −, ², ±, → tampil rusak. Deck ini
        # memuat puluhan di antaranya, dan bug itu sempat lolos ke berkas hasil.
        '<meta charset="utf-8">',
        "<title>Arsitektur head DBH: frozen vs joint (slide)</title>",
        style,
        f"<style>{NAV_CSS}</style>",
        '<div class="progress" id="progress"></div>',
        '<div class="deck">',
        "\n".join(slides_html),
        "</div>",
        '<div class="navbar">',
        '  <button id="navPrev" type="button">&larr; Sebelumnya</button>',
        '  <button id="navNext" type="button">Berikutnya &rarr;</button>',
        '  <span class="nav-title" id="navTitle"></span>',
        '  <span class="nav-hint">panah kiri/kanan untuk berpindah, Ctrl+P untuk cetak ke PDF</span>',
        '  <span class="nav-count" id="navCount"></span>',
        "</div>",
    ]
    html.extend(scripts)
    html.append(f"<script>{NAV_JS}</script>")
    DST.write_text("\n".join(html))
    print(f"Tersimpan: {DST}")


if __name__ == "__main__":
    main()
