#!/usr/bin/env python3
"""Eksekusi semua code cell notebook dengan ml-env & tanam output (stdout/figur/Styler)
ke dalam .ipynb tanpa ipykernel. Lihat reference-eval-env."""
import os, io, ast, sys, json, base64, traceback, contextlib
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NB = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/evaluate_model_template.ipynb'
DPI = 85

nb = json.load(open(NB))
G = {'__name__': '__main__'}
exec_count = 0

def fig_to_b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format='png', dpi=DPI, bbox_inches='tight', facecolor='white')
    b.seek(0)
    return base64.b64encode(b.read()).decode('ascii')

for ci, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    outputs = []
    buf = io.StringIO()

    def flush_text():
        t = buf.getvalue()
        if t:
            outputs.append({'output_type': 'stream', 'name': 'stdout',
                            'text': t.splitlines(keepends=True)})
            buf.truncate(0); buf.seek(0)

    def my_show(*a, **k):
        flush_text()
        for num in plt.get_fignums():
            fig = plt.figure(num)
            outputs.append({'output_type': 'display_data', 'metadata': {},
                            'data': {'image/png': fig_to_b64(fig)}})
            plt.close(fig)
    plt.show = my_show

    # pisahkan trailing expression agar repr-nya tertangkap (DataFrame/Styler → html)
    trailing = None
    try:
        mod = ast.parse(src)
        if mod.body and isinstance(mod.body[-1], ast.Expr):
            trailing = mod.body.pop()
            body_code = compile(ast.Module(body=mod.body, type_ignores=[]), f'<cell {ci}>', 'exec')
            trail_code = compile(ast.Expression(trailing.value), f'<cell {ci}>', 'eval')
        else:
            body_code = compile(src, f'<cell {ci}>', 'exec')
            trail_code = None
    except SyntaxError:
        body_code = compile(src, f'<cell {ci}>', 'exec')
        trail_code = None

    exec_count += 1
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(body_code, G)
            result = eval(trail_code, G) if trail_code is not None else None
    except Exception:
        result = None
        err = traceback.format_exc()
    flush_text()
    # figur yang belum di-show()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        outputs.append({'output_type': 'display_data', 'metadata': {},
                        'data': {'image/png': fig_to_b64(fig)}})
        plt.close(fig)

    if err:
        outputs.append({'output_type': 'error', 'ename': 'Error', 'evalue': '',
                        'traceback': err.splitlines()})
        print(f'[cell {ci}] ERROR:\n{err}', file=sys.stderr)
    elif result is not None:
        data = {}
        if hasattr(result, '_repr_html_'):
            try: data['text/html'] = result._repr_html_()
            except Exception: pass
        if hasattr(result, 'to_html') and 'text/html' not in data:
            try: data['text/html'] = result.to_html()
            except Exception: pass
        data['text/plain'] = repr(result)
        outputs.append({'output_type': 'execute_result', 'execution_count': exec_count,
                        'data': data, 'metadata': {}})

    cell['outputs'] = outputs
    cell['execution_count'] = exec_count
    print(f'[cell {ci}] selesai ({len(outputs)} output)')

json.dump(nb, open(NB, 'w'), indent=1)
print('SAVED', NB)
