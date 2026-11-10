"""Зведення метрик за порогом з results/runs/*.json (середнє за seed + сумарні матриці невідповідностей).
Запуск: python metrics.py  (з папки, де лежить results/). Виводить таблицю і зберігає metrics_table.csv."""
import json, glob, sys, os
import pandas as pd
# папка results: аргумент командного рядка, інакше ..\results або .\results
res = sys.argv[1] if len(sys.argv) > 1 else next((p for p in ('results', os.path.join('..', 'results')) if os.path.isdir(os.path.join(p, 'runs'))), 'results')
rows = []
for f in glob.glob(os.path.join(res, 'runs', '*.json')):
    r = json.load(open(f))
    if r.get('tag'):          # пропускаємо референтні прогони (ref224 тощо)
        continue
    for thr in ('youden', 'spec90'):
        m = r[thr]
        rows.append(dict(model=r['model'], frac=r['frac'], seed=r['seed'], thr=thr, auc=r['test_auc'],
                         recall=m['sensitivity'], specificity=m['specificity'], precision=m['precision'],
                         f1=m['f1'], bacc=m['balanced_acc'], tp=m['tp'], fp=m['fp'], tn=m['tn'], fn=m['fn']))
if not rows:
    sys.exit(f'Не знайдено прогонів у {res}\\runs. Запусти: python metrics.py A:\\MasterThesis\\results')
df = pd.DataFrame(rows)
g = df.groupby(['thr', 'frac', 'model'])
out = g[['auc', 'recall', 'specificity', 'precision', 'f1', 'bacc']].mean().round(3)
out[['tp', 'fp', 'tn', 'fn']] = g[['tp', 'fp', 'tn', 'fn']].mean().round(0)
out['n_seeds'] = g.size()
out.to_csv('metrics_table.csv')
pd.set_option('display.width', 200)
print(out.to_string())
