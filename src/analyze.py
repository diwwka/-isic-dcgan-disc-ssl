"""analyze.py: довірчі інтервали (bootstrap за ураженнями), парні порівняння моделей та AUC за джерелами.
Потребує файлів *_scores.npz, які створює exp.py. Запуск: python analyze.py   (або --csv ... --out ...)"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PAIRS = [('disc_ft', 'disc_scratch'), ('disc_fe', 'disc_scratch'), ('disc_scratch', 'resnet_scratch'),
         ('resnet_imagenet', 'disc_ft'), ('resnet_imagenet', 'disc_fe')]
FRACS = [100, 10, 1]


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=str(HERE / 'split_isic2019.csv'))
    p.add_argument('--out', default=str(HERE.parent / 'results'))
    p.add_argument('--B', type=int, default=1000)
    a = p.parse_args()
    out = Path(a.out); df = pd.read_csv(a.csv)
    runs = []
    for f in sorted((out / 'runs').glob('*.json')):
        r = json.load(open(f))
        z = f.with_name(f.stem + '_scores.npz')
        if r.get('tag') or not z.exists():
            continue
        runs.append((r['model'], r['frac'], r['seed'], np.load(z)))
    if not runs:
        raise SystemExit('Немає файлів *_scores.npz. Спершу перезапусти експерименти оновленим exp.py.')
    te = runs[0][3]['test_idx']
    assert all((r[3]['test_idx'] == te).all() for r in runs), 'тестові індекси мають збігатися'
    y = df.label_bin.values[te].astype(int); grp = df.group.values[te]; src = df.source.values[te]
    ug, inv = np.unique(grp, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(len(ug))]
    rng = np.random.RandomState(0)
    boots = [np.concatenate([rows[g] for g in rng.randint(0, len(ug), len(ug))]) for _ in range(a.B)]
    keys = sorted({(m, f) for m, f, _, _ in runs})
    point, M = {}, {}
    for m, f in keys:
        rs = [z for mm, ff, _, z in runs if (mm, ff) == (m, f)]
        point[(m, f)] = (np.mean([auc(y, z['test']) for z in rs]), np.std([auc(y, z['test']) for z in rs]), len(rs))
        M[(m, f)] = np.mean([[auc(y[b], z['test'][b]) for b in boots] for z in rs], axis=0)   # (B,)
    rows_ci = [dict(model=m, frac=f, n_seeds=point[(m, f)][2], auc=round(point[(m, f)][0], 3),
                    ci_lo=round(np.nanpercentile(M[(m, f)], 2.5), 3), ci_hi=round(np.nanpercentile(M[(m, f)], 97.5), 3),
                    seed_std=round(point[(m, f)][1], 3)) for m, f in keys]
    ci = pd.DataFrame(rows_ci).sort_values(['frac', 'model'], ascending=[False, True])
    ci.to_csv(out / 'auc_ci.csv', index=False)
    print('\n== AUC на тесті (середнє за сідами) з 95% bootstrap-інтервалом за ураженнями ==')
    print(ci.to_string(index=False))
    pr = []
    for f in FRACS:
        for x, z in PAIRS:
            if (x, f) in M and (z, f) in M:
                d = M[(x, f)] - M[(z, f)]; lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pr.append(dict(frac=f, comparison=f'{x} - {z}', diff=round(point[(x, f)][0] - point[(z, f)][0], 3),
                               ci_lo=round(lo, 3), ci_hi=round(hi, 3), excludes_zero=bool(lo > 0 or hi < 0)))
    pr = pd.DataFrame(pr); pr.to_csv(out / 'paired_diffs.csv', index=False)
    print('\n== Парні різниці AUC (одні й ті самі тестові зображення) ==')
    print(pr.to_string(index=False))
    tr = df[df.split == 'train']; rate = tr.groupby('source').label_bin.mean()
    base = auc(y, pd.Series(src).map(rate).values)
    srcs = [s for s in np.unique(src) if len(np.unique(y[src == s])) == 2]
    bs = []
    for m, f in keys:
        rs = [z for mm, ff, _, z in runs if (mm, ff) == (m, f)]
        bs.append(dict(model=m, frac=f, **{s: round(np.mean([auc(y[src == s], z['test'][src == s]) for z in rs]), 3) for s in srcs}))
    bs = pd.DataFrame(bs).sort_values(['frac', 'model'], ascending=[False, True]); bs.to_csv(out / 'auc_by_source.csv', index=False)
    print(f'\n== AUC усередині кожного джерела (усунуто ярлик «джерело»); джерело-лише baseline AUC = {base:.3f} ==')
    print(bs.to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, s in ci.groupby('model'):
        s = s.set_index('frac').reindex(FRACS)
        ax.errorbar(np.arange(3), s.auc, yerr=[s.auc - s.ci_lo, s.ci_hi - s.auc], marker='o', capsize=3, label=k)
    ax.axhline(base, ls='--', c='gray', lw=1, label=f'лише джерело ({base:.2f})')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'{f}%' for f in FRACS]); ax.set_xlabel('Частка розмічених даних')
    ax.set_ylabel('Test ROC-AUC'); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / 'auc_vs_labels_ci.png', dpi=150); print('\nЗбережено: auc_ci.csv, paired_diffs.csv, auc_by_source.csv, auc_vs_labels_ci.png')


if __name__ == '__main__':
    main()
