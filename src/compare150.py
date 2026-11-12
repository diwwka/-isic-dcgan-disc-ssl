"""compare150.py: парні різниці DCGAN-150 проти DCGAN-30 (ті самі сіди й тестові зображення).
Запуск з папки code:  python compare150.py   (або --a ..\results --b ..\results_dcgan150 --seeds 0 1 2)"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from analyze import auc

HERE = Path(__file__).resolve().parent


def load(d, seeds):
    out = {}
    for f in sorted((Path(d) / 'runs').glob('*.json')):
        r = json.load(open(f)); z = f.with_name(f.stem + '_scores.npz')
        if r.get('tag') or not z.exists() or r['model'] not in ('disc_scratch', 'disc_ft', 'disc_fe') or r['frac'] not in (100, 10) or r['seed'] not in seeds:
            continue
        out[(r['model'], r['frac'], r['seed'])] = np.load(z)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=str(HERE / 'split_isic2019.csv'))
    p.add_argument('--a', default=str(HERE.parent / 'results'), help='DCGAN 30 епох (основний експеримент)')
    p.add_argument('--b', default=str(HERE.parent / 'results_dcgan150'), help='DCGAN 150 епох (контроль)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--B', type=int, default=1000)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    df = pd.read_csv(a.csv); A, Bn = load(a.a, a.seeds), load(a.b, a.seeds)
    te = next(iter(A.values()))['test_idx']
    assert all((z['test_idx'] == te).all() for z in list(A.values()) + list(Bn.values()))
    y = df.label_bin.values[te].astype(int); grp = df.group.values[te]
    ug, inv = np.unique(grp, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(len(ug))]
    rng = np.random.RandomState(0)
    boots = [np.concatenate([rows[g] for g in rng.randint(0, len(ug), len(ug))]) for _ in range(a.B)]

    def stat(src, m, f):
        zs = [src[(m, f, s)] for s in a.seeds]
        return np.mean([auc(y, z['test']) for z in zs]), np.mean([[auc(y[b], z['test'][b]) for b in boots] for z in zs], axis=0)
    res = []
    for f in (100, 10):
        for m in ('disc_ft', 'disc_fe', 'disc_scratch'):
            (pa, ma), (pb, mb) = stat(A, m, f), stat(Bn, m, f)
            d = mb - ma; lo, hi = np.nanpercentile(d, [2.5, 97.5])
            res.append(dict(frac=f, comparison=f'{m}: 150 ep - 30 ep', auc30=round(pa, 3), auc150=round(pb, 3), diff=round(pb - pa, 3),
                            ci_lo=round(lo, 3), ci_hi=round(hi, 3), excludes_zero=bool(lo > 0 or hi < 0)))
    r = pd.DataFrame(res); print('\n== DCGAN: 150 епох проти 30 епох (парні різниці, сіди', a.seeds, ') ==\n', r.to_string(index=False))
    outp = Path(a.out or a.b) / 'dcgan_150_vs_30.csv'; r.to_csv(outp, index=False); print('Збережено', outp)


if __name__ == '__main__':
    main()
