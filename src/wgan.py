"""wgan.py: спрощена перевірка H3 (WGAN-GP проти класичного DCGAN) поверх exp.py. exp.py НЕ змінюється.

Ідея: критик WGAN-GP не може мати BatchNorm (штраф градієнта рахується для кожного зразка окремо), тому
для цього варіанта дискримінатор використовує GroupNorm(1, C) (еквівалент LayerNorm). Щоб різницю не
плутати з архітектурою, є контроль wdisc_scratch: та сама архітектура з випадковою ініціалізацією.

Команди (запускати з папки src):
  python wgan.py smoke                      перевірка конвеєра на штучних даних (кілька хвилин)
  python wgan.py pretrain --seed 0          WGAN-GP передтренування (без міток)
  python wgan.py all --seeds 0 1 2          pretrain (якщо немає) + 3 моделі x частки 100/10 % x сіди
  python wgan.py analyze                    таблиці AUC, парні різниці, графік втрат
Моделі: wdisc_scratch | wdisc_ft | wdisc_fe
Результати: ../results_wgan  (основні результати ../results не змінюються)
"""
import argparse, json, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
import exp
from exp import Data, Generator, batch, winit

HERE = Path(__file__).resolve().parent
WMODELS = ['wdisc_scratch', 'wdisc_ft', 'wdisc_fe']
WFRACS = [100, 10]
NC, LAM = 5, 10.0           # кроків критика на крок генератора; коефіцієнт штрафу градієнта


def ln_backbone(nc=3, ndf=64):  # та сама форма, що й exp.disc_backbone (64x64 -> 512x4x4), але без BatchNorm
    return nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.GroupNorm(1, ndf * 2), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.GroupNorm(1, ndf * 4), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.GroupNorm(1, ndf * 8), nn.LeakyReLU(0.2, True))


class Critic(nn.Module):
    def __init__(s):
        super().__init__()
        s.backbone = ln_backbone(); s.head = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(s, x):
        return s.head(s.backbone(x)).view(-1)


class WDiscCls(nn.Module):
    def __init__(s, freeze=False):
        super().__init__()
        s.backbone = ln_backbone(); s.freeze = freeze
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(512 * 4 * 4, 1))

    def forward(s, x):
        if s.freeze:
            with torch.no_grad():
                f = s.backbone(x)
        else:
            f = s.backbone(x)
        return s.head(f).view(-1)


_orig_build = exp.build_model


def build_model(kind, ckpt, dev, smoke=False, size=64):
    if not kind.startswith('wdisc'):
        return _orig_build(kind, ckpt, dev, smoke, size)
    assert size == 64, 'wgan.py розрахований лише на 64 px'
    m = WDiscCls(freeze=(kind == 'wdisc_fe'))
    m.apply(winit)
    if kind in ('wdisc_ft', 'wdisc_fe'):
        m.backbone.load_state_dict(torch.load(ckpt, map_location='cpu')['D_backbone'])
        if kind == 'wdisc_fe':
            for p in m.backbone.parameters():
                p.requires_grad = False
    return m.to(dev)


exp.build_model = build_model   # train_cls з exp.py тепер знає про wdisc_*


def pretrain_wgan(a, df, data, dev):
    out = Path(a.out) / 'pretrain'; out.mkdir(parents=True, exist_ok=True)
    ck = out / f'wgan_s{a.seed}.pt'
    torch.manual_seed(a.seed)
    tr = np.where(df.split.values == 'train')[0]          # усі train-зображення, МІТКИ НЕ ВИКОРИСТОВУЮТЬСЯ
    G, C = Generator().to(dev), Critic().to(dev)
    G.apply(winit); C.apply(winit)
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    oC = torch.optim.Adam(C.parameters(), 1e-4, betas=(0.5, 0.9))
    fixed = torch.randn(64, 100, 1, 1, device=dev)
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        perm = np.random.RandomState(a.seed * 1000 + ep).permutation(tr)
        nb = max(1, len(perm) // a.bs); s = np.zeros(4); ng = 0
        for b in range(nb):
            real = batch(data, perm[b * a.bs:(b + 1) * a.bs], 'disc', True)
            n = real.size(0)
            with torch.no_grad():
                fake = G(torch.randn(n, 100, 1, 1, device=dev))
            c_real, c_fake = C(real).mean(), C(fake).mean()
            eps = torch.rand(n, 1, 1, 1, device=dev)
            xh = (eps * real + (1 - eps) * fake).requires_grad_(True)
            g = torch.autograd.grad(C(xh).sum(), xh, create_graph=True)[0]
            gp = ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
            lossC = c_fake - c_real + LAM * gp
            oC.zero_grad(); lossC.backward(); oC.step()
            s[0] += lossC.item(); s[1] += (c_real - c_fake).item(); s[2] += gp.item()
            if (b + 1) % NC == 0:
                lossG = -C(G(torch.randn(n, 100, 1, 1, device=dev))).mean()
                oG.zero_grad(); lossG.backward(); oG.step()
                s[3] += lossG.item(); ng += 1
        s[:3] /= nb; s[3] /= max(ng, 1)
        hist.append(dict(epoch=ep + 1, loss_C=s[0], w_est=s[1], gp=s[2], loss_G=s[3]))
        print(f'  [wgan s{a.seed}] ep {ep + 1}/{a.epochs} lossC {s[0]:.3f} W-est {s[1]:.3f} GP {s[2]:.3f} lossG {s[3]:.3f}', flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            G.eval()
            with torch.no_grad():
                torchvision.utils.save_image(G(fixed).cpu(), out / f'wgan_samples_s{a.seed}_ep{ep + 1}.png', normalize=True, nrow=8)
            G.train()
    torch.save({'D_backbone': C.backbone.state_dict(), 'G': G.state_dict(), 'epochs': a.epochs}, ck)
    json.dump(dict(seed=a.seed, epochs=a.epochs, nc=NC, lam=LAM, g_steps_per_epoch=nb // NC, seconds=time.time() - t0, history=hist),
              open(out / f'wgan_s{a.seed}.json', 'w'), indent=1)
    print(f'  [wgan] збережено {ck}  ({time.time() - t0:.0f} с)')
    return ck


def run_all(a, df, data, dev):
    for seed in a.seeds:
        ck = Path(a.out) / 'pretrain' / f'wgan_s{seed}.pt'
        a.seed = seed
        if not ck.exists():
            pretrain_wgan(a, df, data, dev)
        for frac in WFRACS:
            for kind in WMODELS:
                a.lr = 1e-3 if kind == 'wdisc_fe' else None      # як у disc_fe в основному експерименті
                exp.train_cls(a, df, data, dev, kind, frac, seed, ck)


# ---------------------------------------------------------------- аналіз
def analyze(a):
    from analyze import auc
    df = pd.read_csv(a.csv)
    runs = []
    for d, kinds in ((Path(a.main), ['disc_scratch', 'disc_ft', 'disc_fe']), (Path(a.out), WMODELS)):
        for f in sorted((d / 'runs').glob('*.json')):
            r = json.load(open(f)); z = f.with_name(f.stem + '_scores.npz')
            if r.get('tag') or not z.exists() or r['model'] not in kinds or r['frac'] not in WFRACS or r['seed'] not in a.seeds:
                continue
            runs.append((r['model'], r['frac'], r['seed'], np.load(z)))
    te = runs[0][3]['test_idx']
    assert all((r[3]['test_idx'] == te).all() for r in runs)
    y = df.label_bin.values[te].astype(int); grp = df.group.values[te]
    ug, inv = np.unique(grp, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(len(ug))]
    rng = np.random.RandomState(0)
    boots = [np.concatenate([rows[g] for g in rng.randint(0, len(ug), len(ug))]) for _ in range(a.B)]
    keys = sorted({(m, f) for m, f, _, _ in runs})
    pt, M, ns = {}, {}, {}
    for m, f in keys:
        rs = [z for mm, ff, _, z in runs if (mm, ff) == (m, f)]
        ns[(m, f)] = len(rs); pt[(m, f)] = np.mean([auc(y, z['test']) for z in rs])
        M[(m, f)] = np.mean([[auc(y[b], z['test'][b]) for b in boots] for z in rs], axis=0)
    ci = pd.DataFrame([dict(model=m, frac=f, n_seeds=ns[(m, f)], auc=round(pt[(m, f)], 3),
                            ci_lo=round(np.nanpercentile(M[(m, f)], 2.5), 3), ci_hi=round(np.nanpercentile(M[(m, f)], 97.5), 3))
                       for m, f in keys]).sort_values(['frac', 'model'], ascending=[False, True])
    ci.to_csv(Path(a.out) / 'wgan_auc_ci.csv', index=False)
    print('\n== AUC на тесті (сіди', a.seeds, ') з 95% bootstrap-інтервалом за ураженнями ==\n', ci.to_string(index=False))
    pairs = [('wdisc_ft', 'wdisc_scratch'), ('wdisc_fe', 'wdisc_scratch'), ('wdisc_ft', 'disc_ft'),
             ('wdisc_fe', 'disc_fe'), ('wdisc_scratch', 'disc_scratch')]
    pr = []
    for f in WFRACS:
        for x, z in pairs:
            if (x, f) in M and (z, f) in M:
                d = M[(x, f)] - M[(z, f)]; lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pr.append(dict(frac=f, comparison=f'{x} - {z}', diff=round(pt[(x, f)] - pt[(z, f)], 3), ci_lo=round(lo, 3),
                               ci_hi=round(hi, 3), excludes_zero=bool(lo > 0 or hi < 0)))
    pr = pd.DataFrame(pr); pr.to_csv(Path(a.out) / 'wgan_paired_diffs.csv', index=False)
    print('\n== Парні різниці AUC ==\n', pr.to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in a.seeds:
        fw = Path(a.out) / 'pretrain' / f'wgan_s{s}.json'
        if fw.exists():
            h = pd.DataFrame(json.load(open(fw))['history'])
            ax[0].plot(h.epoch, h.w_est, label=f'W-оцінка, seed {s}'); ax[1].plot(h.epoch, h.loss_G, label=f'loss G, seed {s}')
    ax[0].set_title('WGAN-GP: оцінка відстані Вассерштейна'); ax[1].set_title('WGAN-GP: втрата генератора')
    for x in ax:
        x.set_xlabel('Епоха'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(a.out) / 'wgan_training.png', dpi=150)
    print('\nЗбережено: wgan_auc_ci.csv, wgan_paired_diffs.csv, wgan_training.png у', a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'pretrain', 'all', 'analyze'])
    p.add_argument('--csv', default=str(HERE / 'split_isic2019.csv'))
    p.add_argument('--cache', default=str(HERE.parent / 'cache'))
    p.add_argument('--out', default=str(HERE.parent / 'results_wgan'))
    p.add_argument('--main', default=str(HERE.parent / 'results'), help='папка основного експерименту (для порівняння з DCGAN)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=150)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--B', type=int, default=1000)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--save_model', action='store_true')
    p.add_argument('--size', type=int, default=64, choices=[64])
    a = p.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Пристрій:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if a.mode == 'analyze':
        return analyze(a)
    if a.mode == 'smoke':
        tmp = tempfile.mkdtemp(prefix='wgan_smoke_')
        df = exp.make_fake(tmp)
        a.out = str(Path(tmp) / 'results_wgan'); a.epochs = 1; a.iters = 20; a.eval_every = 10; a.bs = 32; a.seeds = [0]
        data = Data(tmp, 64, dev)
        run_all(a, df, data, dev)
        print('\nSMOKE OK: WGAN-GP конвеєр працює (результати штучні, це лише перевірка коду).')
        return
    df = pd.read_csv(a.csv)
    data = Data(a.cache, a.size, dev)
    assert data.n == len(df)
    if a.mode == 'pretrain':
        pretrain_wgan(a, df, data, dev)
    else:
        run_all(a, df, data, dev)


if __name__ == '__main__':
    main()
