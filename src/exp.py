"""exp.py: дискримінатор DCGAN як класифікатор проти ResNet18 при різній частці міток (ISIC-2019, бінарна задача).

Команди:
  python exp.py smoke                        швидка перевірка конвеєра на штучних даних (кілька хвилин)
  python exp.py pretrain --seed 0            змагальне передтренування DCGAN на train-зображеннях без міток
  python exp.py train --model resnet_imagenet --frac 100 --seed 0
  python exp.py all --seeds 0 1 2            усі моделі x частки міток x сіди (пропускає вже готові)
  python exp.py train --model resnet_imagenet --frac 100 --seed 0 --tag demo --save_model   зберегти модель для app.py
  python exp.py summary                      зведена таблиця і графік (середнє +- std за сідами)
Моделі: disc_scratch | disc_ft (передтренований, донавчання) | disc_fe (передтренований, заморожений) |
        resnet_scratch | resnet_imagenet
Еталон 224 px: python exp.py train --model resnet_imagenet --frac 100 --seed 0 --size 224 --bs 32 --amp --tag ref224
"""
import argparse, json, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
MODELS = ['disc_scratch', 'disc_ft', 'disc_fe', 'resnet_scratch', 'resnet_imagenet']
FRACS = [100, 10, 1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- дані
class Data:
    """Кеш зображень: для 64 px лежить цілком на GPU (uint8), для 224 px читається з диска (memmap)."""
    def __init__(self, cache, size, dev):
        arr = np.load(Path(cache) / f'isic2019_{size}.npy', mmap_mode='r')
        self.gpu = size <= 128
        self.dev = dev
        self.t = torch.from_numpy(np.ascontiguousarray(arr)).to(dev) if self.gpu else None
        self.m = None if self.gpu else arr
        self.n = len(arr)

    def get(self, idx):
        idx = np.sort(idx)
        if self.gpu:
            return self.t[torch.as_tensor(idx, device=self.dev)]
        return torch.from_numpy(np.ascontiguousarray(self.m[idx])).to(self.dev)


def augment(x):
    """Випадкові симетрії квадрата (флипи + транспонування) на рівні окремих зображень."""
    B = x.size(0)
    r = lambda: (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)
    x = torch.where(r(), x.flip(3), x)
    x = torch.where(r(), x.flip(2), x)
    return torch.where(r(), x.transpose(2, 3), x)


def batch(data, idx, kind, aug):
    x = data.get(idx).permute(0, 3, 1, 2).float() / 255.0
    if aug:
        x = augment(x)
    if kind.startswith('resnet'):
        return (x - MEAN.to(x.device)) / STD.to(x.device)
    return x * 2 - 1


# ---------------------------------------------------------------- моделі
def winit(m):
    n = m.__class__.__name__
    if 'Conv' in n:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif 'BatchNorm' in n:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """Генератор DCGAN: 1x1 -> 4x4 -> ... -> size x size. Для size=64 збігається з початковою архітектурою."""
    def __init__(s, nz=100, ngf=64, nc=3, size=64):
        super().__init__()
        c = ngf * (size // 8)
        L = [nn.ConvTranspose2d(nz, c, 4, 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
        while c > ngf:
            L += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        L += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()]
        s.main = nn.Sequential(*L)

    def forward(s, z):
        return s.main(z)


def feat_ch(size=64, ndf=64):
    return ndf * (size // 8)          # 64 -> 512, 128 -> 1024; просторовий розмір карти завжди 4x4


def disc_backbone(nc=3, ndf=64, size=64):  # size x size -> feat_ch x 4 x 4
    L = [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
    c, r = ndf, size // 2
    while r > 4:
        L += [nn.Conv2d(c, c * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, True)]
        c *= 2; r //= 2
    return nn.Sequential(*L)


class DiscGAN(nn.Module):  # режим GAN: логіт «справжнє / згенероване»
    def __init__(s, size=64):
        super().__init__()
        s.backbone = disc_backbone(size=size); s.head = nn.Conv2d(feat_ch(size), 1, 4, 1, 0, bias=False)

    def forward(s, x):
        return s.head(s.backbone(x)).view(-1)


class DiscCls(nn.Module):  # режим класифікації: логіт «злоякісне»
    def __init__(s, freeze=False, size=64):
        super().__init__()
        s.backbone = disc_backbone(size=size); s.freeze = freeze
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(feat_ch(size) * 4 * 4, 1))

    def forward(s, x):
        if s.freeze:
            with torch.no_grad():
                f = s.backbone(x)
        else:
            f = s.backbone(x)
        return s.head(f).view(-1)


def build_model(kind, ckpt, dev, smoke=False, size=64):
    if kind in ('resnet_scratch', 'resnet_imagenet'):
        w = None
        if kind == 'resnet_imagenet':
            try:
                w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                m = torchvision.models.resnet18(weights=w)
            except Exception as e:
                if not smoke:
                    raise
                print('  [smoke] ваги ImageNet недоступні, беру випадкові:', type(e).__name__)
                m = torchvision.models.resnet18(weights=None)
        else:
            m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(512, 1)
        m.forward_orig = m.forward
        m.forward = lambda x, _m=m: _m.forward_orig(x).view(-1)
        return m.to(dev)
    m = DiscCls(freeze=(kind == 'disc_fe'), size=size)
    m.apply(winit)
    if kind in ('disc_ft', 'disc_fe'):
        sd = torch.load(ckpt, map_location='cpu')
        m.backbone.load_state_dict(sd['D_backbone'])
        if kind == 'disc_fe':
            for p in m.backbone.parameters():
                p.requires_grad = False
    return m.to(dev)


# ---------------------------------------------------------------- передтренування GAN
def pretrain(a, df, data, dev):
    out = Path(a.out) / 'pretrain'; out.mkdir(parents=True, exist_ok=True)
    ck = out / f'gan_s{a.seed}.pt'
    torch.manual_seed(a.seed)
    tr = np.where(df.split.values == 'train')[0]          # всі train-зображення, МІТКИ НЕ ВИКОРИСТОВУЮТЬСЯ
    G, D = Generator(size=a.size).to(dev), DiscGAN(size=a.size).to(dev)
    G.apply(winit); D.apply(winit)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed = torch.randn(64, 100, 1, 1, device=dev)
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        perm = np.random.RandomState(a.seed * 1000 + ep).permutation(tr)
        nb = max(1, len(perm) // a.bs); s = np.zeros(4)
        for b in range(nb):
            real = batch(data, perm[b * a.bs:(b + 1) * a.bs], 'disc', True)
            n = real.size(0)
            oD.zero_grad()
            o_r = D(real); l_r = bce(o_r, torch.full_like(o_r, 0.9))      # згладжування мітки 0.9
            fake = G(torch.randn(n, 100, 1, 1, device=dev))
            o_f = D(fake.detach()); l_f = bce(o_f, torch.zeros_like(o_f))
            (l_r + l_f).backward(); oD.step()
            oG.zero_grad()
            o_g = D(fake); l_g = bce(o_g, torch.ones_like(o_g)); l_g.backward(); oG.step()
            s += [(l_r + l_f).item(), l_g.item(), torch.sigmoid(o_r).mean().item(), torch.sigmoid(o_f).mean().item()]
        s /= nb
        hist.append(dict(epoch=ep + 1, loss_D=s[0], loss_G=s[1], D_real=s[2], D_fake=s[3]))
        print(f'  [pretrain s{a.seed}] ep {ep + 1}/{a.epochs} lossD {s[0]:.3f} lossG {s[1]:.3f} D(x) {s[2]:.2f} D(G(z)) {s[3]:.2f}', flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            G.eval()
            with torch.no_grad():
                torchvision.utils.save_image(G(fixed).cpu(), out / f'samples_s{a.seed}_ep{ep + 1}.png', normalize=True, nrow=8)
            G.train()
    torch.save({'D_backbone': D.backbone.state_dict(), 'G': G.state_dict(), 'epochs': a.epochs}, ck)
    json.dump(dict(seed=a.seed, epochs=a.epochs, seconds=time.time() - t0, history=hist), open(out / f'gan_s{a.seed}.json', 'w'), indent=1)
    print(f'  [pretrain] збережено {ck}  ({time.time() - t0:.0f} с)')
    return ck


# ---------------------------------------------------------------- класифікація
@torch.no_grad()
def predict(m, data, idx, kind, bs, amp):
    m.eval(); out = []
    for i in range(0, len(idx), bs):
        x = batch(data, idx[i:i + bs], kind, False)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(m(x).float().cpu().numpy())
    return np.concatenate(out)


def conf_at(y, s, t):
    p = s >= t
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return dict(threshold=float(t), sensitivity=sens, specificity=spec, precision=prec,
                f1=2 * prec * sens / max(prec + sens, 1e-9), balanced_acc=(sens + spec) / 2,
                accuracy=(tp + tn) / len(y), pred_malignant_share=float(p.mean()), tp=tp, fp=fp, tn=tn, fn=fn)


def all_metrics(yv, sv, yt, st, src_t):
    fpr, tpr, thr = roc_curve(yv, sv)
    t_y = thr[int(np.argmax(tpr - fpr))]                         # поріг за валідацією (Youden)
    t_90 = thr[np.where(fpr <= 0.10)[0][-1]]                      # поріг за валідацією: специфічність ~90%
    by = {}
    for k in np.unique(src_t):
        mk = src_t == k
        if len(np.unique(yt[mk])) == 2:
            by[str(k)] = float(roc_auc_score(yt[mk], st[mk]))
    return dict(test_auc=float(roc_auc_score(yt, st)), val_auc=float(roc_auc_score(yv, sv)),
                youden=conf_at(yt, st, t_y), spec90=conf_at(yt, st, t_90), auc_by_source=by)


def train_cls(a, df, data, dev, kind, frac, seed, ckpt=None, smoke=False):
    tag = f'{kind}_f{frac}_s{seed}' + (f'_{a.tag}' if a.tag else '')
    runs = Path(a.out) / 'runs'; runs.mkdir(parents=True, exist_ok=True)
    fn = runs / f'{tag}.json'
    if fn.exists():
        print(f'  пропуск (вже є): {tag}'); return
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    y = df.label_bin.values.astype(np.float32)
    sp = df.split.values
    lab = np.where(sp == 'train')[0] if frac == 100 else np.where(df[f'lab{frac}_s{seed}'].values == 1)[0]
    iv, it = np.where(sp == 'val')[0], np.where(sp == 'test')[0]
    npos = y[lab].sum(); pw = float((len(lab) - npos) / max(npos, 1))
    m = build_model(kind, ckpt, dev, smoke, a.size)
    lr = a.lr or (1e-3 if kind == 'disc_fe' else 3e-4)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    bs = min(a.bs, len(lab)); amp = a.amp and dev.type == 'cuda'
    best, best_it, best_sd, hist, t0 = -1, 0, None, [], time.time()
    for i in range(1, a.iters + 1):
        m.train()
        if kind == 'disc_fe':
            m.backbone.eval()
        idx = np.sort(rng.choice(lab, bs, replace=False))
        x = batch(data, idx, kind, True); t = torch.from_numpy(y[idx]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            loss = lossf(m(x).float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if i % a.eval_every == 0 or i == a.iters:
            sv = predict(m, data, iv, kind, 256, amp); auc = roc_auc_score(y[iv], sv)
            hist.append(dict(it=i, train_loss=loss.item(), val_auc=float(auc)))
            if auc > best:
                best, best_it = auc, i; best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            print(f'  [{tag}] it {i}/{a.iters} loss {loss.item():.3f} val AUC {auc:.3f} (best {best:.3f} @ {best_it})', flush=True)
    m.load_state_dict(best_sd)
    sv, st = predict(m, data, iv, kind, 256, amp), predict(m, data, it, kind, 256, amp)
    res = all_metrics(y[iv], sv, y[it], st, df.source.values[it])
    np.savez_compressed(runs / f'{tag}_scores.npz', val=sv.astype(np.float32), test=st.astype(np.float32), val_idx=iv, test_idx=it)
    res.update(model=kind, frac=frac, seed=seed, tag=a.tag, n_labeled=int(len(lab)), pos_weight=pw, lr=lr, iters=a.iters,
               bs=bs, size=a.size, best_it=best_it, seconds=time.time() - t0, history=hist)
    json.dump(res, open(fn, 'w'), indent=1)
    if getattr(a, 'save_model', False):
        md = Path(a.out) / 'models'; md.mkdir(parents=True, exist_ok=True)
        torch.save(dict(kind=kind, size=a.size, state_dict=best_sd, thr_youden=res['youden']['threshold'],
                        thr_spec90=res['spec90']['threshold'], test_auc=res['test_auc'], val_auc=res['val_auc'],
                        youden=res['youden'], spec90=res['spec90'], frac=frac, seed=seed), md / f'{tag}.pt')
        print(f'  модель збережено: {md / (tag + ".pt")}')
    print(f'  => {tag}: TEST AUC {res["test_auc"]:.3f} | bAcc {res["youden"]["balanced_acc"]:.3f} | '
          f'sens@spec90 {res["spec90"]["sensitivity"]:.3f} | {res["seconds"]:.0f} с')


def run_all(a, df, data, dev, smoke=False):
    for seed in a.seeds:
        ck = Path(a.out) / 'pretrain' / f'gan_s{seed}.pt'
        a.seed = seed
        if any(k.startswith('disc_f') for k in a.models) and not ck.exists():
            pretrain(a, df, data, dev)
        for frac in a.fracs:
            for kind in a.models:
                train_cls(a, df, data, dev, kind, frac, seed, ck, smoke)


# ---------------------------------------------------------------- зведення
def summary(a):
    rows = []
    for f in sorted((Path(a.out) / 'runs').glob('*.json')):
        r = json.load(open(f))
        if r.get('tag'):
            continue
        rows.append(dict(model=r['model'], frac=r['frac'], seed=r['seed'], auc=r['test_auc'],
                         bacc=r['youden']['balanced_acc'], sens90=r['spec90']['sensitivity'],
                         sens=r['youden']['sensitivity'], spec=r['youden']['specificity'], f1=r['youden']['f1']))
    if not rows:
        sys.exit('Немає результатів у ' + str(Path(a.out) / 'runs'))
    d = pd.DataFrame(rows)
    g = d.groupby(['model', 'frac']).agg(['mean', 'std', 'count'])
    out = pd.DataFrame({'n_seeds': g[('auc', 'count')]})
    for c in ('auc', 'bacc', 'sens90', 'sens', 'spec', 'f1'):
        out[c] = g[(c, 'mean')].round(3).astype(str) + ' ± ' + g[(c, 'std')].fillna(0).round(3).astype(str)
    out = out.reset_index().sort_values(['frac', 'model'], ascending=[False, True])
    out.to_csv(Path(a.out) / 'summary.csv', index=False)
    print(out.to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, s in d.groupby('model'):
        mm = s.groupby('frac').auc.agg(['mean', 'std']).reindex(FRACS)
        ax.errorbar(range(len(FRACS)), mm['mean'], yerr=mm['std'].fillna(0), marker='o', capsize=3, label=k)
    ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f'{f}%' for f in FRACS])
    ax.set_xlabel('Частка розмічених даних'); ax.set_ylabel('Test ROC-AUC'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(a.out) / 'auc_vs_labels.png', dpi=150)
    print('Збережено summary.csv та auc_vs_labels.png у', a.out)


# ---------------------------------------------------------------- smoke-тест
def make_fake(tmp, n=640):
    rng = np.random.RandomState(0)
    y = (rng.rand(n) < 0.4).astype(int)
    sp = np.array(['train'] * int(n * .7) + ['val'] * int(n * .15) + ['test'] * (n - int(n * .7) - int(n * .15)))
    df = pd.DataFrame(dict(image=[f'f{i}' for i in range(n)], label_bin=y, class8='NV',
                           source=rng.choice(['BCN', 'HAM'], n), group=[f'g{i}' for i in range(n)], split=sp))
    tr = np.where(sp == 'train')[0]
    for s in range(3):
        r = np.random.RandomState(s)
        a10 = r.choice(tr, int(len(tr) * .1), replace=False)
        a1 = np.concatenate([r.choice(tr[y[tr] == 1], 6, replace=False), r.choice(tr[y[tr] == 0], 6, replace=False)])
        df[f'lab10_s{s}'] = 0; df.loc[a10, f'lab10_s{s}'] = 1
        df[f'lab1_s{s}'] = 0; df.loc[a1, f'lab1_s{s}'] = 1
    base = (rng.rand(n, 64, 64, 3) * 200).astype(np.uint8)
    base[y == 1, :32] += 40                                    # слабкий сигнал, щоб AUC відхилявся від 0.5
    np.save(Path(tmp) / 'isic2019_64.npy', base)
    df.to_csv(Path(tmp) / 'fake.csv', index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'pretrain', 'train', 'all', 'summary'])
    p.add_argument('--csv', default=str(HERE / 'split_isic2019.csv'))
    p.add_argument('--cache', default=str(HERE.parent / 'cache'))
    p.add_argument('--out', default=str(HERE.parent / 'results'))
    p.add_argument('--model', default='resnet_imagenet', choices=MODELS)
    p.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    p.add_argument('--frac', type=int, default=100, choices=FRACS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--size', type=int, default=64, choices=[64, 128, 224])
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=150)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30, help='епохи GAN-передтренування')
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--fracs', type=int, nargs='+', default=FRACS, choices=FRACS, help='частки міток для команди all')
    p.add_argument('--save_model', action='store_true', help='зберегти вагу найкращої контрольної точки і пороги (results/models/<tag>.pt)')
    a = p.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Пристрій:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if a.mode == 'summary':
        return summary(a)
    if a.mode == 'smoke':
        tmp = tempfile.mkdtemp(prefix='smoke_')
        df = make_fake(tmp)
        a.out = str(Path(tmp) / 'results'); a.epochs = 1; a.iters = 20; a.eval_every = 10; a.bs = 32; a.seeds = [0]
        data = Data(tmp, 64, dev)
        run_all(a, df, data, dev, smoke=True)
        summary(a)
        print('\nSMOKE OK: конвеєр працює (результати тут штучні, це лише перевірка коду).')
        return
    df = pd.read_csv(a.csv)
    data = Data(a.cache, a.size, dev)
    assert data.n == len(df), 'Кеш і CSV мають однакову кількість рядків'
    if a.mode == 'pretrain':
        pretrain(a, df, data, dev)
    elif a.mode == 'train':
        ck = Path(a.out) / 'pretrain' / f'gan_s{a.seed}.pt'
        if a.model in ('disc_ft', 'disc_fe') and not ck.exists():
            sys.exit(f'Спершу запусти: python exp.py pretrain --seed {a.seed}')
        train_cls(a, df, data, dev, a.model, a.frac, a.seed, ck)
    else:
        run_all(a, df, data, dev)


if __name__ == '__main__':
    main()
