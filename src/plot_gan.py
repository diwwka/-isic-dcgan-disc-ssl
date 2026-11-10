"""Криві втрат DCGAN за епохами для всіх seed: results/pretrain/gan_s*.json -> gan_loss.png"""
import json, glob, sys, os
res = sys.argv[1] if len(sys.argv) > 1 else next((p for p in ('results', os.path.join('..', 'results')) if os.path.isdir(p)), 'results')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for f in sorted(glob.glob(os.path.join(res, 'pretrain', 'gan_s*.json'))):
    h = json.load(open(f)); s = h['seed']; e = [x['epoch'] for x in h['history']]
    ax[0].plot(e, [x['loss_D'] for x in h['history']], label=f'seed {s}')
    ax[1].plot(e, [x['loss_G'] for x in h['history']], label=f'seed {s}')
    ax[2].plot(e, [x['D_real'] for x in h['history']], '-', label=f'D(x), seed {s}')
    ax[2].plot(e, [x['D_fake'] for x in h['history']], '--', label=f'D(G(z)), seed {s}')
for a, t in zip(ax, ['Втрата дискримінатора L_D', 'Втрата генератора L_G', 'Середня вихідна ймовірність D']):
    a.set_title(t); a.set_xlabel('Епоха'); a.grid(alpha=.3)
ax[0].axhline(1.386, color='gray', ls=':'); ax[1].axhline(0.693, color='gray', ls=':')
ax[0].legend(fontsize=8); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig('gan_loss.png', dpi=200); print('збережено gan_loss.png')
