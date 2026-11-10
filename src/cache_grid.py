import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = pd.read_csv('split_isic2019.csv')
a = np.load(r'A:\MasterThesis\cache\isic2019_64.npy', mmap_mode='r')   # рядки відповідають рядкам CSV
classes = ['NV', 'MEL', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
print('класи в CSV:', sorted(d.class8.unique()))
fig, ax = plt.subplots(len(classes), 6, figsize=(7, 9.5))
rng = np.random.RandomState(0)
for i, c in enumerate(classes):
    idx = rng.choice(np.where(d.class8.values == c)[0], 6, replace=False)
    for j, k in enumerate(np.sort(idx)):
        ax[i, j].imshow(a[k]); ax[i, j].axis('off')
    ax[i, 0].set_title(c, loc='left', fontsize=9)
plt.tight_layout(); plt.savefig('cache_grid.png', dpi=200); print('збережено cache_grid.png')
