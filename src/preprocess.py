"""Одноразова підготовка даних: читає JPEG з папки (рекурсивно), робить квадратні кропи 64 та 224 px
(зменшення за коротшою стороною + центральний кроп) і зберігає кеш .npy у порядку рядків split_isic2019.csv.
Запуск:  python preprocess.py  <папка_із_зображеннями>  <папка_для_кешу>
Приклад: python preprocess.py D:\\isic\\ISIC_2019_Training_Input C:\\mag\\cache"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

SIZES = (64, 224)   # можна задати іншу множину третім аргументом: python preprocess.py <img_dir> <out_dir> 128

def square(im, s):
    w, h = im.size; k = s / min(w, h)
    im = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.LANCZOS)
    w, h = im.size; l, t = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.uint8)

def main(img_dir, out_dir, sizes=SIZES):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv('split_isic2019.csv')
    files = {p.stem: p for p in Path(img_dir).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    missing = [i for i in df.image if i not in files]
    print(f'У CSV {len(df)} зображень; знайдено файлів {len(files)}; не знайдено {len(missing)}')
    if missing:
        print('Приклади відсутніх:', missing[:10]); sys.exit('Зупинка: не всі зображення знайдено. Перевір шлях/завантаження.')
    arrs = {s: np.lib.format.open_memmap(out / f'isic2019_{s}.npy', mode='w+', dtype=np.uint8, shape=(len(df), s, s, 3)) for s in sizes}
    for n, name in enumerate(tqdm(df.image, desc='Обробка')):
        im = Image.open(files[name]).convert('RGB')
        for s in sizes: arrs[s][n] = square(im, s)
    for a in arrs.values(): a.flush()
    df[['image']].to_csv(out / 'order.csv', index=False)
    print('Готово. Кеш у', out)

if __name__ == '__main__':
    if len(sys.argv) not in (3, 4): sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], tuple(int(x) for x in sys.argv[3:]) or SIZES)
