# isic-dcgan-disc-ssl

Код до магістерської роботи «Розроблення інтелектуальної системи підтримки медичної діагностики на основі дискримінатора генеративно-змагальної нейронної мережі DCGAN» (Чернівецький національний університет імені Юрія Федьковича, 2026).

Експеримент порівнює дискримінатор DCGAN, передтренований змагально на зображеннях без міток, із ResNet18 у задачі бінарної класифікації (злоякісне / доброякісне) на ISIC-2019 за різної частки розмічених даних (100 %, 10 %, 1 %).

## Дані

ISIC-2019 Training (25 331 зображення, 8 класів) завантажується окремо з <https://challenge.isic-archive.com/data/#2019>. Ліцензія даних CC BY-NC, тому **зображення та кеші в репозиторій не входять**. Злоякісні класи: MEL, BCC, AK, SCC.

Розбиття: `StratifiedGroupKFold` (10 фолдів, `random_state=42`) за `lesion_id`; test = фолд 0, val = фолд 1. Файл розбиття `split_isic2019.csv` створює `make_split.py`.

## Структура

```
src/        код (exp.py, analyze.py, make_split.py, preprocess.py, ...)
cache/      кеш зображень isic2019_64.npy тощо (не в git)
results/    основний експеримент 64×64, 5 сідів (runs/*.json, *_scores.npz, auc_ci.csv, paired_diffs.csv, auc_by_source.csv, графіки);
            також 5 прогонів ResNet18 на 224 px (файли з суфіксом _ref224)
results128/ перевірка чутливості на 128×128 (disc_scratch, disc_ft, resnet_imagenet; 100 % і 10 %; 3 сіди)
results_wgan/ WGAN-GP, 30 епох (wdisc_*; 100 % і 10 %; сіди 0–4): runs, pretrain, wgan_auc_ci.csv, wgan_paired_diffs.csv, wgan_training.png
results_wgan150/ WGAN-GP, 150 епох (сіди 0–2), той самий формат
results_dcgan150/ контроль: DCGAN, 150 епох (disc_scratch, disc_ft, disc_fe; 100 % і 10 %; сіди 0–2); dcgan_150_vs_30.csv
```

Шляхи за замовчуванням: `--cache` = `../cache`, `--out` = `../results` (відносно `src/`).

## Середовище

Python 3.14.10, PyTorch 2.15.0+cu134 (CUDA 13.4), torchvision 0.29.2+cu134, GPU NVIDIA GeForce RTX 3060 Laptop (6 ГБ), Windows. Точні версії пакетів: `requirements.txt`.

```
pip install -r requirements.txt
```

## Запуск

```
cd src
python make_split.py            # розбиття за lesion_id
python preprocess.py            # кеш зображень 64 px
python exp.py smoke             # перевірка конвеєра на штучних даних
python exp.py pretrain --seed 0 # змагальне передтренування DCGAN
python exp.py all --seeds 0 1 2 3 4
python exp.py summary
python analyze.py               # довірчі інтервали (кластерний bootstrap), парні різниці

# спрощена перевірка WGAN-GP (гіпотеза H3); exp.py не змінюється
python wgan.py smoke
python wgan.py all --seeds 0 1 2
python wgan.py analyze --seeds 0 1 2 3 4

# довша тривалість (150 епох) і контроль DCGAN-150
python wgan.py all --seeds 0 1 2 --epochs 150 --out ../results_wgan150
python exp.py all --seeds 0 1 2 --models disc_scratch disc_ft disc_fe --fracs 100 10 --epochs 150 --out ../results_dcgan150
python wgan.py analyze --seeds 0 1 2 --out ../results_wgan150 --main ../results_dcgan150
python analyze.py --out ../results_dcgan150
python compare150.py            # DCGAN-150 проти DCGAN-30 (парні різниці)
```

Моделі: `disc_scratch`, `disc_ft`, `disc_fe`, `resnet_scratch`, `resnet_imagenet`; у `wgan.py` додатково `wdisc_scratch`, `wdisc_ft`, `wdisc_fe` (критик WGAN-GP з GroupNorm). Подробиці режимів: докстрінг на початку `exp.py`.

## Результати (test AUC, 64×64, середнє за 5 сідів)

| Модель | 100 % | 10 % | 1 % |
|---|---|---|---|
| resnet_imagenet | 0,879 | 0,837 | 0,762 |
| disc_ft | 0,860 | 0,797 | 0,705 |
| disc_fe | 0,837 | 0,813 | 0,713 |
| disc_scratch | 0,844 | 0,784 | 0,703 |
| resnet_scratch | 0,844 | 0,782 | 0,702 |

Файли `*_scores.npz` містять оцінки моделей на val/test, на них працює `analyze.py` (довірчі інтервали можна відтворити без повторного навчання). Інтервали, парні порівняння та обмеження дослідження наведені в тексті роботи (розділ 3). Це дослідницький код, а не медичний виріб: результати не призначені для клінічного використання.

## Ліцензія

Код: MIT (див. `LICENSE`). Дані ISIC-2019 мають власну ліцензію (CC BY-NC).
