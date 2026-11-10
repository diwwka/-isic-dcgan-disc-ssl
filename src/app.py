"""Демонстраційний застосунок: імовірність злоякісності за дерматоскопічним зображенням.
Запуск (з папки code): streamlit run app.py
Потрібна модель: python exp.py train --model resnet_imagenet --frac 100 --seed 0 --tag demo --save_model
ЦЕ ДОСЛІДНИЙ ПРОТОТИП, НЕ МЕДИЧНИЙ ВИРІБ. Результат не є діагнозом."""
import sys, io
from pathlib import Path
import numpy as np, torch, streamlit as st
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp import build_model, DiscCls, MEAN, STD

RES = Path(__file__).resolve().parent.parent / 'results' / 'models'
st.set_page_config(page_title='Скринінг шкірних уражень (прототип)', layout='centered')
st.title('Підтримка діагностики шкірних новоутворень')
st.warning('Дослідний прототип магістерської роботи. Не є медичним виробом і не замінює висновку лікаря.')


@st.cache_resource
def load(path):
    ck = torch.load(path, map_location='cpu')
    # для disc_* створюємо архітектуру без завантаження GAN-ваг: вони вже всередині збереженої моделі
    m = DiscCls() if ck['kind'].startswith('disc') else build_model(ck['kind'], None, torch.device('cpu'), smoke=True)
    m.load_state_dict(ck['state_dict']); m.eval()
    return m, ck


def square(im, s):  # як у preprocess.py: коротша сторона -> s, центральне обрізання
    w, h = im.size; k = s / min(w, h)
    im = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.LANCZOS)
    w, h = im.size; l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s))


files = sorted(RES.glob('*.pt')) if RES.exists() else []
if not files:
    st.error(f'Немає збережених моделей у {RES}. Запусти команду з docstring app.py.'); st.stop()
name = st.sidebar.selectbox('Модель', [f.name for f in files])
m, ck = load(str(RES / name))
rule = st.sidebar.radio('Поріг', ['Юдена (збалансований)', 'Специфічність ≈ 90 %'])
thr = ck['thr_youden'] if rule.startswith('Юдена') else ck['thr_spec90']
mt = ck['youden'] if rule.startswith('Юдена') else ck['spec90']
st.sidebar.markdown(f"**Тестові метрики моделі**  \nAUC {ck['test_auc']:.3f}  \nRecall {mt['sensitivity']:.2f}  \n"
                    f"Specificity {mt['specificity']:.2f}  \nPrecision {mt['precision']:.2f}")
st.sidebar.caption(f"Тип: {ck['kind']}, вхід {ck['size']} px, seed {ck['seed']}, мітки {ck['frac']} %")

up = st.file_uploader('Дерматоскопічне зображення (JPEG/PNG)', type=['jpg', 'jpeg', 'png'])
if up:
    im = Image.open(io.BytesIO(up.read())).convert('RGB')
    s = ck['size']; crop = square(im, s)
    x = torch.from_numpy(np.asarray(crop, dtype=np.uint8)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    x = (x - MEAN) / STD if ck['kind'].startswith('resnet') else x * 2 - 1
    with torch.no_grad():
        logit = m(x).item()
    p = 1 / (1 + np.exp(-logit))
    c1, c2 = st.columns(2)
    c1.image(im, caption='Вхідне зображення', use_container_width=True)
    c2.image(crop.resize((256, 256), Image.NEAREST), caption=f'Вхід моделі ({s}×{s})', use_container_width=True)
    st.metric('Оцінка моделі (сигмоїда логіта)', f'{p:.2f}')
    if logit >= thr:
        st.error('Вище порога: рекомендовано звернутися до дерматолога (підозра на злоякісне ураження).')
    else:
        st.success('Нижче порога: за оцінкою моделі підозра слабша. Це не виключає захворювання.')
    st.caption('Оцінка не є ймовірністю у клінічному сенсі: модель навчена зі зваженою функцією втрат, '
               'тому порівнюйте з порогом, а не з 50 %. Точність обмежена (див. метрики ліворуч).')
