"""Груповий розподіл ISIC-2019 за lesion_id (train/val/test = 80/10/10), seed=42.
Стратифікація за (джерело x бінарна мітка). Підмножини міток 10% і 1% (вкладені, на рівні груп) для 3 сідів.
Запуск: python make_split.py  (потрібні ISIC_2019_Training_GroundTruth.csv та ISIC_2019_Training_Metadata.csv поруч)"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
MAL=['MEL','BCC','AK','SCC']; CLS=['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
g=pd.read_csv('ISIC_2019_Training_GroundTruth.csv'); m=pd.read_csv('ISIC_2019_Training_Metadata.csv')
df=g.merge(m,on='image')
df['label_bin']=(df[MAL].sum(axis=1)>0).astype(int)           # 1 = malignant (MEL, BCC, AK, SCC)
df['class8']=df[CLS].idxmax(axis=1)
df['source']=df.lesion_id.str.extract(r'^([A-Za-z]+)')[0].fillna('NO_ID')
df['group']=df.lesion_id.fillna(df.image)                      # без lesion_id: кожне зображення = окрема група
strat=df.source+'_'+df.label_bin.astype(str)
folds=[te for _,te in StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42).split(df,strat,df.group)]
df['split']='train'; df.loc[folds[0],'split']='test'; df.loc[folds[1],'split']='val'
assert not (set(df.group[df.split=='train'])&set(df.group[df.split!='train']))
assert not (set(df.group[df.split=='val'])&set(df.group[df.split=='test']))
tr=df[df.split=='train']; groups=tr.group.unique(); sizes=tr.groupby('group').size()
for s in range(5):
    rng=np.random.RandomState(s); perm=rng.permutation(groups); cum=sizes.loc[perm].cumsum().values
    for name,frac in (('lab10',0.10),('lab1',0.01)):
        keep=set(perm[:int(np.searchsorted(cum,frac*len(tr)))+1])
        df[f'{name}_s{s}']=((df.split=='train')&df.group.isin(keep)).astype(int)
cols=['image','label_bin','class8','source','group','split']+[c for c in df if c.startswith(('lab10_','lab1_'))]
df[cols].to_csv('split_isic2019.csv',index=False)
print(df.groupby('split').agg(n=('image','size'),mal=('label_bin','mean'),groups=('group','nunique')).round(3))
for c in [c for c in df if c.startswith(('lab10_','lab1_'))]: print(c,int(df[c].sum()),'mal=%.3f'%df.loc[df[c]==1,'label_bin'].mean())
