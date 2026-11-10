"""Швидка перевірка форм тензорів для 64 та 128 px (без даних). Запуск: python check128.py"""
import torch
from exp import Generator, DiscGAN, DiscCls
for size in (64, 128):
    G, D, C = Generator(size=size), DiscGAN(size=size), DiscCls(size=size)
    z = torch.randn(2, 100, 1, 1)
    x = G(z)
    assert x.shape == (2, 3, size, size), x.shape
    assert D(x).shape == (2,), D(x).shape
    assert C(x).shape == (2,), C(x).shape
    nb = sum(p.numel() for p in D.backbone.parameters()) / 1e6
    ng = sum(p.numel() for p in G.parameters()) / 1e6
    print(f'size {size}: G {tuple(x.shape)}, параметри backbone D = {nb:.2f} млн, G = {ng:.2f} млн ... OK')
