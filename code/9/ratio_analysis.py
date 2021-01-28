# coding: utf-8

import pandas as pd, numpy as np
import sys
ds =sys.argv[1]
save_dir = sys.argv[2]
root = "./data/%s/stats"%ds

det_rd_trva = pd.read_csv('%s/RD.csv'%(root))
det_dr_trva = pd.read_csv('%s/DR.csv'%(root))
det_trva = pd.read_csv('%s/D.csv'%(root))
random_trva = pd.read_csv('%s/R.csv'%(root))

ctrs = []

for trva in [random_trva, det_trva, det_rd_trva, det_dr_trva]:
    ctrs.append(trva[[c for c in trva.columns if c.startswith('click')]].values / trva[[c for c in trva.columns if c.startswith('count')]].values)

betas = []
for ctr in ctrs[1:]:
    betas.append(np.log10(ctr/ctrs[0]))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Result Analysis')
ax1.set_xlabel('AD_idx')
ax1.set_ylabel('Sc/St')

det_rd_beta = betas[1]   # rnd->det, position bias
item_num = 100 if 'kk' in ds else 300
ax1.set_xlim(xmax=item_num, xmin=-1)
ax1.set_ylim(ymax=1.5, ymin=-1.5)
det_rd_counts = det_rd_trva[[c for c in det_rd_trva.columns if c.startswith('count')]].values[:item_num, :]
print(np.sum(det_rd_counts), item_num)

stride=1
for i in range(0, det_rd_beta.shape[0],stride):
    xs = []
    ys = []
    ps = []
    ad_idx = i
    for j in range(det_rd_beta.shape[-1]):
        v = det_rd_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            continue
        else:
            if j == np.argmax(det_rd_counts[i, :]):
                ax1.scatter([ad_idx], [v], s=80, c='r', marker='x')
            else:
                xs.append(ad_idx)
                ys.append(v)
    ax1.scatter(xs, ys, s=20, c='k', marker='.')

ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.spines['bottom'].set_position(('data',0))
ax1.xaxis.set_major_locator(plt.MultipleLocator(stride))
ax1.grid(axis="x")
plt.xticks(rotation=30)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

del fig, ax1
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Result Analysis')
ax1.set_xlabel('Position_idx')
ax1.set_ylabel('Sc/St')

det_dr_beta = betas[-1]   # det->rnd, selection bias
pos_num=det_dr_beta.shape[1]
plt.xlim(xmax=pos_num, xmin=-1)
plt.ylim(ymax=2, ymin=-0.5)
plt.axhline(0, linestyle=(45,(55,20)), lw=0.5, color='b')
det_dr_counts = det_dr_trva[[c for c in det_dr_trva.columns if c.startswith('count')]].values[:item_num, :]
print(np.sum(det_dr_counts))

for j in range(0, det_dr_beta.shape[1]):
    xs = []
    ys = []
    pos_idx = j
    for i in range(det_dr_beta.shape[0]):
        v = det_dr_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            continue
        else:
            if i == np.argmax(det_dr_counts[:, j]):
                ax1.scatter([pos_idx], [v], s=80, c='r', marker='x')
            else:
                xs.append(pos_idx)
                ys.append(v)
    ax1.scatter(xs, ys, s=20, c='b', marker='.')
ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.grid(axis="x")

del fig, ax1
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Result Analysis')
ax1.set_xlabel('Dc/Dt')
ax1.set_ylabel('Sc/St')

def get_color(v):
    color = []
    if (v == 1):
        color.append('r')
    elif (v == 2):
        color.append('g')
    elif (v == 3):
        color.append('b')
    elif (v == 4):
        color.append('c')
    elif (v == 5):
        color.append('m')
    elif (v == 6):
        color.append('c')
    elif (v == 7):
        color.append('m')
    elif (v == 8):
        color.append('y')
    elif (v == 9):
        color.append('y')
    else:
        color.append('y')
    return color

det_beta = betas[0]
pos_num=det_beta.shape[1]
plt.xlim(xmax=1, xmin=1e-6)
plt.ylim(ymax=3.5, ymin=-1.5)
plt.axhline(0, linestyle=(45,(55,20)), lw=0.5, color='b')
plt.xscale('log')
det_counts = det_trva[[c for c in det_trva.columns if c.startswith('count')]].values[:item_num, :]
det_alpha = det_counts/(np.sum(det_counts)/pos_num)

xs = []
ys = []
top10_show_ads = np.argsort(np.sum(det_counts, axis=1))[-50:]
for i in range(0, det_beta.shape[0]):
    x1 = []
    y1 = []
    for j in range(0, det_beta.shape[1]):
        v = det_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            continue
        else:
            xs.append(det_alpha[i, j])
            ys.append(v)
            x1.append(det_alpha[i, j])  # pos 0 -> pos 9
            y1.append(v)
ax1.scatter(xs, ys, s=20, c='b', marker='.')

ax1.grid(axis="x")
ax1.invert_xaxis()

del fig, ax1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

fig = plt.figure(figsize=(10,10))
ax1 = Axes3D(fig)
ax1.set_title('%s'%('Placement Bias'), pad=-50, fontsize=20)
ax1.set_xlabel('k', fontsize=16)
ax1.set_ylabel('j', fontsize=16)
ax1.set_zlabel('ρ', fontsize=16)

det_rd_beta = betas[1][:item_num, :]   # rnd->det, position bias
det_rd_counts = det_rd_trva[[c for c in det_rd_trva.columns if c.startswith('count')]].values[:item_num, :]


xs = []
ys = []
zs = []
z = np.zeros((10, item_num)).T
for i in range(0, det_rd_beta.shape[0]):
    xb = []
    yb = []
    zb = []
    ad_idx = i
    for j in range(det_rd_beta.shape[-1]):
        v = det_rd_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            z[i,j] = np.nan
            continue
        else:
            ys.append(ad_idx)
            xs.append(j)
            zs.append(v)
            xb.append(j)
            yb.append(v)
            zb.append(ad_idx)
            z[i,j] = v

ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_major_locator(plt.MultipleLocator(item_num/10))
xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
ax1.scatter(xs[zs>0], ys[zs>0], zs[zs>0], label='ρ>0', c='#61e160', marker='o', alpha=1, s=15)
ax1.scatter(xs[zs<=0], ys[zs<=0], zs[zs<=0], label='ρ≤0', c='#840000', marker='v', alpha=1, s=30)
ax1.legend(bbox_to_anchor=(0.75, 0.8), fontsize=14)

ax1.view_init(azim=-98, elev=11)
ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.savefig("%s/%s_pbias.pdf"%(save_dir, ds), format="pdf", dpi=1000, bbox_inches='tight', pad_inches=-1.1)

del fig, ax1
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))
ax1 = Axes3D(fig)
ax1.set_title('%s'%('Retrieval Bias'), pad=-50, fontsize=20)
ax1.set_xlabel('k', fontsize=16)
ax1.set_ylabel('j', fontsize=16)
ax1.set_zlabel('ρ', fontsize=16)

det_dr_beta = betas[-1][:item_num, :]   # det->rnd, selection bias
pos_num=det_dr_beta.shape[1]
det_dr_counts = det_dr_trva[[c for c in det_dr_trva.columns if c.startswith('count')]].values
print(np.sum(det_dr_counts))

xs = []
ys = []
zs = []
xb = []
yb = []
zb = []
z = np.ones((10, item_num)).T
for j in range(0, det_dr_beta.shape[1]):
    for i in range(det_dr_beta.shape[0]):
        v = det_dr_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            continue
        else:
            if i == np.argmax(det_dr_counts[:, j]):
                ax1.scatter([j], [i], [v], s=80, c='b', marker='')
                xb.append(j)
                yb.append(i)
                zb.append(v)
                z[:, j] *= v
            else:
                xs.append(j)
                ys.append(i)
                zs.append(v)

x, y= np.meshgrid(np.arange(10), np.arange(item_num))
ax1.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.7)#, cmap='Wistia', vmin=-0.5, vmax=0.5)
ax1.plot(xb, zb, zs=item_num, zdir='y', c='b', ls='--')

ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_major_locator(plt.MultipleLocator(item_num/10))
xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
ax1.scatter(xs[zs>0], ys[zs>0], zs[zs>0], label='ρ>0', c='#61e160', marker='o', alpha=1, s=15)
ax1.scatter(xs[zs<=0], ys[zs<=0], zs[zs<=0], label='ρ≤0', c='#840000', marker='v', alpha=1, s=30)
ax1.legend(bbox_to_anchor=(0.75, 0.8), fontsize=14)
ax1.view_init(azim=-98, elev=11)
ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.savefig("%s/%s_rbias.pdf"%(save_dir, ds), format="pdf", dpi=1000, bbox_inches='tight', pad_inches=-1.1)

del fig, ax1
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))
ax1 = Axes3D(fig)
ax1.set_title('%s'%('Retrieval Bias + Placement Bias'), pad=-50, fontsize=20)
ax1.set_xlabel('k', fontsize=16)
ax1.set_ylabel('j', fontsize=16)
ax1.set_zlabel('ρ', fontsize=16)

det_beta = betas[0][:item_num, :]
pos_num=det_beta.shape[1]
det_counts = det_trva[[c for c in det_trva.columns if c.startswith('count')]].values
print(np.sum(det_dr_counts))

xs = []
ys = []
zs = []
for j in range(0, det_beta.shape[1]):
    for i in range(det_beta.shape[0]):
        v = det_beta[i, j]
        if np.isnan(v) or np.isinf(v):
            continue
        else:
            xs.append(j)
            ys.append(i)
            zs.append(v)

ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_major_locator(plt.MultipleLocator(item_num/10))
xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
ax1.scatter(xs[zs>0], ys[zs>0], zs[zs>0], label='ρ>0', c='#61e160', marker='o', alpha=1, s=15)
ax1.scatter(xs[zs<=0], ys[zs<=0], zs[zs<=0], label='ρ≤0', c='#840000', marker='v', alpha=1, s=30)
ax1.legend(bbox_to_anchor=(0.75, 0.8), fontsize=14)#loc='center right')
ax1.view_init(azim=-98, elev=11)
ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.savefig("%s/%s_mbias.pdf"%(save_dir, ds), format="pdf", dpi=1000, bbox_inches='tight', pad_inches=-1.1)
