"""Large, high-impact version of the channel-density × direction figure."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import csv

ARENA = Path("d:/ProjData/proj_python/vep_arena")
GRID  = ARENA / "tasks/ssvep_hd_200target_tdca_sample/results/offline_tdca_grid"
CONF  = GRID / "confusions"
OUT   = ARENA / "docs/ppt_figures"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 250,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

SUBJECTS = [f"S{i}" for i in range(1, 15)]
DIRS_ZH  = ['右', '下', '左', '上', '中']

cmap_strong = LinearSegmentedColormap.from_list('strong',
    ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
     '#ef3b2c', '#cb181d', '#99000d'], N=256)


def avg_confusion_5x5(target, ch, window):
    cms = []
    for s in SUBJECTS:
        p = CONF / f"confusion_{s}_{target}target_{ch}ch_w{window}.npy"
        if p.exists():
            cms.append(np.load(p))
    if not cms:
        return None
    cm = np.mean(cms, axis=0)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    n_dirs = target // (target // 5) if target >= 5 else 5
    n_freqs = target // 5
    avg_block = np.zeros((5, 5))
    for f in range(n_freqs):
        avg_block += cm_norm[f*5:(f+1)*5, f*5:(f+1)*5]
    avg_block /= n_freqs
    return avg_block


def load_summary():
    rows = []
    with open(GRID / "summary.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def standard_66ch_positions():
    pos = []
    rows = [
        (0.85,  [-0.25, 0.00, 0.25]),
        (0.72,  [-0.30, 0.30]),
        (0.55,  [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        (0.35,  [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        (0.12,  [-0.85, -0.62, -0.40, -0.19, 0.00, 0.19, 0.40, 0.62, 0.85]),
        (-0.02, [-0.92]),
        (-0.12, [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        (-0.02, [0.92]),
        (-0.35, [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        (-0.55, [-0.65, -0.42, -0.22, 0.00, 0.22, 0.42, 0.65]),
        (-0.72, [-0.50]),
        (-0.72, [-0.22, 0.00, 0.22]),
        (-0.72, [0.50]),
        (-0.82, [-0.35, 0.35]),
    ]
    for y, xs in rows:
        for x in xs:
            pos.append((x, y))
    return np.array(pos[:66])


CH9_IDX  = [21, 27, 28, 29, 31, 32, 33, 59, 63]
CH21_IDX = list(range(17, 36)) + [59, 63]
CH32_IDX = list(range(17, 36)) + list(range(53, 66))
CH66_IDX = list(range(66))
SUBSETS = {9: CH9_IDX, 21: CH21_IDX, 32: CH32_IDX, 66: CH66_IDX}


def draw_head_big(ax, positions, active_idx, label):
    head = Circle((0, 0), 0.95, fill=False, linewidth=2, color='#444444')
    ax.add_patch(head)
    ax.plot([-0.08, 0, 0.08], [0.93, 1.06, 0.93], color='#444444', linewidth=2)
    ax.plot([-0.97, -1.06, -0.97], [0.12, 0, -0.12], color='#444444', linewidth=1.2)
    ax.plot([0.97, 1.06, 0.97], [0.12, 0, -0.12], color='#444444', linewidth=1.2)

    for i, (x, y) in enumerate(positions):
        if i in active_idx:
            ax.plot(x, y, 'o', color='#2980b9', markersize=7, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=3)
        else:
            ax.plot(x, y, 'o', color='#e8e8e8', markersize=4, markeredgecolor='#d0d0d0',
                    markeredgewidth=0.3, zorder=2)

    ax.text(0, -1.12, label, ha='center', fontsize=11, color='#444444',
            style='italic')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.18)
    ax.set_aspect('equal')
    ax.axis('off')


def make_big_figure():
    """2×4 layout: top = confusion matrices, bottom = head maps. Much bigger."""
    summary = load_summary()
    positions = standard_66ch_positions()
    channels = [9, 21, 32, 66]
    regions = {9: '枕-顶区', 21: '中央-顶-枕', 32: '顶-枕扩展', 66: '全头覆盖'}

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 0.8],
                          hspace=0.15, wspace=0.22,
                          left=0.04, right=0.96, top=0.91, bottom=0.03)

    accs = {}
    for col, ch in enumerate(channels):
        ax = fig.add_subplot(gs[0, col])
        block = avg_confusion_5x5(200, ch, 300)
        if block is None:
            continue

        match = [r for r in summary
                 if r['target_set'] == '200' and r['channel_set'] == str(ch)
                 and r['window_ms'] == '300']
        acc = float(match[0]['accuracy']) * 100 if match else 0
        accs[ch] = acc

        im = ax.imshow(block, cmap=cmap_strong, interpolation='nearest',
                       vmin=0, vmax=1.0)

        for i in range(5):
            for j in range(5):
                val = block[i, j]
                if i == j:
                    color = 'white' if val > 0.55 else '#222222'
                    ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                            fontsize=16, color=color, fontweight='bold')
                else:
                    if val >= 0.05:
                        ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                                fontsize=11, color='#333333')

        ax.set_xticks(range(5))
        ax.set_xticklabels(DIRS_ZH, fontsize=14)
        ax.set_yticks(range(5))
        ax.set_yticklabels(DIRS_ZH, fontsize=14)
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('#333333')

        title_color = '#c0392b' if ch == 9 else ('#27ae60' if ch == 66 else '#2c3e50')
        ax.set_title(f'{ch} 导联\nacc = {acc:.1f}%', fontsize=16,
                     fontweight='bold', color=title_color, pad=8)

        if col == 0:
            ax.set_ylabel('真实方向', fontsize=14, labelpad=5)

        # Bottom: head
        ax_h = fig.add_subplot(gs[1, col])
        draw_head_big(ax_h, positions, SUBSETS[ch], regions[ch])

    # Arrows between panels
    for col in range(3):
        ch1, ch2 = channels[col], channels[col+1]
        if ch1 in accs and ch2 in accs:
            delta = accs[ch2] - accs[ch1]
            x_pos = 0.04 + (col + 1) * (0.92 / 4) + 0.02
            fig.text(x_pos, 0.58, f'+{delta:.1f}pp →',
                     fontsize=12, fontweight='bold', color='#27ae60',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1',
                               edgecolor='#27ae60', alpha=0.9))

    fig.suptitle('空间方向分辨力 × 导联密度', fontsize=22, fontweight='bold',
                 y=0.97, color='#1a1a2e')
    fig.text(0.5, 0.925, '200目标 · TDCA · 300ms · 14被试均值 · 40频率平均',
             ha='center', fontsize=13, color='#666666')

    fig.savefig(OUT / "fig9_big_channel_density.png")
    plt.close()
    print("fig9_big saved")


def make_big_subject_heatmap():
    """Large per-subject × direction heatmap."""
    valid_subjects = []
    valid_data = []
    for s in SUBJECTS:
        p = CONF / f"confusion_{s}_200target_66ch_w300.npy"
        if not p.exists():
            continue
        cm = np.load(p)
        if cm.sum() == 0:
            continue
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        block = np.zeros((5, 5))
        for f in range(40):
            block += cm_norm[f*5:(f+1)*5, f*5:(f+1)*5]
        block /= 40
        valid_subjects.append(s)
        valid_data.append(np.diag(block))

    data = np.array(valid_data)
    n_subj = len(valid_subjects)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data * 100, cmap='RdYlGn', aspect='auto',
                   vmin=50, vmax=100, interpolation='nearest')

    for i in range(n_subj):
        for j in range(5):
            val = data[i, j] * 100
            color = 'white' if val < 63 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=13, color=color, fontweight='bold')

    dir_labels = ['右 (R)', '下 (D)', '左 (L)', '上 (U)', '中 (C)']
    ax.set_xticks(range(5))
    ax.set_xticklabels(dir_labels, fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels(valid_subjects, fontsize=13)
    ax.set_ylabel('被试', fontsize=14)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.08, aspect=30)
    cbar.set_label('方向准确率 (%)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    dir_means = data.mean(axis=0) * 100
    subj_means = data.mean(axis=1) * 100

    for j in range(5):
        color = '#c0392b' if dir_means[j] == dir_means.min() else (
                '#27ae60' if dir_means[j] == dir_means.max() else '#2c3e50')
        ax.text(j, n_subj + 0.5, f'{dir_means[j]:.1f}%', ha='center',
                fontsize=14, fontweight='bold', color=color)
    ax.text(-0.8, n_subj + 0.5, '均值:', ha='right', fontsize=13, fontweight='bold')

    for i in range(n_subj):
        ax.text(5.5, i, f'{subj_means[i]:.0f}%', ha='center', va='center',
                fontsize=12, color='#2c3e50', fontweight='bold')
    ax.text(5.5, -1.0, '被试\n均值', ha='center', fontsize=10, fontweight='bold',
            color='#2c3e50')

    best_j = np.argmax(dir_means)
    worst_j = np.argmin(dir_means)
    ax.text(best_j, n_subj + 1.2, '▲ 最优', ha='center', fontsize=11,
            color='#27ae60', fontweight='bold')
    ax.text(worst_j, n_subj + 1.2, '▼ 最差', ha='center', fontsize=11,
            color='#c0392b', fontweight='bold')

    ax.set_title('各被试 × 各方向 分类准确率 — 200目标, 66ch, 300ms',
                 fontsize=16, fontweight='bold', pad=30, color='#1a1a2e')

    fig.savefig(OUT / "fig11_big_subject_direction.png")
    plt.close()
    print("fig11_big saved")


def make_big_zoom_comparison():
    """Large zoom of one frequency block: 9ch vs 66ch side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    positions_66 = standard_66ch_positions()
    freq_idx = 5
    freq_hz = 8.0 + freq_idx * 0.2

    for ax, ch, cmap_name, subset_key in [
        (axes[0], 9,  'Reds',  CH9_IDX),
        (axes[1], 66, 'Blues', CH66_IDX),
    ]:
        block_full = avg_confusion_5x5(200, ch, 300)
        if block_full is None:
            continue

        im = ax.imshow(block_full, cmap=cmap_name, interpolation='nearest',
                       vmin=0, vmax=1.0)

        for i in range(5):
            for j in range(5):
                val = block_full[i, j]
                color = 'white' if val > 0.40 else 'black'
                fontw = 'bold' if i == j else 'normal'
                fs = 18 if i == j else 14
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=fs, color=color, fontweight=fontw)

        ax.set_xticks(range(5))
        ax.set_xticklabels(DIRS_ZH, fontsize=15)
        ax.set_yticks(range(5))
        ax.set_yticklabels(DIRS_ZH, fontsize=15)
        ax.tick_params(length=0)

        diag_acc = np.diag(block_full).mean() * 100
        title_color = '#c0392b' if ch == 9 else '#2980b9'
        ax.set_title(f'{ch} 导联   方向均准 {diag_acc:.1f}%',
                     fontsize=17, fontweight='bold', color=title_color, pad=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color(title_color)

    axes[0].set_ylabel('真实方向', fontsize=15)

    fig.text(0.5, 0.02,
             '40频率均值 | 对角线=正确分类 | 9ch"中"仅49% → 66ch"中"提升至74%',
             ha='center', fontsize=12, color='#666666')

    fig.suptitle('空间方向分辨力对比: 低密度 vs 高密度 EEG',
                 fontsize=19, fontweight='bold', y=0.98, color='#1a1a2e')
    fig.text(0.5, 0.93, '200目标 · 300ms · 14被试 × 40频率均值',
             ha='center', fontsize=12, color='#888888')

    fig.savefig(OUT / "fig12_big_zoom_9vs66.png")
    plt.close()
    print("fig12_big saved")


if __name__ == '__main__':
    make_big_figure()
    make_big_subject_heatmap()
    make_big_zoom_comparison()
    print("All big figures done!")
