"""Generate detailed multi-density confusion + head topology figure."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
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
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

SUBJECTS = [f"S{i}" for i in range(1, 15)]
CHANNELS = [9, 21, 32, 66]
DIRS_ZH  = ['右', '下', '左', '上', '中']
DIRS_EN  = ['R', 'D', 'L', 'U', 'C']

def standard_66ch_positions():
    """Return (x, y) for 66 electrodes using standard 10-20 extended layout.
    y>0 = anterior (nose), y<0 = posterior (inion). Rows go front-to-back."""
    pos = []
    rows = [
        # (y_coord, x_positions) — front to back
        # Row 1: Fp (ch 1-3)
        (0.85,  [-0.25, 0.00, 0.25]),
        # Row 2: AF (ch 4-5)
        (0.72,  [-0.30, 0.30]),
        # Row 3: F (ch 6-14)
        (0.55,  [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        # Row 4: FC/FT (ch 15-23)
        (0.35,  [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        # Row 5: C/T (ch 24-32)
        (0.12,  [-0.85, -0.62, -0.40, -0.19, 0.00, 0.19, 0.40, 0.62, 0.85]),
        # Row 5.5: M1 (ch 33)
        (-0.02, [-0.92]),
        # Row 6: CP/TP (ch 34-42)
        (-0.12, [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        # Row 6.5: M2 (ch 43)
        (-0.02, [0.92]),
        # Row 7: P (ch 44-52)
        (-0.35, [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        # Row 8: PO (ch 53-59)
        (-0.55, [-0.65, -0.42, -0.22, 0.00, 0.22, 0.42, 0.65]),
        # Row 8.5: CB1 (ch 60)
        (-0.72, [-0.50]),
        # Row 9: O (ch 61-63)
        (-0.72, [-0.22, 0.00, 0.22]),
        # Row 9.5: CB2 (ch 64)
        (-0.72, [0.50]),
        # Extra ch 65-66 (high-density system extras, near mastoids/occipital)
        (-0.82, [-0.35, 0.35]),
    ]
    for y, xs in rows:
        for x in xs:
            pos.append((x, y))
    return np.array(pos[:66])

# Channel subsets (0-based indices) — from MATLAB code
# 9ch: occipital-parietal focus (indices from MATLAB [22,28,29,30,32:34,60,64] -> 0-based)
CH9_IDX  = [21, 27, 28, 29, 31, 32, 33, 59, 63]
# 21ch: central-parietal-occipital
CH21_IDX = list(range(17, 36)) + [59, 63]
# 32ch: extended parietal-occipital
CH32_IDX = list(range(17, 36)) + list(range(53, 66))
# 66ch: all
CH66_IDX = list(range(66))

SUBSETS = {9: CH9_IDX, 21: CH21_IDX, 32: CH32_IDX, 66: CH66_IDX}


def avg_confusion_5x5(target, ch, window):
    """Load and average 5x5 direction confusion across subjects and frequencies."""
    cms = []
    for s in SUBJECTS:
        p = CONF / f"confusion_{s}_{target}target_{ch}ch_w{window}.npy"
        if p.exists():
            cms.append(np.load(p))
    if not cms:
        return None
    cm = np.mean(cms, axis=0)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    avg_block = np.zeros((5, 5))
    for f in range(target // 5):
        avg_block += cm_norm[f*5:(f+1)*5, f*5:(f+1)*5]
    avg_block /= (target // 5)
    return avg_block


def draw_head(ax, positions, active_idx, ch_count):
    """Draw a schematic head with electrode positions. Nose at top."""
    head = Circle((0, 0), 0.95, fill=False, linewidth=1.5, color='#333333')
    ax.add_patch(head)
    nose_x = [-0.08, 0, 0.08]
    nose_y = [0.93, 1.05, 0.93]
    ax.plot(nose_x, nose_y, color='#333333', linewidth=1.5)
    ax.plot([-0.97, -1.05, -0.97], [0.12, 0, -0.12], color='#333333', linewidth=1)
    ax.plot([0.97, 1.05, 0.97], [0.12, 0, -0.12], color='#333333', linewidth=1)

    for i, (x, y) in enumerate(positions):
        if i in active_idx:
            ax.plot(x, y, 'o', color='#2980b9', markersize=5.5, markeredgecolor='white',
                    markeredgewidth=0.4, zorder=3)
        else:
            ax.plot(x, y, 'o', color='#e0e0e0', markersize=3, markeredgecolor='#ccc',
                    markeredgewidth=0.2, zorder=2)

    n_active = len(active_idx)
    region = {9: '枕-顶区', 21: '中央-顶-枕', 32: '顶-枕扩展', 66: '全头覆盖'}
    ax.text(0, -1.08, region.get(ch_count, ''), ha='center', fontsize=8,
            color='#555555', style='italic')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{ch_count} 导联', fontsize=11, fontweight='bold', pad=4)


def make_detail_figure():
    positions = standard_66ch_positions()

    fig = plt.figure(figsize=(16, 9))

    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 0.9],
                          hspace=0.25, wspace=0.25,
                          left=0.04, right=0.96, top=0.92, bottom=0.04)

    # Summary data
    summary_rows = []
    with open(GRID / "summary.csv") as f:
        for r in csv.DictReader(f):
            summary_rows.append(r)

    for col, ch in enumerate(CHANNELS):
        # Top: 5x5 confusion matrix
        ax_cm = fig.add_subplot(gs[0, col])
        block = avg_confusion_5x5(200, ch, 300)
        if block is None:
            continue

        im = ax_cm.imshow(block, cmap='YlOrRd', interpolation='nearest',
                          vmin=0, vmax=1.0)

        for i in range(5):
            for j in range(5):
                val = block[i, j]
                color = 'white' if val > 0.45 else 'black'
                fontw = 'bold' if i == j else 'normal'
                ax_cm.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=9, color=color, fontweight=fontw)

        ax_cm.set_xticks(range(5))
        ax_cm.set_xticklabels(DIRS_ZH, fontsize=10)
        ax_cm.set_yticks(range(5))
        ax_cm.set_yticklabels(DIRS_ZH, fontsize=10)

        match = [r for r in summary_rows
                 if r['target_set'] == '200' and r['channel_set'] == str(ch)
                 and r['window_ms'] == '300']
        acc = float(match[0]['accuracy']) * 100 if match else 0
        ax_cm.set_title(f'{ch} 导联  (acc={acc:.1f}%)', fontsize=12, fontweight='bold')
        if col == 0:
            ax_cm.set_ylabel('真实方向', fontsize=11)

        # Bottom: head topology
        ax_head = fig.add_subplot(gs[1, col])
        draw_head(ax_head, positions, SUBSETS[ch], ch)

    fig.suptitle('空间方向分辨力 × 导联密度 (200目标, 300ms, 14被试均值)',
                 fontsize=15, fontweight='bold', y=0.98)

    fig.text(0.5, 0.52, '↑ 混淆矩阵 (40频率均值)     ↓ 导联分布示意',
             ha='center', fontsize=10, color='#666666')

    fig.savefig(OUT / "fig9_channel_density_detail.png")
    plt.close()
    print("Fig9 saved:", OUT / "fig9_channel_density_detail.png")


def make_center_analysis_figure():
    """Detailed analysis of Center direction confusion."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for col, ch in enumerate(CHANNELS):
        ax = axes[col]
        block = avg_confusion_5x5(200, ch, 300)
        if block is None:
            continue

        center_row = block[4, :]
        center_col = block[:, 4]

        x = np.arange(5)
        w = 0.3
        bars1 = ax.barh(x + w/2, center_row * 100, w, label='中→X (误判为X)',
                        color='#e74c3c', alpha=0.8)
        bars2 = ax.barh(x - w/2, center_col * 100, w, label='X→中 (误判为中)',
                        color='#3498db', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(DIRS_ZH, fontsize=10)
        ax.set_xlabel('概率 (%)', fontsize=10)
        ax.set_title(f'{ch}ch', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 80)
        ax.grid(True, alpha=0.2, axis='x')

        for i, (v1, v2) in enumerate(zip(center_row * 100, center_col * 100)):
            if v1 > 3:
                ax.text(v1 + 1, i + w/2, f'{v1:.0f}%', va='center', fontsize=8)
            if v2 > 3:
                ax.text(v2 + 1, i - w/2, f'{v2:.0f}%', va='center', fontsize=8)

    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('"中"方向的混淆模式分析 (200目标, 300ms)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "fig10_center_confusion_analysis.png")
    plt.close()
    print("Fig10 saved")


def make_per_subject_heatmap():
    """Per-subject per-direction accuracy heatmap at 66ch."""
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

    fig, ax = plt.subplots(figsize=(9, 0.45 * n_subj + 2.5))

    im = ax.imshow(data * 100, cmap='RdYlGn', aspect='auto',
                   vmin=50, vmax=100, interpolation='nearest')

    for i in range(n_subj):
        for j in range(5):
            val = data[i, j] * 100
            color = 'white' if val < 65 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=9, color=color)

    dir_labels = ['右 (R)', '下 (D)', '左 (L)', '上 (U)', '中 (C)']
    ax.set_xticks(range(5))
    ax.set_xticklabels(dir_labels, fontsize=11)
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels(valid_subjects, fontsize=10)
    ax.set_ylabel('被试', fontsize=12)
    ax.set_title('各被试 × 各方向 分类准确率 (%) — 200目标, 66ch, 300ms', fontsize=13)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('准确率 (%)', fontsize=10)

    dir_means = data.mean(axis=0) * 100
    subj_means = data.mean(axis=1) * 100
    for j in range(5):
        ax.text(j, n_subj + 0.3, f'{dir_means[j]:.1f}%', ha='center', fontsize=10,
                fontweight='bold', color='#2c3e50')
    ax.text(-0.8, n_subj + 0.3, '方向均值:', ha='right', fontsize=9, fontweight='bold')
    for i in range(n_subj):
        ax.text(5.3, i, f'{subj_means[i]:.0f}%', ha='left', va='center',
                fontsize=9, color='#2c3e50')

    fig.savefig(OUT / "fig11_per_subject_direction_accuracy.png")
    plt.close()
    print("Fig11 saved")


if __name__ == '__main__':
    make_detail_figure()
    make_center_analysis_figure()
    make_per_subject_heatmap()
    print("All detail figures done!")
