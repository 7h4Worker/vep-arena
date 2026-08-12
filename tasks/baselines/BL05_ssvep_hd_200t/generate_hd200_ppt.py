"""Generate figures and PPT for HD 200-target SSVEP information theory analysis."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from pathlib import Path
import csv

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ── paths ──
TASK  = Path(__file__).resolve().parent
ARENA = TASK.parents[2]
GRID  = TASK / "results" / "offline_tdca_grid"
CONF  = GRID / "confusions"
OUT   = ARENA / "docs" / "ppt_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

SUBJECTS = [f"S{i}" for i in range(1, 15)]
TARGETS  = [40, 80, 120, 160, 200]
CHANNELS = [9, 21, 32, 66]
WINDOWS  = [100, 200, 300, 400, 500]

# ── helpers ──

def load_summary():
    rows = []
    with open(GRID / "summary.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'target_set': int(r['target_set']),
                'channel_set': int(r['channel_set']),
                'window_ms': int(r['window_ms']),
                'accuracy': float(r['accuracy']),
                'itr_bpm': float(r['itr_bpm']),
            })
    return rows

def load_error_topology():
    rows = []
    with open(GRID / "offline_tdca_200target_66ch_error_topology.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'window_ms': int(r['window_ms']),
                'same_freq': float(r['same_frequency_error_rate']),
                'same_spatial': float(r['same_spatial_slot_error_rate']),
                'neighbor_freq': float(r['adjacent_frequency_error_rate']),
            })
    return rows

def load_best_itr():
    rows = []
    with open(GRID / "offline_tdca_best_itr_by_subject_config.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'subject': r['subject'],
                'target_set': int(r['target_set']),
                'channel_set': int(r['channel_set']),
                'window_ms': int(r['window_ms']),
                'accuracy': float(r['accuracy']),
                'itr_bpm': float(r['itr_bpm']),
            })
    return rows

def avg_confusion(target, ch, window):
    cms = []
    for s in SUBJECTS:
        p = CONF / f"confusion_{s}_{target}target_{ch}ch_w{window}.npy"
        if p.exists():
            cms.append(np.load(p))
    return np.mean(cms, axis=0) if cms else None

def itr_formula(M, P, T_sec):
    if P <= 0 or P >= 1:
        P = np.clip(P, 1e-6, 1 - 1e-6)
    if M <= 1:
        return 0
    bits = np.log2(M) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (M - 1))
    return max(0, bits * 60 / T_sec)

# ── Fig 1: 200-target confusion matrix, 66ch, 300ms (avg) ──

def fig_confusion_200():
    cm = avg_confusion(200, 66, 300)
    if cm is None:
        print("WARN: no confusion data for 200/66/300")
        return
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', interpolation='nearest', vmin=0, vmax=0.15)

    for f in range(1, 40):
        ax.axhline(f * 5 - 0.5, color='red', lw=0.3, alpha=0.5)
        ax.axvline(f * 5 - 0.5, color='red', lw=0.3, alpha=0.5)

    ax.set_xlabel('预测目标', fontsize=12)
    ax.set_ylabel('真实目标', fontsize=12)
    ax.set_title('200目标混淆矩阵 (66ch, 300ms, 14被试均值)', fontsize=13)

    ticks = [0, 39, 79, 119, 159, 199]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t+1) for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t+1) for t in ticks])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('分类概率', fontsize=10)

    ax.text(0.98, 0.02, '红线 = 同频分组边界\n(每5个目标共享一个基频)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))
    fig.savefig(OUT / "fig1_confusion_200target_66ch_300ms.png")
    plt.close()
    print("Fig1 saved")

# ── Fig 2: 9ch vs 66ch confusion comparison ──

def fig_confusion_channel_compare():
    cm9  = avg_confusion(200, 9,  300)
    cm66 = avg_confusion(200, 66, 300)
    if cm9 is None or cm66 is None:
        print("WARN: missing confusion data for channel comparison")
        return

    cm9_n  = cm9  / cm9.sum(axis=1, keepdims=True)
    cm66_n = cm66 / cm66.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [(ax1, cm9_n, '9 导联 (acc=61.6%)'),
                             (ax2, cm66_n, '66 导联 (acc=81.4%)')]:
        im = ax.imshow(data, cmap='Blues', interpolation='nearest', vmin=0, vmax=0.15)
        for f in range(1, 40):
            ax.axhline(f * 5 - 0.5, color='red', lw=0.3, alpha=0.4)
            ax.axvline(f * 5 - 0.5, color='red', lw=0.3, alpha=0.4)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('预测目标', fontsize=11)
        ticks = [0, 49, 99, 149, 199]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t+1) for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t+1) for t in ticks])

    ax1.set_ylabel('真实目标', fontsize=11)
    fig.suptitle('200目标混淆矩阵对比: 低密度 vs 高密度 EEG (300ms)', fontsize=14, y=1.02)
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label='分类概率')
    fig.savefig(OUT / "fig2_confusion_9ch_vs_66ch.png")
    plt.close()
    print("Fig2 saved")

# ── Fig 3: Error topology stacked bar ──

def fig_error_topology():
    topo = load_error_topology()
    windows = [t['window_ms'] for t in topo]
    same_f  = [t['same_freq'] * 100 for t in topo]
    same_s  = [t['same_spatial'] * 100 for t in topo]
    neigh_f = [t['neighbor_freq'] * 100 for t in topo]
    other   = [100 - s - sp - n for s, sp, n in zip(same_f, same_s, neigh_f)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(windows))
    w = 0.55

    b1 = ax.bar(x, same_f,  w, label='同频误差 (空间混淆)', color='#e74c3c')
    b2 = ax.bar(x, neigh_f, w, bottom=same_f, label='邻频误差', color='#f39c12')
    b3 = ax.bar(x, same_s,  w, bottom=[a+b for a,b in zip(same_f, neigh_f)],
                label='同空间误差', color='#3498db')
    bottom3 = [a+b+c for a,b,c in zip(same_f, neigh_f, same_s)]
    b4 = ax.bar(x, other, w, bottom=bottom3, label='其他', color='#95a5a6')

    for i, (sf, win) in enumerate(zip(same_f, windows)):
        ax.text(i, sf / 2, f'{sf:.1f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}ms' for w in windows])
    ax.set_xlabel('时间窗口', fontsize=12)
    ax.set_ylabel('误差占比 (%)', fontsize=12)
    ax.set_title('200目标误差拓扑结构 (66ch)', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105)
    fig.savefig(OUT / "fig3_error_topology.png")
    plt.close()
    print("Fig3 saved")

# ── Fig 4: ITR vs target count with ceiling ──

def fig_itr_vs_targets():
    summary = load_summary()
    best_itr = load_best_itr()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for win_ms in [200, 300]:
        T = win_ms / 1000
        ceiling = [np.log2(M) * 60 / T for M in TARGETS]
        ax.plot(TARGETS, ceiling, '--', alpha=0.4,
                label=f'理论上界 (P=1, {win_ms}ms)', linewidth=1.5)

        itrs = []
        for M in TARGETS:
            match = [r for r in summary
                     if r['target_set']==M and r['channel_set']==66 and r['window_ms']==win_ms]
            itrs.append(match[0]['itr_bpm'] if match else 0)
        ax.plot(TARGETS, itrs, 'o-', linewidth=2,
                label=f'实际均值 (66ch, {win_ms}ms)', markersize=7)

    s1_itrs = []
    for M in TARGETS:
        match = [r for r in best_itr
                 if r['subject']=='S1' and r['target_set']==M and r['channel_set']==66]
        s1_itrs.append(match[0]['itr_bpm'] if match else 0)
    ax.plot(TARGETS, s1_itrs, 's--', linewidth=1.5, color='darkred',
            label='S1最优 (66ch, 200ms)', markersize=6)

    ax.set_xlabel('目标数 M', fontsize=12)
    ax.set_ylabel('ITR (bits/min)', fontsize=12)
    ax.set_title('空分复用的ITR增益: 理论上界 vs 实际', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xticks(TARGETS)
    ax.grid(True, alpha=0.3)

    ax.annotate('空分净增益\n+29% (S1)',
                xy=(160, 586), xytext=(170, 650),
                fontsize=9, ha='center', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    fig.savefig(OUT / "fig4_itr_vs_targets.png")
    plt.close()
    print("Fig4 saved")

# ── Fig 5: Accuracy by channel count ──

def fig_accuracy_by_channels():
    summary = load_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    for i, ch in enumerate(CHANNELS):
        accs = []
        for M in TARGETS:
            match = [r for r in summary
                     if r['target_set']==M and r['channel_set']==ch and r['window_ms']==300]
            accs.append(match[0]['accuracy'] * 100 if match else 0)
        ax1.plot(TARGETS, accs, 'o-', color=colors[i], linewidth=2,
                 label=f'{ch}ch', markersize=7)

    ax1.set_xlabel('目标数', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('导联数 × 目标数: 准确率 (300ms)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xticks(TARGETS)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 105)

    for i, ch in enumerate(CHANNELS):
        itrs = []
        for M in TARGETS:
            match = [r for r in summary
                     if r['target_set']==M and r['channel_set']==ch and r['window_ms']==300]
            itrs.append(match[0]['itr_bpm'] if match else 0)
        ax2.plot(TARGETS, itrs, 's-', color=colors[i], linewidth=2,
                 label=f'{ch}ch', markersize=7)

    ax2.set_xlabel('目标数', fontsize=12)
    ax2.set_ylabel('ITR (bits/min)', fontsize=12)
    ax2.set_title('导联数 × 目标数: ITR (300ms)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xticks(TARGETS)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('高密度EEG对空分复用的影响', fontsize=14, y=1.02)
    fig.savefig(OUT / "fig5_channel_count_effect.png")
    plt.close()
    print("Fig5 saved")

# ── Fig 6: Channel efficiency waterfall ──

def fig_channel_efficiency():
    summary = load_summary()

    fig, ax = plt.subplots(figsize=(8, 5))

    targets_66_200 = []
    ceilings = []
    efficiencies = []
    for M in TARGETS:
        match = [r for r in summary
                 if r['target_set']==M and r['channel_set']==66 and r['window_ms']==200]
        if match:
            actual = match[0]['itr_bpm']
            ceiling = np.log2(M) * 60 / 0.2
            targets_66_200.append(actual)
            ceilings.append(ceiling)
            efficiencies.append(actual / ceiling * 100)

    x = np.arange(len(TARGETS))
    w = 0.35
    ax.bar(x - w/2, ceilings, w, label='理论上界 (P=1)', color='#bdc3c7', alpha=0.7)
    ax.bar(x + w/2, targets_66_200, w, label='实际ITR', color='#2980b9')

    for i, (eff, actual) in enumerate(zip(efficiencies, targets_66_200)):
        ax.text(i + w/2, actual + 30, f'{eff:.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='#2980b9')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{M}目标' for M in TARGETS])
    ax.set_ylabel('ITR (bits/min)', fontsize=12)
    ax.set_title('信道效率: 实际ITR / 理论上界 (66ch, 200ms)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    fig.savefig(OUT / "fig6_channel_efficiency.png")
    plt.close()
    print("Fig6 saved")

# ── Fig 7: Confusion matrix zoomed 5x5 block ──

def fig_confusion_zoom():
    cm = avg_confusion(200, 66, 300)
    if cm is None:
        return
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    block_idx = 5
    sl = slice(block_idx*5, (block_idx+1)*5)
    block = cm_norm[sl, sl]
    im1 = ax1.imshow(block, cmap='Blues', interpolation='nearest', vmin=0, vmax=1.0)
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, f'{block[i,j]:.3f}', ha='center', va='center',
                    fontsize=10, color='white' if block[i,j] > 0.3 else 'black')
    dirs = ['右', '下', '左', '上', '中']
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(dirs)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(dirs)
    freq_hz = 8.0 + block_idx * 0.2
    ax1.set_title(f'同频块内部 (f={freq_hz:.1f}Hz, 66ch)', fontsize=12)
    ax1.set_xlabel('预测方向', fontsize=11)
    ax1.set_ylabel('真实方向', fontsize=11)

    block9 = avg_confusion(200, 9, 300)
    if block9 is not None:
        block9_norm = block9 / block9.sum(axis=1, keepdims=True)
        b9 = block9_norm[sl, sl]
        im2 = ax2.imshow(b9, cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)
        for i in range(5):
            for j in range(5):
                ax2.text(j, i, f'{b9[i,j]:.3f}', ha='center', va='center',
                        fontsize=10, color='white' if b9[i,j] > 0.3 else 'black')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(dirs)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(dirs)
        ax2.set_title(f'同频块内部 (f={freq_hz:.1f}Hz, 9ch)', fontsize=12)
        ax2.set_xlabel('预测方向', fontsize=11)

    fig.suptitle('空间分辨力对比: 同一频率下5方向的分类概率', fontsize=13, y=1.02)
    fig.savefig(OUT / "fig7_confusion_zoom_5x5.png")
    plt.close()
    print("Fig7 saved")

# ── Fig 8: Accuracy across windows for different targets (66ch) ──

def fig_accuracy_windows():
    summary = load_summary()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'40': '#2ecc71', '80': '#3498db', '120': '#9b59b6',
              '160': '#f39c12', '200': '#e74c3c'}

    for M in TARGETS:
        accs = []
        for w in WINDOWS:
            match = [r for r in summary
                     if r['target_set']==M and r['channel_set']==66 and r['window_ms']==w]
            accs.append(match[0]['accuracy'] * 100 if match else 0)
        ax.plot(WINDOWS, accs, 'o-', color=colors[str(M)], linewidth=2,
                label=f'{M}目标', markersize=7)

    ax.set_xlabel('时间窗口 (ms)', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('准确率 vs 窗口长度 (66ch)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(WINDOWS)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 105)
    fig.savefig(OUT / "fig8_accuracy_vs_window.png")
    plt.close()
    print("Fig8 saved")


# ── PPT generation ──

def make_ppt():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)
    BLANK = prs.slide_layouts[6]

    def add_title_slide(title, subtitle=''):
        slide = prs.slides.add_slide(BLANK)
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2.5), Inches(11.7), Inches(2))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        p.alignment = PP_ALIGN.CENTER
        if subtitle:
            p2 = tf.add_paragraph()
            p2.text = subtitle
            p2.font.size = Pt(18)
            p2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            p2.alignment = PP_ALIGN.CENTER
        return slide

    def add_image_slide(title, img_path, notes='', img_top=1.3, img_h=5.5):
        slide = prs.slides.add_slide(BLANK)
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        p.alignment = PP_ALIGN.CENTER

        if Path(img_path).exists():
            from PIL import Image
            im = Image.open(img_path)
            w_px, h_px = im.size
            aspect = w_px / h_px
            img_h_in = img_h
            img_w_in = img_h_in * aspect
            if img_w_in > 12.3:
                img_w_in = 12.3
                img_h_in = img_w_in / aspect
            left = (13.333 - img_w_in) / 2
            slide.shapes.add_picture(str(img_path),
                                      Inches(left), Inches(img_top),
                                      Inches(img_w_in), Inches(img_h_in))

        if notes:
            txBox2 = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.5))
            tf2 = txBox2.text_frame
            p2 = tf2.paragraphs[0]
            p2.text = notes
            p2.font.size = Pt(12)
            p2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            p2.alignment = PP_ALIGN.CENTER
        return slide

    def add_text_slide(title, bullets, col2_bullets=None):
        slide = prs.slides.add_slide(BLANK)
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        p.alignment = PP_ALIGN.CENTER

        col_w = Inches(5.8) if col2_bullets else Inches(11.5)
        txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), col_w, Inches(5.5))
        tf2 = txBox2.text_frame
        tf2.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
            p.text = b
            p.font.size = Pt(18)
            p.space_after = Pt(8)
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

        if col2_bullets:
            txBox3 = slide.shapes.add_textbox(Inches(7.0), Inches(1.5), Inches(5.8), Inches(5.5))
            tf3 = txBox3.text_frame
            tf3.word_wrap = True
            for i, b in enumerate(col2_bullets):
                p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
                p.text = b
                p.font.size = Pt(18)
                p.space_after = Pt(8)
                p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return slide

    def add_table_slide(title, headers, rows, note=''):
        slide = prs.slides.add_slide(BLANK)
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        p.alignment = PP_ALIGN.CENTER

        n_rows = len(rows) + 1
        n_cols = len(headers)
        tbl_w = min(11.5, n_cols * 2.0)
        left = (13.333 - tbl_w) / 2
        table = slide.shapes.add_table(n_rows, n_cols,
                                        Inches(left), Inches(1.5),
                                        Inches(tbl_w), Inches(0.5 * n_rows)).table

        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = h
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(14)
                para.font.bold = True
                para.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
                para.alignment = PP_ALIGN.CENTER
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x2c, 0x3e, 0x50)

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = table.cell(i + 1, j)
                cell.text = str(val)
                for para in cell.text_frame.paragraphs:
                    para.font.size = Pt(13)
                    para.alignment = PP_ALIGN.CENTER
                if i % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0xec, 0xf0, 0xf1)

        if note:
            txBox2 = slide.shapes.add_textbox(Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.6))
            tf2 = txBox2.text_frame
            p2 = tf2.paragraphs[0]
            p2.text = note
            p2.font.size = Pt(12)
            p2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            p2.alignment = PP_ALIGN.CENTER
        return slide

    # ── Slide 1: Title ──
    add_title_slide(
        '高密度 200 目标 SSVEP-BCI\n空分复用信息论分析',
        'VEP Arena 复现结果 · 基于 Ming et al. 2026 公开数据集 · 14 被试 · 1400 配置'
    )

    # ── Slide 2: System framework ──
    add_text_slide(
        '通信系统视角: SSVEP-BCI 信道模型',
        [
            '■ SSVEP-BCI ≈ 离散无记忆信道 (DMC)',
            '',
            '  发送端 (视觉刺激 X ∈ {1,...,M})',
            '  → 信道 (皮层响应 + EEG 采集噪声)',
            '  → 接收机 (空间滤波 + TDCA 分类器)',
            '  → 输出 X̂',
            '',
            '■ 三维编码空间:',
            '  频率: 40 基频 (8.0–15.8Hz, Δ0.2Hz)',
            '  相位: 隐含在 JFPM 编码中',
            '  空间: 5 注视方向 (右/下/左/上/中)',
            '',
            '■ 空分复用 ≡ 通信中的 SDMA',
            '  同一频谱 × 空间正交 → 5× 命令扩展',
        ],
        [
            '■ 理论容量增益:',
            '',
            '  40 目标: log₂(40) = 5.32 bits/trial',
            '  200 目标: log₂(200) = 7.64 bits/trial',
            '  空分增益: +2.32 bits (+44%)',
            '',
            '■ 关键问题:',
            '',
            '  ① 空分的 bits 增益能否抵消准确率损失?',
            '  ② 实际 ITR 与理论上界的差距有多大?',
            '  ③ 空间信道的瓶颈在哪里?',
            '  ④ 高密度 EEG 对空分解码有多关键?',
        ]
    )

    # ── Slide 3: Confusion matrix 200-target ──
    add_image_slide(
        '200目标混淆矩阵形态: 块对角结构',
        OUT / "fig1_confusion_200target_66ch_300ms.png",
        '14被试均值 | 每5个相邻目标共享同一基频 | 红线标记频率分组边界 | 误差集中在同频5×5块内',
        img_h=5.3
    )

    # ── Slide 4: 9ch vs 66ch ──
    add_image_slide(
        '空间分辨力: 低密度 vs 高密度 EEG',
        OUT / "fig2_confusion_9ch_vs_66ch.png",
        '9导联: 空间区分能力弱，混淆弥散 → 准确率 61.6% | 66导联: 空间分辨清晰 → 准确率 81.4%',
        img_h=5.0
    )

    # ── Slide 5: Zoomed 5x5 block ──
    add_image_slide(
        '同频块放大: 5个空间方向的分类概率',
        OUT / "fig7_confusion_zoom_5x5.png",
        '66ch: 对角线突出 = 空间方向可分 | 9ch: 对角线模糊 = 空间信道严重退化',
        img_h=4.8
    )

    # ── Slide 6: Error topology ──
    add_image_slide(
        '误差拓扑结构: 频率信道饱和，空间信道是瓶颈',
        OUT / "fig3_error_topology.png",
        '窗口 ≥ 300ms 时 96%+ 误差为同频不同方向混淆 → 频率解码接近完美，空间解码是系统瓶颈',
        img_h=4.8
    )

    # ── Slide 7: Accuracy vs window ──
    add_image_slide(
        '准确率随窗口长度的增长曲线',
        OUT / "fig8_accuracy_vs_window.png",
        '40目标在200ms已达97.5% | 200目标在500ms仍仅87.2% → 空分目标的准确率恢复速率显著慢于纯频率目标',
        img_h=4.8
    )

    # ── Slide 8: ITR vs targets ──
    add_image_slide(
        'ITR增益分析: 空分复用的收益与代价',
        OUT / "fig4_itr_vs_targets.png",
        'S1: 40→160 目标 ITR +29% (453→586 bpm) | 准确率仅降 3.3pp → 空分编码增益 > 准确率惩罚',
        img_h=4.8
    )

    # ── Slide 9: Channel efficiency ──
    add_image_slide(
        '信道效率: 实际 ITR 仅为理论上界的 17–27%',
        OUT / "fig6_channel_efficiency.png",
        '差距来源: 分类错误、固定窗口(无自适应停止)、无个性化微调、视觉延迟等',
        img_h=4.8
    )

    # ── Slide 10: Channel count effect ──
    add_image_slide(
        '导联数对空分复用的决定性影响',
        OUT / "fig5_channel_count_effect.png",
        '40目标: 9→66ch 仅提升3pp | 200目标: 9→66ch 提升20pp → 高密度EEG是空分编码的物理前提',
        img_h=4.8
    )

    # ── Slide 10b: Multi-density confusion + head topology (BIG) ──
    add_image_slide(
        '导联密度 × 空间方向分辨力',
        OUT / "fig9_big_channel_density.png",
        '9ch→66ch: "中"方向从49%提升至74% | 9→21ch跳跃最大(+12.3pp) | 导联覆盖从枕-顶区扩展到全头',
        img_h=5.5
    )

    # ── Slide 10c: 9ch vs 66ch zoom (BIG) ──
    add_image_slide(
        '空间方向分辨力对比: 低密度 vs 高密度 EEG',
        OUT / "fig12_big_zoom_9vs66.png",
        '9导联: 方向均准仅60% | 66导联: 方向均准81% | "中"方向改善最大(+25pp) | "上"方向始终最优',
        img_h=5.5
    )

    # ── Slide 10d: Per-subject × direction (BIG) ──
    add_image_slide(
        '各被试 × 各方向分类准确率',
        OUT / "fig11_big_subject_direction.png",
        '"上"最优87.3% | "中"最差73.9% | S8的"中"仅59% | 中心注视产生最对称皮层响应, 空间特征最弱',
        img_h=5.8
    )

    # ── Slide 11: Key data table ──
    add_table_slide(
        '核心数据汇总 (66ch, 14被试均值)',
        ['目标数', '最优窗口', '准确率', '实际ITR', '理论上界', '信道效率', '空分增益'],
        [
            ['40',  '200ms', '97.5%', '432 bpm', '1597 bpm', '27.0%', '基线'],
            ['80',  '200ms', '92.5%', '470 bpm', '1897 bpm', '24.8%', '+8.8%'],
            ['120', '200ms', '86.2%', '464 bpm', '2072 bpm', '22.4%', '+7.4%'],
            ['160', '200ms', '82.2%', '461 bpm', '2197 bpm', '21.0%', '+6.7%'],
            ['200', '200ms', '73.3%', '411 bpm', '2293 bpm', '17.9%', '-4.9%'],
        ],
        '空分从40→80目标收益最大 (+38 bpm)，到200目标时准确率惩罚超过编码增益 | S1 最优: 160目标 586 bpm'
    )

    # ── Slide 12: Conclusions ──
    add_text_slide(
        '总结: 空分复用的信息论意义',
        [
            '■ 编码增益',
            '  log₂(5) = 2.32 bits/trial',
            '  ITR 理论上界提升 44%',
            '',
            '■ 实际净增益',
            '  S1: 40→160 目标, ITR +29%',
            '  准确率仅降 3.3pp (99.7→96.5%)',
            '  空分编码增益 > 准确率惩罚',
            '',
            '■ 瓶颈定位',
            '  96%+ 误差 = 同频空间混淆',
            '  频率信道饱和, 空间信道是瓶颈',
        ],
        [
            '■ 高密度 EEG 是物理前提',
            '  9→66ch: 200目标 ITR +46%',
            '  空间滤波器需足够维度分离5方向',
            '',
            '■ 优化方向',
            '  ① 提升空间滤波质量 (比频率辨识更有边际收益)',
            '  ② 自适应停止策略 (回收固定窗口损耗)',
            '  ③ 个性化微调 (online +15%)',
            '  ④ 最优空分阶数: 80–160 目标区间净收益最大',
            '',
            '■ 范式层面',
            '  SSVEP + 空分 ≈ FDMA + SDMA',
            '  高密度EEG提供了空间信道的物理采样基础',
        ]
    )

    out_path = ARENA / "docs" / "hd200_ssvep_analysis_v3.pptx"
    prs.save(str(out_path))
    print(f"\nPPT saved: {out_path}")
    return out_path


if __name__ == '__main__':
    print("Generating figures...")
    fig_confusion_200()
    fig_confusion_channel_compare()
    fig_error_topology()
    fig_itr_vs_targets()
    fig_accuracy_by_channels()
    fig_channel_efficiency()
    fig_confusion_zoom()
    fig_accuracy_windows()
    print("\nGenerating PPT...")
    make_ppt()
    print("Done!")
