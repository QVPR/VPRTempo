import glob, re
import matplotlib.animation as animation
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import numpy as np
        
def demo(data_dir, query, reference, dist_matrix_seq, GTtol, N, RN, R, P):
    
    plt.style.use('seaborn-v0_8-darkgrid')        # ✨ base style  
    # ── 1.  discover + sort png ────────────────────────────────────────────
    def sorted_pngs(folder_path):
        files = glob.glob(os.path.join(folder_path, '*.png'))
        if not files:
            raise RuntimeError(f"No PNGs in {folder_path}")
        files.sort(key=lambda p: int(re.search(r'-(\d+)\.png$', p).group(1)))
        return files

    q_dir = os.path.join(data_dir, query)
    r_dir = os.path.join(data_dir, reference)
    query_paths, ref_paths = sorted_pngs(q_dir), sorted_pngs(r_dir)

    n_frames = min(len(query_paths), dist_matrix_seq.shape[1])

    # ── 2.  figure (high‑res DPI, nice colours) ────────────────────────────
    fig = plt.figure(figsize=(14, 8), dpi=120, constrained_layout=True,
                    facecolor='#f8f9fa')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
            width_ratios=[1, 1, 1.25], height_ratios=[1, 1])

    ax_q, ax_ref, ax_heat  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_line, ax_rec, ax_pr = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ── 2a. titles / axes tidy‑ups ─────────────────────────────────────────
    for ax in (ax_q, ax_ref):
        ax.axis('off')
    ax_q.set_title('Query',  fontsize=12, weight='bold')
    ax_ref.set_title('Matched Reference', fontsize=12, weight='bold')
    ax_heat.set_title('Similarity Matrix', fontsize=12, weight='bold')

    # ── 3.  distance matrix setup with NaN mask ────────────────────────────
    vmin, vmax = dist_matrix_seq.min(), dist_matrix_seq.max()
    vis_mat = np.full_like(dist_matrix_seq, np.nan, dtype=float)
    im_heat = ax_heat.imshow(vis_mat, aspect='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax)
    ax_heat.set_xlabel('Query idx'); ax_heat.set_ylabel('Reference idx')

    # place to collect scatter markers so blit=False redraws accumulate nicely
    scatter_pts = []

    # ── 4.  static plots (Recall scatter & PR curve) ───────────────────────
    # Recall scatter‑line
    ax_rec.plot(N, RN, marker='o', linestyle='-', linewidth=2,
                color='#2087c2', markersize=6)
    ax_rec.scatter(N, RN, s=50, c='#20c27e', zorder=3, edgecolors='white')
    ax_rec.set_xticks(N); ax_rec.set_ylim(0, 1)
    ax_rec.set_title('Recall @ K', fontsize=12, weight='bold')
    ax_rec.set_ylabel('Recall')

    # PR curve
    # PR curve  ➜  line + scatter
    ax_pr.plot(R, P, linewidth=2, color='#c22088')
    ax_pr.scatter(R, P, s=40, color='#2087c2', zorder=3, edgecolors='white')
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.set_title('PR Curve', fontsize=12, weight='bold')

    # ── 5.  distance‑vector line (auto‑scaled) ─────────────────────────────
    line, = ax_line.plot([], [], lw=2, color='#ffb000')
    ax_line.set_xlim(0, dist_matrix_seq.shape[0] - 1)
    ax_line.set_title('Similarities for Current Query',
                    fontsize=12, weight='bold')
    ax_line.set_xlabel('Reference idx'); ax_line.set_ylabel('Similarity')
    # global distance range for the whole run  ➜  y‑axis never jumps
    d_min, d_max = dist_matrix_seq.min(), dist_matrix_seq.max()
    if np.isclose(d_min, d_max):                      # all‑equal safeguard
        d_max = d_min + 1e-6

    ax_line.set_ylim(d_min * 0.95, d_max * 1.05)      # fixed limits

    # ── 6.  image panels & border rectangle ───────────────────────────────
    blank = np.zeros((80, 80), np.uint8)
    im_q   = ax_q.imshow(blank, cmap='viridis', vmin=0, vmax=255)
    im_ref = ax_ref.imshow(blank, cmap='viridis', vmin=0, vmax=255)
    border = Rectangle((0, 0), 1, 1, transform=ax_ref.transAxes,
                    fill=False, linewidth=4)
    ax_ref.add_patch(border)

    # ── 7.  update function ───────────────────────────────────────────────
    def update(col_idx):
        # query image
        q_img = imageio.imread(query_paths[col_idx])
        im_q.set_data(q_img)

        # select reference = arg‑max on column
        ref_idx = int(np.argmax(dist_matrix_seq[:, col_idx]))
        r_img = imageio.imread(ref_paths[ref_idx])
        im_ref.set_data(r_img)

        # correctness border colour
        correct = bool(GTtol[ref_idx, col_idx])
        border.set_edgecolor('#2ecc71' if correct else '#e74c3c')

        # reveal column in matrix & add marker
        vis_mat[:, col_idx] = dist_matrix_seq[:, col_idx]
        im_heat.set_data(vis_mat)

        col_color = '#2ecc71' if correct else '#e74c3c'
        scatter_pts.append(
            ax_heat.scatter(col_idx, ref_idx, s=50, c=col_color,
                            edgecolors='black', linewidths=0.5, zorder=4)
        )

        # update distance line + y‑axis scaling
        y_vals = dist_matrix_seq[:, col_idx]
        line.set_data(np.arange(len(y_vals)), y_vals)

        return [im_q, im_ref, im_heat, border, line] + scatter_pts

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=250, blit=False, repeat=False
    )
    plt.show()