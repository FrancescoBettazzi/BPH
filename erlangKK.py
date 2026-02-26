import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde
from bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

K_VALUES = [2, 4, 8, 16]
M_N_PAIRS = [
    (27, 8), (68, 16), (163, 32),
    (381, 64), (866, 128), (1938, 256)
]

NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

GLOBAL_Y_LIM_PDF = (0, 2.0)
GLOBAL_Y_LIM_KL = (0, 0.6)

# =============================================================================
# 2. COMPARISON LOOP
# =============================================================================

results_table_data = []
today_str = datetime.now().strftime("%Y%m%d") + "_erlangKK"

for K in K_VALUES:
    dist_obj = stats.gamma(a=K, scale=1 / K)
    dist_name = f"Erlang (K={K}, \u03bb={K}, \u03bc=1)"
    dist_string = f"erlang_K{K}"

    print(f"Analisi: {dist_name}")

    for M, N_pdf in M_N_PAIRS:
        kl_bernstein_list = []
        kl_kde_list = []
        plot_runs = []

        upper_lim = dist_obj.ppf(0.999)
        x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)
        pdf_true = dist_obj.pdf(x_eval)

        for i in range(NUM_SIMULATIONS):
            campioni = dist_obj.rvs(size=M)
            ecdf = create_ecdf(campioni)

            pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
            kde_func = gaussian_kde(campioni)
            pdf_kde = kde_func(x_eval)

            kl_bernstein_list.append(entropy(pk=pdf_true, qk=pdf_bern + 1e-12))
            kl_kde_list.append(entropy(pk=pdf_true, qk=pdf_kde + 1e-12))

            if i < N_PLOT_LINES:
                plot_runs.append({'x': x_eval, 'pdf_bern': pdf_bern, 'pdf_kde': pdf_kde})

        # =============================================================================
        # 3. VISUALIZATION
        # =============================================================================

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Comparison: {dist_name} | M={M} | N={N_pdf}", fontsize=16)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

        ax_bern_curve = fig.add_subplot(gs[0, 0])
        ax_kde_curve = fig.add_subplot(gs[0, 1])
        ax_box_bern = fig.add_subplot(gs[1, 0])
        ax_box_kde = fig.add_subplot(gs[1, 1])

        for run in plot_runs:
            ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.1, lw=1)
            ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.1, lw=1)

        for ax, title in zip([ax_bern_curve, ax_kde_curve],
                             [f"Bernstein Estimations (N={N_pdf})", "Standard KDE Estimations"]):
            ax.plot(x_eval, pdf_true, 'k-', lw=2.5, label='Ground Truth', zorder=10)
            ax.set_title(title)
            ax.set_ylim(GLOBAL_Y_LIM_PDF)
            ax.set_xlim(0, upper_lim)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')


        def draw_custom_boxplot(ax, data, color, title):
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)

            ax.boxplot(data, medianprops=dict(color='black', linewidth=1.5))

            ax.axhline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
            ax.axhline(median_val, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

            mean_text = rf"Mean: {mean_val:.4f} $\pm$ {std_val:.4f}"
            mean_style = 'bold'

            median_text = f"Median: {median_val:.4f}"
            median_style = 'normal'

            if mean_val >= median_val:
                top_data = (mean_val, mean_text, mean_style)
                bot_data = (median_val, median_text, median_style)
            else:
                top_data = (median_val, median_text, median_style)
                bot_data = (mean_val, mean_text, mean_style)

            val_top, txt_top, style_top = top_data
            val_bot, txt_bot, style_bot = bot_data

            # Utilizzo del limite globale per calcolare il range e l'overlap
            y_range = GLOBAL_Y_LIM_KL[1] - GLOBAL_Y_LIM_KL[0]
            if y_range == 0: y_range = 1.0

            dist = abs(mean_val - median_val)
            overlap_threshold = 0.07 * y_range
            text_x_offset = 1.02

            # Logica per evitare sovrapposizioni tra i testi di Mean e Median
            if dist < overlap_threshold:
                mid_point = (mean_val + median_val) / 2

                ax.text(text_x_offset, mid_point, txt_top,
                        transform=ax.get_yaxis_transform(),
                        color=color, fontsize=8, va='bottom', fontweight=style_top)

                ax.text(text_x_offset, mid_point, txt_bot,
                        transform=ax.get_yaxis_transform(),
                        color=color, fontsize=8, va='top', fontweight=style_bot)
            else:
                ax.text(text_x_offset, val_top, txt_top,
                        transform=ax.get_yaxis_transform(),
                        color=color, fontsize=8, va='center', fontweight=style_top)

                ax.text(text_x_offset, val_bot, txt_bot,
                        transform=ax.get_yaxis_transform(),
                        color=color, fontsize=8, va='center', fontweight=style_bot)

            ax.set_title(title)
            ax.set_ylabel("KL Divergence")
            ax.set_ylim(GLOBAL_Y_LIM_KL)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks([])


        draw_custom_boxplot(ax_box_bern, kl_bernstein_list, 'blue', "Bernstein KL Error")
        draw_custom_boxplot(ax_box_kde, kl_kde_list, 'red', "KDE KL Error")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_dir = f"img/{today_str}/{dist_string}"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"kde_vs_bernstein_{dist_string}_M{M}_N{N_pdf}_runs{NUM_SIMULATIONS}.jpg"),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        results_table_data.append(
            [f"Erlang({K},{K})", M, N_pdf, f"{np.mean(kl_bernstein_list):.4f}", f"{np.mean(kl_kde_list):.4f}"]
        )

# =============================================================================
# 4. SUMMARY TABLE GENERATION
# =============================================================================

fig_height = 2 + 0.4 * len(results_table_data)
fig_table, ax_table = plt.subplots(figsize=(8, fig_height))
ax_table.axis('tight')
ax_table.axis('off')

col_labels = ["Distribution", "M", "N", "KL_bph", "KL_KDE"]
table = ax_table.table(cellText=results_table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0')

fig_table.suptitle(f"BPH Summary Results ({NUM_SIMULATIONS} runs)", fontweight='bold', fontsize=14, y=0.99)
fig_table.subplots_adjust(top=0.96)
output_dir_table = f"img/{today_str}"
os.makedirs(output_dir_table, exist_ok=True)
table_full_path = os.path.join(output_dir_table, f"summary_table_runs{NUM_SIMULATIONS}_erlangKK.jpg")
plt.savefig(table_full_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig_table)

print("Analisi completata.")
