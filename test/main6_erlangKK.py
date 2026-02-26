import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde
from test.bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

K_VALUES = [2, 4, 8, 16]

M_N_PAIRS = [
    (27, 8),
    (68, 16),
    (163, 32),
    (381, 64),
    (866, 128),
    (1938, 256)
]

NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

# --- LIMITI UNIFICATI PER ARTICOLO ---
# Erlang(16,16) ha un picco intorno a 1.6. Impostiamo 2.0 per dare respiro.
GLOBAL_Y_LIM_PDF = (0, 2.0)
# La KL divergence cala drasticamente; 0.6 è un buon compromesso per vedere i dettagli
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

    print(f"\n{'=' * 70}\nINIZIO ANALISI: {dist_name}\n{'=' * 70}")

    for M, N_pdf in M_N_PAIRS:
        print(f"-> Test: M={M} | N={N_pdf}")

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
        # 3. VISUALIZATION (WITH UNIFIED AXES)
        # =============================================================================

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Comparison: {dist_name} | M={M} | N={N_pdf}", fontsize=16)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

        ax_bern_curve = fig.add_subplot(gs[0, 0])
        ax_kde_curve = fig.add_subplot(gs[0, 1])
        ax_box_bern = fig.add_subplot(gs[1, 0])
        ax_box_kde = fig.add_subplot(gs[1, 1])

        # --- Grafici PDF ---
        for run in plot_runs:
            ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.1, lw=1)
            ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.1, lw=1)

        for ax, title in zip([ax_bern_curve, ax_kde_curve],
                             [f"Bernstein Estimations (N={N_pdf})", "Standard KDE Estimations"]):
            ax.plot(x_eval, pdf_true, 'k-', lw=2.5, label='Ground Truth', zorder=10)
            ax.set_title(title)
            ax.set_ylim(GLOBAL_Y_LIM_PDF)  # <--- UNIFICAZIONE PDF
            ax.set_xlim(0, upper_lim)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')


        # --- Boxplot KL Divergence ---
        def draw_custom_boxplot(ax, data, color, title):
            mean_val, median_val, std_val = np.mean(data), np.median(data), np.std(data)
            ax.boxplot(data, medianprops=dict(color='black', linewidth=1.5))
            ax.axhline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.9)

            # Posizionamento testo statistiche
            txt = rf"Mean: {mean_val:.4f} $\pm$ {std_val:.4f}"
            ax.text(1.02, mean_val, txt, transform=ax.get_yaxis_transform(),
                    color=color, fontsize=9, va='center', fontweight='bold')

            ax.set_title(title)
            ax.set_ylabel("KL Divergence")
            ax.set_ylim(GLOBAL_Y_LIM_KL)  # <--- UNIFICAZIONE KL
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks([])


        draw_custom_boxplot(ax_box_bern, kl_bernstein_list, 'blue', "Bernstein KL Error")
        draw_custom_boxplot(ax_box_kde, kl_kde_list, 'red', "KDE KL Error")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_dir = f"img/{today_str}/{dist_string}"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"plot_M{M}_N{N_pdf}.jpg"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        results_table_data.append(
            [f"Erlang({K},{K})", M, N_pdf, f"{np.mean(kl_bernstein_list):.4f}", f"{np.mean(kl_kde_list):.4f}"])

# =============================================================================
# 4. SUMMARY TABLE (Immagine finale)
# =============================================================================
fig_height = 2 + 0.4 * len(results_table_data)
fig_table, ax_table = plt.subplots(figsize=(10, fig_height))
ax_table.axis('off')
table = ax_table.table(cellText=results_table_data, colLabels=["Dist", "M", "N", "KL_bph", "KL_KDE"], loc='center',
                       cellLoc='center')
table.scale(1, 2)
table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    if row == 0: cell.set_facecolor('#e0e0e0')

plt.savefig(f"img/{today_str}/summary_table.jpg", dpi=200, bbox_inches='tight')
plt.close(fig_table)
print("Analisi completata con assi unificati.")
