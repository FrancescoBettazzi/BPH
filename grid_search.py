import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde
from bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# Configuration
K_VALUES = [2, 4, 8, 16]
M_VALUES = [27, 68, 163, 381, 866, 1938]
N_VALUES = [8, 16, 32, 64, 128, 256]

NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

GLOBAL_Y_LIM_PDF = (0, 2.0)
GLOBAL_Y_LIM_KL_BOXPLOT = (0, 0.6)
GLOBAL_Y_LIM_KL_SENSITIVITY = (1e-4, 1.0)

# Comparison loop & Grid search
results_table_data = []
today_str = datetime.now().strftime("%Y%m%d") + "_grid_search"

for K in K_VALUES:
    dist_obj = stats.gamma(a=K, scale=1 / K)
    dist_name = f"Erlang (K={K}, \u03bb={K}, \u03bc=1)"
    dist_string = f"erlang_K{K}"

    print(f"Analisi: {dist_name}")
    kl_matrix_bph = np.zeros((len(M_VALUES), len(N_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        kl_kde_list_M = []
        kl_bph_lists = {N: [] for N in N_VALUES}
        plot_runs_dict = {N: [] for N in N_VALUES}

        upper_lim = dist_obj.ppf(0.999)
        x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)
        pdf_true = dist_obj.pdf(x_eval)

        for i in range(NUM_SIMULATIONS):
            campioni = dist_obj.rvs(size=M)
            ecdf = create_ecdf(campioni)

            kde_func = gaussian_kde(campioni)
            pdf_kde = kde_func(x_eval)
            kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)
            kl_kde_list_M.append(kl_k)

            for N_pdf in N_VALUES:
                pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
                kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
                kl_bph_lists[N_pdf].append(kl_b)

                if i < N_PLOT_LINES:
                    plot_runs_dict[N_pdf].append({
                        'x': x_eval, 'pdf_bern': pdf_bern,
                        'pdf_kde': pdf_kde, 'pdf_true': pdf_true
                    })

        mean_kl_k = np.mean(kl_kde_list_M)

        for i_n, N_pdf in enumerate(N_VALUES):
            mean_kl_b = np.mean(kl_bph_lists[N_pdf])
            kl_matrix_bph[i_m, i_n] = mean_kl_b
            results_table_data.append([f"Erlang({K},{K})", M, N_pdf, f"{mean_kl_b:.4f}", f"{mean_kl_k:.4f}"])

            # 2x2 Comparison Plot
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f"Comparison: {dist_name} | M={M} | N={N_pdf}", fontsize=16)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

            ax_bern_curve = fig.add_subplot(gs[0, 0])
            ax_kde_curve = fig.add_subplot(gs[0, 1])
            ax_box_bern = fig.add_subplot(gs[1, 0])
            ax_box_kde = fig.add_subplot(gs[1, 1])

            for run in plot_runs_dict[N_pdf]:
                ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.1, lw=1)
                ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.1, lw=1)

            for ax, title in zip([ax_bern_curve, ax_kde_curve],
                                 [f"Bernstein (N={N_pdf})", f"KDE (M={M})"]):
                ax.plot(x_eval, pdf_true, 'k-', lw=2, label='Ground Truth')
                ax.set_title(title)
                ax.set_ylim(GLOBAL_Y_LIM_PDF)
                ax.set_xlim(0, upper_lim)
                ax.grid(True, alpha=0.3)
                ax.legend()

            def draw_custom_boxplot(ax, data, color, title):
                mean_val, median_val, std_val = np.mean(data), np.median(data), np.std(data)
                ax.boxplot(data, medianprops=dict(color='black', linewidth=1.5))
                ax.axhline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
                ax.axhline(median_val, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

                top_data = (mean_val, rf"Mean: {mean_val:.4f} $\pm$ {std_val:.4f}",
                            'bold') if mean_val >= median_val else (median_val, f"Median: {median_val:.4f}", 'normal')
                bot_data = (median_val, f"Median: {median_val:.4f}", 'normal') if mean_val >= median_val else (mean_val,
                                                                                                              rf"Mean: {mean_val:.4f} $\pm$ {std_val:.4f}",
                                                                                                              'bold')

                dist = abs(mean_val - median_val)
                # Sostituito il common_box_ylim con il limite globale
                overlap_threshold = 0.07 * (GLOBAL_Y_LIM_KL_BOXPLOT[1] - GLOBAL_Y_LIM_KL_BOXPLOT[0] or 1.0)
                text_x_offset = 1.02

                if dist < overlap_threshold:
                    mid_point = (mean_val + median_val) / 2
                    ax.text(text_x_offset, mid_point, top_data[1], transform=ax.get_yaxis_transform(), color=color,
                            fontsize=8, va='bottom', fontweight=top_data[2])
                    ax.text(text_x_offset, mid_point, bot_data[1], transform=ax.get_yaxis_transform(), color=color,
                            fontsize=8, va='top', fontweight=bot_data[2])
                else:
                    ax.text(text_x_offset, top_data[0], top_data[1], transform=ax.get_yaxis_transform(), color=color,
                            fontsize=8, va='center', fontweight=top_data[2])
                    ax.text(text_x_offset, bot_data[0], bot_data[1], transform=ax.get_yaxis_transform(), color=color,
                            fontsize=8, va='center', fontweight=bot_data[2])

                ax.set_title(title)
                ax.set_ylabel("KL Divergence")
                ax.set_ylim(GLOBAL_Y_LIM_KL_BOXPLOT)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_xticks([])

            draw_custom_boxplot(ax_box_bern, kl_bph_lists[N_pdf], 'blue', "Bernstein KL Error")
            draw_custom_boxplot(ax_box_kde, kl_kde_list_M, 'red', "KDE KL Error")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_dir_dist = f"img/{today_str}/{dist_string}"
            os.makedirs(output_dir_dist, exist_ok=True)
            fig.savefig(os.path.join(output_dir_dist, f"kde_vs_bernstein_{dist_string}_M{M}_N{N_pdf}.jpg"), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

    # BPH Sensitivity Plot
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(M_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        ax_sens.plot(N_VALUES, kl_matrix_bph[i_m, :], marker='o', lw=2, color=colors[i_m], label=f'M={M}')

    ax_sens.set_title(f"BPH Trend: {dist_name}")
    ax_sens.set_yscale('log')
    ax_sens.set_ylim(GLOBAL_Y_LIM_KL_SENSITIVITY)
    ax_sens.set_xlabel("Bernstein Degree (N)")
    ax_sens.set_ylabel("Mean KL Divergence")
    ax_sens.grid(True, which="both", ls="--", alpha=0.5)
    ax_sens.legend(title="Sample Size (M)", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    sens_output_dir = f"img/{today_str}/bph_sensitivity"
    os.makedirs(sens_output_dir, exist_ok=True)
    fig_sens.savefig(os.path.join(sens_output_dir, f"bph_trend_K{K}.jpg"), dpi=150)
    plt.close(fig_sens)

# SUMMARY TABLE GENERATION (TERMINAL & IMAGE)

# Stampa nel terminale
'''
print("\n" + "=" * 65)
print(f"RISULTATI FINALI SINTETICI (TUTTE LE COMBINAZIONI - {NUM_SIMULATIONS} RUNS)")
print("=" * 65)
print(f"{'Distribution':<15} | {'M':<6} | {'N':<6} | {'KL_bph':<10} | {'KL_KDE':<10}")
print("-" * 65)
for row in results_table_data:
    print(f"{row[0]:<15} | {row[1]:<6} | {row[2]:<6} | {row[3]:<10} | {row[4]:<10}")
print("=" * 65 + "\n")
'''

# Summary table
fig_height = 2 + 0.3 * len(results_table_data)
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

# Salvataggio
output_dir_table = f"img/{today_str}"
os.makedirs(output_dir_table, exist_ok=True)
table_full_path = os.path.join(output_dir_table, f"summary_table_runs{NUM_SIMULATIONS}_grid_search.jpg")
plt.savefig(table_full_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig_table)

# print(f"Tabella riassuntiva salvata in: {table_full_path}")
print("Analisi completata.")
