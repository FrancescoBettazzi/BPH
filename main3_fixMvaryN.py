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

# Valori di K per le distribuzioni Erlang(K, K) con media unitaria.
K_VALUES = [2, 4, 8, 16]

# Griglia completa per l'analisi di sensibilità BPH e confronto KDE
M_VALUES = [27, 68, 163, 381, 866, 1938]
N_VALUES = [8, 16, 32, 64, 128, 256]

NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

# =============================================================================
# 2. COMPARISON LOOP & GRID SEARCH
# =============================================================================

results_table_data = []
today_str = datetime.now().strftime("%Y%m%d")

for K in K_VALUES:
    dist_obj = stats.gamma(a=K, scale=1 / K)
    dist_name = f"Erlang (K={K}, \u03bb={K}, \u03bc=1)"
    dist_string = f"erlang_K{K}"

    print(f"\n{'=' * 80}")
    print(f"INIZIO ANALISI: {dist_name}")
    print(f"{'=' * 80}")

    # Matrice per il grafico di sensibilità: righe = M, colonne = N
    kl_matrix_bph = np.zeros((len(M_VALUES), len(N_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        print(f"\n-> Test in corso: Campioni M={M} (esplorazione di tutti gli N...)")

        # Liste per raccogliere i risultati di questa specifica M
        kl_kde_list_M = []
        kl_bph_lists = {N: [] for N in N_VALUES}

        # Raccogliamo i dati per il plot 2x2 per TUTTI gli N
        plot_runs_dict = {N: [] for N in N_VALUES}

        # Generiamo la baseline per il plotting
        upper_lim = dist_obj.ppf(0.999)
        x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)
        pdf_true = dist_obj.pdf(x_eval)

        for i in range(NUM_SIMULATIONS):
            # 1. Campionamento unico per questo run
            campioni = dist_obj.rvs(size=M)
            ecdf = create_ecdf(campioni)

            # 2. Calcolo KDE (dipende solo da M, si calcola una volta per run)
            kde_func = gaussian_kde(campioni)
            pdf_kde = kde_func(x_eval)
            kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)
            kl_kde_list_M.append(kl_k)

            # 3. Testiamo tutti gli N richiesti sugli stessi campioni
            for N_pdf in N_VALUES:
                pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
                kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
                kl_bph_lists[N_pdf].append(kl_b)

                # Salviamo i dati per il plot 2x2 (fino a N_PLOT_LINES)
                if i < N_PLOT_LINES:
                    plot_runs_dict[N_pdf].append({
                        'x': x_eval,
                        'pdf_bern': pdf_bern,
                        'pdf_kde': pdf_kde,
                        'pdf_true': pdf_true
                    })

        # =============================================================================
        # 3. ELABORAZIONE DATI E CREAZIONE GRAFICI VECCHI (2x2 KDE vs BPH) PER TUTTI GLI N
        # =============================================================================
        mean_kl_k = np.mean(kl_kde_list_M)

        for i_n, N_pdf in enumerate(N_VALUES):
            mean_kl_b = np.mean(kl_bph_lists[N_pdf])
            kl_matrix_bph[i_m, i_n] = mean_kl_b

            # Aggiorniamo la tabella per tutte le combinazioni
            results_table_data.append([f"Erlang({K},{K})", M, N_pdf, f"{mean_kl_b:.4f}", f"{mean_kl_k:.4f}"])

            # --- INIZIO CREAZIONE PLOT 2x2 ---
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f"Comparison: {dist_name} | M={M} | N={N_pdf} | {NUM_SIMULATIONS} runs", fontsize=16)

            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
            ax_bern_curve = fig.add_subplot(gs[0, 0])
            ax_kde_curve = fig.add_subplot(gs[0, 1])
            ax_box_bern = fig.add_subplot(gs[1, 0])
            ax_box_kde = fig.add_subplot(gs[1, 1])

            base_x = x_eval
            base_y = pdf_true
            max_pdf_val = np.max(base_y)
            plot_runs = plot_runs_dict[N_pdf]

            for run in plot_runs:
                ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.15, lw=1)
                max_pdf_val = max(max_pdf_val, np.max(run['pdf_bern']))
                ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.15, lw=1)
                max_pdf_val = max(max_pdf_val, np.max(run['pdf_kde']))

            ax_bern_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
            ax_bern_curve.set_title(f"Bernstein Estimations (N={N_pdf})")
            ax_bern_curve.set_ylabel("PDF")
            ax_bern_curve.legend(loc='upper right')
            ax_bern_curve.grid(True, alpha=0.3)

            # Il grafico della KDE rimarrà identico per ogni M, indipendentemente da N
            ax_kde_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
            ax_kde_curve.set_title(f"Standard KDE Estimations (M={M})")
            ax_kde_curve.set_ylabel("PDF")
            ax_kde_curve.legend(loc='upper right')
            ax_kde_curve.grid(True, alpha=0.3)

            common_pdf_ylim = (0, max_pdf_val * 1.05)
            for ax in [ax_bern_curve, ax_kde_curve]:
                ax.set_ylim(common_pdf_ylim)
                ax.set_xlim(base_x[0], base_x[-1])

            all_errors = kl_bph_lists[N_pdf] + kl_kde_list_M
            min_err, max_err = min(all_errors), max(all_errors)
            y_box_margin = max((max_err - min_err) * 0.1, 1e-3)
            common_box_ylim = (max(0, min_err - y_box_margin), max_err + y_box_margin)


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
                overlap_threshold = 0.07 * (common_box_ylim[1] - common_box_ylim[0] or 1.0)
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
                ax.set_ylim(common_box_ylim)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_xticks([])


            draw_custom_boxplot(ax_box_bern, kl_bph_lists[N_pdf], 'blue', "Bernstein KL Error Distribution")
            draw_custom_boxplot(ax_box_kde, kl_kde_list_M, 'red', "Standard KDE KL Error Distribution")

            plt.tight_layout(rect=[0, 0, 1, 1])

            output_dir_dist = f"img/{today_str}/{dist_string}"
            os.makedirs(output_dir_dist, exist_ok=True)
            fig.savefig(os.path.join(output_dir_dist, f"kde_vs_bernstein_{dist_string}_M{M}_N{N_pdf}.jpg"), dpi=150,
                        bbox_inches='tight', facecolor='white')
            plt.close(fig)
            # --- FINE CREAZIONE PLOT 2x2 ---

    # =============================================================================
    # 4. CREAZIONE GRAFICO SOGLIA PRECISIONE (NOVITÀ BPH SENSITIVITY)
    # =============================================================================
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(M_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        ax_sens.plot(N_VALUES, kl_matrix_bph[i_m, :], marker='o', linewidth=2, color=colors[i_m], label=f'M = {M}')

    ax_sens.set_title(f"BPH KL Divergence vs Polynomial Degree (N)\n{dist_name}", fontsize=14, fontweight='bold')
    ax_sens.set_xlabel("Bernstein Degree (N)", fontsize=12)
    ax_sens.set_ylabel("Mean KL Divergence (Log Scale)", fontsize=12)
    ax_sens.set_yscale('log')
    ax_sens.set_xticks(N_VALUES)
    ax_sens.set_xticklabels([str(n) for n in N_VALUES])
    ax_sens.grid(True, which="both", ls="--", alpha=0.5)
    ax_sens.legend(title="Sample Size (M)", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_sens.axhline(y=1e-3, color='red', linestyle=':', alpha=0.7, label='Noise Threshold ($10^{-3}$)')

    plt.tight_layout()
    sens_output_dir = f"img/{today_str}/bph_sensitivity"
    os.makedirs(sens_output_dir, exist_ok=True)
    fig_sens.savefig(os.path.join(sens_output_dir, f"bph_trend_Erlang_K{K}.jpg"), dpi=150, bbox_inches='tight')
    plt.close(fig_sens)
    print(f"-> Grafico sensibilità soglia salvato per K={K}")

# =============================================================================
# 5. SUMMARY TABLE GENERATION (TERMINAL & IMAGE)
# =============================================================================

# Stampa nel terminale
print("\n" + "=" * 65)
print(f"RISULTATI FINALI SINTETICI (TUTTE LE COMBINAZIONI - {NUM_SIMULATIONS} RUNS)")
print("=" * 65)
print(f"{'Distribution':<15} | {'M':<6} | {'N':<6} | {'KL_bph':<10} | {'KL_KDE':<10}")
print("-" * 65)
for row in results_table_data:
    print(f"{row[0]:<15} | {row[1]:<6} | {row[2]:<6} | {row[3]:<10} | {row[4]:<10}")
print("=" * 65 + "\n")

# Generazione Immagine JPG della tabella
# Dato che ci saranno 144 righe, l'immagine sarà molto "alta".
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

# plt.title(f"Summary Results ({NUM_SIMULATIONS} simulations per config)", fontweight='bold', pad=20)
fig_table.suptitle(f"BPH Summary Results ({NUM_SIMULATIONS} runs)", fontweight='bold', fontsize=14, y=0.99)
fig_table.subplots_adjust(top=0.96)
output_dir_table = f"img/{today_str}"
table_full_path = os.path.join(output_dir_table, f"summary_table_runs{NUM_SIMULATIONS}_full.jpg")
plt.savefig(table_full_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig_table)

print(f"Tabella riassuntiva salvata in: {table_full_path}")
