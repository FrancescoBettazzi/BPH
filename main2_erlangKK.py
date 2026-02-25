import os
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde
from bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Valori di K per le distribuzioni Erlang(K, K) con media unitaria.
# K_VALUES = [16] K_VALUES = [2, 4, 8, 16]
K_VALUES = [2, 4, 8, 16]

# Coppie (M_campioni, N_grado_bernstein) basate sul limite di Babu: N <= M / ln(M)
# set ridotto: M_N_PAIRS = [(68, 16), (381, 64), (1938, 256)]
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

# =============================================================================
# 2. COMPARISON LOOP
# =============================================================================

results_table_data = []

for K in K_VALUES:
    # Configurazione della Erlang(K, K)
    # In scipy.stats.gamma, 'a' è il parametro di forma (K),
    # 'scale' è l'inverso del parametro di rate (1/lambda = 1/K).
    dist_obj = stats.gamma(a=K, scale=1 / K)
    dist_name = f"Erlang (K={K}, \u03bb={K}, \u03bc=1)"
    support_type = "semi-infinite"
    dist_string = f"erlang_K{K}"

    print(f"\n{'=' * 70}")
    print(f"INIZIO ANALISI DISTRIBUZIONE: {dist_name}")
    print(f"{'=' * 70}")

    for M, N_pdf in M_N_PAIRS:
        print(f"\n-> Test in corso: Campioni M={M} | Grado Bernstein N={N_pdf}")
        print("-" * 60)

        kl_bernstein_list = []
        kl_kde_list = []
        plot_runs = []

        for i in range(NUM_SIMULATIONS):
            campioni = dist_obj.rvs(size=M)

            # Il supporto della Erlang è semi-infinito [0, +inf)
            upper_lim = dist_obj.ppf(0.999)
            x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)

            pdf_true = dist_obj.pdf(x_eval)
            ecdf = create_ecdf(campioni)

            pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)

            kde_func = gaussian_kde(campioni)
            pdf_kde = kde_func(x_eval)

            kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
            kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)

            kl_bernstein_list.append(kl_b)
            kl_kde_list.append(kl_k)

            if i < N_PLOT_LINES:
                plot_runs.append({
                    'x': x_eval,
                    'pdf_bern': pdf_bern,
                    'pdf_kde': pdf_kde,
                    'pdf_true': pdf_true
                })

            if (i + 1) % 25 == 0:  # Print ridotto per non intasare il terminale
                print(f"Run {i + 1}/{NUM_SIMULATIONS} | KL Bern: {kl_b:.4f} | KL KDE: {kl_k:.4f}")

        # =============================================================================
        # 3. VISUALIZATION
        # =============================================================================

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Comparison: {dist_name} | M={M} | N={N_pdf} | {NUM_SIMULATIONS} runs", fontsize=16)

        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

        ax_bern_curve = fig.add_subplot(gs[0, 0])
        ax_kde_curve = fig.add_subplot(gs[0, 1])
        ax_box_bern = fig.add_subplot(gs[1, 0])
        ax_box_kde = fig.add_subplot(gs[1, 1])

        base_x = plot_runs[0]['x']
        base_y = plot_runs[0]['pdf_true']
        max_pdf_val = np.max(base_y)

        for run in plot_runs:
            ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.15, lw=1)
            max_pdf_val = max(max_pdf_val, np.max(run['pdf_bern']))

        ax_bern_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
        ax_bern_curve.set_title(f"Bernstein Estimations (N={N_pdf})")
        ax_bern_curve.set_ylabel("PDF")
        ax_bern_curve.legend(loc='upper right')
        ax_bern_curve.grid(True, alpha=0.3)

        for run in plot_runs:
            ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.15, lw=1)
            max_pdf_val = max(max_pdf_val, np.max(run['pdf_kde']))

        ax_kde_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
        ax_kde_curve.set_title(f"Standard KDE Estimations")
        ax_kde_curve.set_ylabel("PDF")
        ax_kde_curve.legend(loc='upper right')
        ax_kde_curve.grid(True, alpha=0.3)

        common_pdf_ylim = (0, max_pdf_val * 1.05)
        ax_bern_curve.set_ylim(common_pdf_ylim)
        ax_kde_curve.set_ylim(common_pdf_ylim)
        ax_bern_curve.set_xlim(base_x[0], base_x[-1])
        ax_kde_curve.set_xlim(base_x[0], base_x[-1])

        all_errors = kl_bernstein_list + kl_kde_list
        min_err, max_err = min(all_errors), max(all_errors)
        y_box_margin = (max_err - min_err) * 0.1
        if y_box_margin == 0: y_box_margin = 1e-3

        common_box_ylim = (max(0, min_err - y_box_margin), max_err + y_box_margin)


        def draw_custom_boxplot(ax, data, color, title, label_y=True):
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

            y_range = common_box_ylim[1] - common_box_ylim[0]
            if y_range == 0: y_range = 1.0

            dist = abs(mean_val - median_val)
            overlap_threshold = 0.07 * y_range

            text_x_offset = 1.02

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
            if label_y:
                ax.set_ylabel("KL Divergence")

            ax.set_ylim(common_box_ylim)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks([])


        draw_custom_boxplot(ax_box_bern, kl_bernstein_list, 'blue',
                            "Bernstein KL Error Distribution", label_y=True)

        draw_custom_boxplot(ax_box_kde, kl_kde_list, 'red',
                            "Standard KDE KL Error Distribution", label_y=True)

        plt.tight_layout(rect=[0, 0, 1, 1])

        # Nome file reso dinamico per non sovrascrivere i grafici
        today_str = datetime.now().strftime("%Y%m%d")
        output_dir = f"img/{today_str}/{dist_string}"
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"kde_vs_bernstein_{dist_string}_M{M}_N{N_pdf}_runs{NUM_SIMULATIONS}.jpg"
        full_path = os.path.join(output_dir, file_name)
        fig.savefig(full_path, dpi=150, bbox_inches='tight', facecolor='white')

        plt.close(fig)

        mean_kl_b = np.mean(kl_bernstein_list)
        mean_kl_k = np.mean(kl_kde_list)
        dist_label = f"Erlang({K},{K})"
        results_table_data.append([dist_label, M, N_pdf, f"{mean_kl_b:.4f}", f"{mean_kl_k:.4f}"])

# =============================================================================
# 4. SUMMARY TABLE GENERATION
# =============================================================================

# --- A. Stampa nel terminale ---
print("\n" + "="*65)
print(f"RISULTATI FINALI SINTETICI ({NUM_SIMULATIONS} RUNS)")
print("="*65)
print(f"{'Distribution':<15} | {'M':<6} | {'N':<6} | {'KL_bph':<10} | {'KL_KDE':<10}")
print("-" * 65)
for row in results_table_data:
    print(f"{row[0]:<15} | {row[1]:<6} | {row[2]:<6} | {row[3]:<10} | {row[4]:<10}")
print("="*65 + "\n")

# --- B. Generazione Immagine JPG ---
# Calcoliamo dinamicamente l'altezza dell'immagine in base al numero di righe
fig_height = 2 + 0.5 * len(results_table_data)
fig_table, ax_table = plt.subplots(figsize=(8, fig_height))
ax_table.axis('tight')
ax_table.axis('off')

col_labels = ["Distribution", "M", "N", "KL_bph", "KL_KDE"]

# Creazione della tabella su Matplotlib
table = ax_table.table(
    cellText=results_table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

# Stile della tabella
table.scale(1, 1.8) # Aumenta l'altezza delle celle per leggibilità
table.auto_set_font_size(False)
table.set_fontsize(11)

# Rendiamo in grassetto l'intestazione e diamo un colore di sfondo
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0') # Grigio chiaro

plt.title(f"Summary Results ({NUM_SIMULATIONS} simulations per config)", fontweight='bold', pad=20)

# Salvataggio
output_dir_table = f"img/{today_str}"
os.makedirs(output_dir_table, exist_ok=True)
table_file_name = f"summary_table_runs{NUM_SIMULATIONS}.jpg"
table_full_path = os.path.join(output_dir_table, table_file_name)

plt.savefig(table_full_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig_table)

# print(f"Tabella riassuntiva salvata in: {table_full_path}")
