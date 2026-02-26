import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from test.bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Valori di K per le distribuzioni Erlang(K, K) con media unitaria.
K_VALUES = [2, 4, 8, 16]

# Griglia completa per l'analisi di sensibilità BPH
M_VALUES = [27, 68, 163, 381, 866, 1938]
N_VALUES = [8, 16, 32, 64, 128, 256]

NUM_SIMULATIONS = 100
NUM_POINTS = 500

# =============================================================================
# 2. BPH SENSITIVITY LOOP
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

    # Generiamo la baseline
    upper_lim = dist_obj.ppf(0.999)
    x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)
    pdf_true = dist_obj.pdf(x_eval)

    for i_m, M in enumerate(M_VALUES):
        print(f"-> Test in corso: Campioni M={M:<4} (esplorazione di tutti gli N...)")

        # Dizionario per raccogliere i risultati BPH per i vari N
        kl_bph_lists = {N: [] for N in N_VALUES}

        for i in range(NUM_SIMULATIONS):
            # Campionamento unico per questo run
            campioni = dist_obj.rvs(size=M)
            ecdf = create_ecdf(campioni)

            # Testiamo tutti gli N richiesti sugli stessi campioni (solo BPH)
            for N_pdf in N_VALUES:
                pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
                kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
                kl_bph_lists[N_pdf].append(kl_b)

        # Salviamo le medie per la matrice e la tabella
        for i_n, N_pdf in enumerate(N_VALUES):
            mean_kl_b = np.mean(kl_bph_lists[N_pdf])
            kl_matrix_bph[i_m, i_n] = mean_kl_b

            # Aggiorniamo la tabella (rimosso il riferimento a KDE)
            results_table_data.append([f"Erlang({K},{K})", M, N_pdf, f"{mean_kl_b:.4f}"])

    # =============================================================================
    # 3. CREAZIONE GRAFICO SOGLIA PRECISIONE (BPH SENSITIVITY)
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
    print(f"\n-> Grafico sensibilità soglia salvato per K={K}")

# =============================================================================
# 4. SUMMARY TABLE GENERATION (TERMINAL & IMAGE)
# =============================================================================

# Stampa nel terminale
print("\n" + "=" * 55)
print(f"RISULTATI FINALI SINTETICI (SOLO BPH - {NUM_SIMULATIONS} RUNS)")
print("=" * 55)
print(f"{'Distribution':<15} | {'M':<6} | {'N':<6} | {'KL_bph':<10}")
print("-" * 55)
for row in results_table_data:
    print(f"{row[0]:<15} | {row[1]:<6} | {row[2]:<6} | {row[3]:<10}")
print("=" * 55 + "\n")

# Generazione Immagine JPG della tabella
fig_height = 2 + 0.3 * len(results_table_data)
fig_table, ax_table = plt.subplots(figsize=(6, fig_height))
ax_table.axis('tight')
ax_table.axis('off')

col_labels = ["Distribution", "M", "N", "KL_bph"]
table = ax_table.table(cellText=results_table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0')

# plt.title(f"BPH Summary Results ({NUM_SIMULATIONS} sim/config)", fontweight='bold', pad=20)
fig_table.suptitle(f"BPH Summary Results ({NUM_SIMULATIONS} runs)", fontweight='bold', fontsize=14, y=0.99)
fig_table.subplots_adjust(top=0.96)
output_dir_table = f"img/{today_str}"
table_full_path = os.path.join(output_dir_table, f"summary_table_bph_runs{NUM_SIMULATIONS}.jpg")
plt.savefig(table_full_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig_table)

print(f"Tabella riassuntiva salvata in: {table_full_path}")
