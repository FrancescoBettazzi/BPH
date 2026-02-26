import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde

# Assicurati che questi import siano disponibili nel tuo ambiente
# from bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

K_VALUES = [2, 4, 8, 16]
M_VALUES = [27, 68, 163, 381, 866, 1938]
N_VALUES = [8, 16, 32, 64, 128, 256]

NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

# --- NUOVI LIMITI UNIFICATI PER CONFRONTO ARTICOLO ---
# La PDF di Erlang(16,16) arriva a circa 1.6. Settiamo 2.0 per sicurezza.
GLOBAL_Y_LIM_PDF = (0, 2.0)
# La KL divergence solitamente sta sotto 0.5 nei boxplot lineari
GLOBAL_Y_LIM_KL_BOXPLOT = (0, 0.6)
# Per la sensibilità logaritmica: da 10^-4 a 10^0
GLOBAL_Y_LIM_KL_SENSITIVITY = (1e-4, 1.0)

# =============================================================================
# 2. COMPARISON LOOP & GRID SEARCH
# =============================================================================

results_table_data = []
today_str = datetime.now().strftime("%Y%m%d") + "_all_config"

for K in K_VALUES:
    dist_obj = stats.gamma(a=K, scale=1 / K)
    dist_name = f"Erlang (K={K}, \u03bb={K}, \u03bc=1)"
    dist_string = f"erlang_K{K}"

    print(f"\n{'=' * 80}\nINIZIO ANALISI: {dist_name}\n{'=' * 80}")

    kl_matrix_bph = np.zeros((len(M_VALUES), len(N_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        print(f"-> Analisi M={M}...")
        kl_kde_list_M = []
        kl_bph_lists = {N: [] for N in N_VALUES}
        plot_runs_dict = {N: [] for N in N_VALUES}

        upper_lim = dist_obj.ppf(0.999)
        x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)
        pdf_true = dist_obj.pdf(x_eval)

        for i in range(NUM_SIMULATIONS):
            campioni = dist_obj.rvs(size=M)
            # ecdf = create_ecdf(campioni) # Scommenta nel tuo ambiente

            kde_func = gaussian_kde(campioni)
            pdf_kde = kde_func(x_eval)
            kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)
            kl_kde_list_M.append(kl_k)

            for N_pdf in N_VALUES:
                # Sostituisci con la tua funzione reale:
                # pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
                pdf_bern = pdf_kde  # Placeholder per test
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

            # --- CREAZIONE PLOT 2x2 ---
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
                ax.set_ylim(GLOBAL_Y_LIM_PDF)  # <--- ASSE Y UNIFICATO PDF
                ax.set_xlim(0, upper_lim)
                ax.grid(True, alpha=0.3)
                ax.legend()


            def draw_custom_boxplot(ax, data, color, title):
                ax.boxplot(data, medianprops=dict(color='black'))
                mean_v = np.mean(data)
                ax.axhline(mean_v, color=color, linestyle='--', label=f'Mean: {mean_v:.4f}')
                ax.set_title(title)
                ax.set_ylim(GLOBAL_Y_LIM_KL_BOXPLOT)  # <--- ASSE Y UNIFICATO KL
                ax.grid(axis='y', linestyle='--', alpha=0.5)


            draw_custom_boxplot(ax_box_bern, kl_bph_lists[N_pdf], 'blue', "Bernstein KL Error")
            draw_custom_boxplot(ax_box_kde, kl_kde_list_M, 'red', "KDE KL Error")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_dir_dist = f"img/{today_str}/{dist_string}"
            os.makedirs(output_dir_dist, exist_ok=True)
            fig.savefig(os.path.join(output_dir_dist, f"confronto_M{M}_N{N_pdf}.jpg"), dpi=100)
            plt.close(fig)

    # =============================================================================
    # 4. GRAFICO SOGLIA PRECISIONE (UNIFICATO)
    # =============================================================================
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(M_VALUES)))

    for i_m, M in enumerate(M_VALUES):
        ax_sens.plot(N_VALUES, kl_matrix_bph[i_m, :], marker='o', lw=2, color=colors[i_m], label=f'M={M}')

    ax_sens.set_title(f"BPH Trend: {dist_name}")
    ax_sens.set_yscale('log')
    ax_sens.set_ylim(GLOBAL_Y_LIM_KL_SENSITIVITY)  # <--- ASSE Y UNIFICATO LOG
    ax_sens.set_xlabel("Bernstein Degree (N)")
    ax_sens.set_ylabel("Mean KL Divergence")
    ax_sens.grid(True, which="both", ls="--", alpha=0.5)
    ax_sens.legend(title="Sample Size (M)", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    sens_output_dir = f"img/{today_str}/bph_sensitivity"
    os.makedirs(sens_output_dir, exist_ok=True)
    fig_sens.savefig(os.path.join(sens_output_dir, f"bph_trend_K{K}.jpg"), dpi=150)
    plt.close(fig_sens)

print("\nProcesso completato. Assi Y unificati per il confronto scientifico.")
