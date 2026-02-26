import os
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde
from test.bernstein_exp import create_ecdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Choose the distribution here:
# 'u', 'u_0_1', 'u_1_2', 'n', 'k', 'k_d', 'k_u', 'erlang',
# 'weibull_1_5', 'weibull_0_5',
# 'lognormal_1_8', 'lognormal_0_8', 'lognormal_0_2'

DIST_KEY = 'erlang'

# DIST_KEYS = ['u_0_1', 'u_1_2', 'erlang', 'weibull_1_5', 'weibull_0_5', 'lognormal_1_8', 'lognormal_0_8', 'lognormal_0_2']

M = 100
NUM_SIMULATIONS = 100
NUM_POINTS = 500
N_PLOT_LINES = 50

# for DIST_KEY in DIST_KEYS:
dist_config = {}
dist_obj = None
dist_name = ""
dist_string = ""
support_type = "bounded"

if DIST_KEY == 'u_0_1':
    dist_obj = stats.uniform(loc=0, scale=1)
    dist_name = "Uniform [0, 1]"
    support_type = "bounded"
    dist_string = "uniform_a0_b1"

elif DIST_KEY == 'u_1_2':
    dist_obj = stats.uniform(loc=1, scale=1)
    dist_name = "Uniform [1, 2]"
    support_type = "bounded"
    dist_string = "uniform_a1_b2"

elif DIST_KEY == 'n':
    dist_obj = stats.norm(loc=0, scale=1)
    dist_name = "Normal (mu=0, sigma=1)"
    support_type = "unbounded"
    dist_string = "gaussian_mu0_sigma1"

elif DIST_KEY == 'erlang':
    n_erl = 5
    dist_obj = stats.gamma(a=n_erl, scale=1/n_erl)
    dist_name = f"Erlang (n={n_erl}, mu=1)"
    support_type = "semi-infinite"
    dist_string = f"erlang_n{n_erl}_mu1"

elif DIST_KEY == 'weibull_1_5':
    dist_obj = stats.weibull_min(c=1.5, scale=1)
    dist_name = "Weibull (shape=1.5, scale=1)"
    support_type = "semi-infinite"
    dist_string = "weibull_scale1.0_shape1.5"

elif DIST_KEY == 'weibull_0_5':
    dist_obj = stats.weibull_min(c=0.5, scale=1)
    dist_name = "Weibull (shape=0.5, scale=1)"
    support_type = "semi-infinite"
    dist_string = "weibull_scale1.0_shape0.5"

elif DIST_KEY == 'lognormal_1_8':
    dist_obj = stats.lognorm(s=1.8, scale=1)
    dist_name = "Lognormal (s=1.8, scale=1)"
    support_type = "semi-infinite"
    dist_string = "lognormal_scale1.0_shape1.8"

elif DIST_KEY == 'lognormal_0_8':
    dist_obj = stats.lognorm(s=0.8, scale=1)
    dist_name = "Lognormal (s=0.8, scale=1)"
    support_type = "semi-infinite"
    dist_string = "lognormal_scale1.0_shape0.8"

elif DIST_KEY == 'lognormal_0_2':
    dist_obj = stats.lognorm(s=0.2, scale=1)
    dist_name = "Lognormal (s=0.2, scale=1)"
    support_type = "semi-infinite"
    dist_string = "lognormal_scale1.0_shape0.2"

else:
    raise ValueError(f"Distribution '{DIST_KEY}' not recognized.")

N_pdf = math.ceil(M / math.log(M, 2))

kl_bernstein_list = []
kl_kde_list = []
plot_runs = []

# =============================================================================
# 2. COMPARISON LOOP
# =============================================================================

print(f"\nDistribution Analysis: {dist_name}")
print(f"Bernstein Method: EXP")
print(f"Samples M={M} | N_Bernstein={N_pdf}")
print("-" * 60)

for i in range(NUM_SIMULATIONS):
    campioni = dist_obj.rvs(size=M)

    if support_type == "bounded":
        if DIST_KEY == 'u_0_1':
            lower, upper = 0, 1
        elif DIST_KEY == 'u_1_2':
            lower, upper = 1, 2
        else:
            lower, upper = 0.0001, 0.9999
        x_eval = np.linspace(lower, upper, NUM_POINTS)

    elif support_type == "unbounded":
        lower_lim = dist_obj.ppf(0.001)
        upper_lim = dist_obj.ppf(0.999)
        x_eval = np.linspace(lower_lim, upper_lim, NUM_POINTS)

    else:
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

    if (i + 1) % 10 == 0:
        print(f"Run {i + 1}/{NUM_SIMULATIONS} | KL Bern: {kl_b:.4f} | KL KDE: {kl_k:.4f}")

# =============================================================================
# 3. VISUALIZATION
# =============================================================================

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Comparison: {dist_name} (num. samples M={M}) - {NUM_SIMULATIONS} runs", fontsize=16)

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

    today_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"img/{today_str}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"kde_vs_bernstein_M{M}_SIMUL{NUM_SIMULATIONS}_{dist_string}.jpg"
    full_path = os.path.join(output_dir, file_name)
    fig.savefig(full_path, dpi=150, bbox_inches='tight', facecolor='white')

    plt.close(fig)
