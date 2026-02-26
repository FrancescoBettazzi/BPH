import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gammaln


# Helpers
def create_ecdf(campioni):
    M = len(campioni)
    campioni_ordinati = np.sort(campioni)
    y_gradino = np.arange(1, M + 1) / M
    return interp1d(
        campioni_ordinati,
        y_gradino,
        kind='previous',
        bounds_error=False,
        fill_value=(0.0, 1.0)
    )


def get_bernstein_basis(z, N):
    z = z[:, np.newaxis]
    n = np.arange(N + 1)

    # Log-sum-exp trick per stabilità con N elevati
    log_coeffs = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    eps = 1e-16
    z_safe = np.clip(z, eps, 1.0 - eps)

    log_pow_z = n * np.log(z_safe)
    log_pow_1_z = (N - n) * np.log(1.0 - z_safe)

    return np.exp(log_coeffs + log_pow_z + log_pow_1_z)


# Bernstein estimation on [0, inf)
def calculate_bernstein_exp_cdf(ecdf, N, asse_x, scale=1.0):
    asse_x = np.asarray(asse_x)
    z = np.clip(1.0 - np.exp(-scale * asse_x), 0.0, 1.0)

    k_range = np.arange(N + 1)
    z_nodes = k_range / N
    weights = np.zeros(N + 1)

    # Mapping inverso per i pesi: x = -ln(1-z)/s
    mask_finite = (z_nodes < 1.0)
    if np.any(mask_finite):
        x_nodes = -np.log(1.0 - z_nodes[mask_finite]) / scale
        weights[mask_finite] = ecdf(x_nodes)
    weights[~mask_finite] = 1.0

    basis = get_bernstein_basis(z, N)
    return basis @ weights


def calculate_bernstein_exp_pdf(ecdf, N, asse_x, scale=1.0):
    asse_x = np.asarray(asse_x)
    z = np.clip(1.0 - np.exp(-scale * asse_x), 0.0, 1.0)

    k_range = np.arange(N + 1)
    z_nodes = k_range / N
    weights = np.zeros(N + 1)

    mask_finite = (z_nodes < 1.0)
    if np.any(mask_finite):
        x_nodes = -np.log(1.0 - z_nodes[mask_finite]) / scale
        weights[mask_finite] = ecdf(x_nodes)
    weights[~mask_finite] = 1.0

    # Calcolo derivata della base
    diffs = np.diff(weights)
    basis_deriv = get_bernstein_basis(z, N - 1)
    pdf_z = N * (basis_deriv @ diffs)

    # Regola della catena: pdf_x = pdf_z * dz/dx
    # dz/dx = scale * exp(-scale * x)
    jacobian = scale * np.exp(-scale * asse_x)

    return pdf_z * jacobian
