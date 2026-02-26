# Density Estimation Comparison: Bernstein Polynomials vs KDE

This project compares two methods for estimating the Probability Density Function (PDF) using samples drawn from Erlang distributions:
1. **Bernstein Polynomials (BPH)**
2. **Standard Kernel Density Estimation (KDE)**

## File Structure

* `bernstein_exp.py`: Contains the core mathematical functions for creating the ECDF (Empirical Cumulative Distribution Function) and calculating the PDF based on Bernstein polynomials. **It must not be run directly.**
* `erlangKK.py`: Executable script to test specific pairs of sample sizes (*M*) and polynomial degrees (*N*).
* `grid_search.py`: Executable script to perform a grid search over all possible combinations of *M* and *N*, which is useful for sensitivity analysis.

---

## 1. Running `erlangKK.py`

This script analyzes the performance of the two estimators on targeted pairs of *M* (number of samples) and *N* (degree of the Bernstein polynomial).

### Configurations
At the beginning of the file, you can modify the following variables:
* `K_VALUES`: List of shape/scale parameters for the Erlang distributions to test (e.g., `[2, 4, 8, 16]`).
* `M_N_PAIRS`: List of `(M, N)` tuples. It links a specific sample size to a specific polynomial degree.
* `NUM_SIMULATIONS`: Number of independent iterations for each configuration (to ensure a robust statistical estimate).
* `NUM_POINTS`: Number of points on the x-axis used to evaluate and draw the curves.
* `N_PLOT_LINES`: Maximum number of curves (runs) to actually draw in the overlapping plots (to avoid visually cluttered graphs).
* `GLOBAL_Y_LIM_PDF` / `GLOBAL_Y_LIM_KL`: Fixed limits for the y-axes of the plots, useful for visually comparing different distributions on the same scale.

### What it does and Generated Results
The script draws *M* samples, calculates the BPH and KDE estimates, and measures the Kullback-Leibler (KL) Divergence against the ground truth Erlang distribution.

Upon completion, it creates a folder named `img/YYYYMMDD_erlangKK/` (where `YYYYMMDD` is today's date) containing:
1.  **Summary Table (`summary_table_runs{NUM_SIMULATIONS}_erlangKK.jpg`)**: An image containing a summary table. The columns show the Distribution, *M*, *N*, and the average errors (KL_bph and KL_KDE) calculated over the simulations.
2.  **Distribution Folders (e.g., `erlang_K2/`)**: Inside, you will find the comparison plots (`kde_vs_bernstein_...jpg`). Each plot is a 2x2 figure showing:
    * *Top*: The estimated PDF curves (with transparency) overlapping the black ground truth (Bernstein on the left, KDE on the right).
    * *Bottom*: Boxplots of the KL divergence to display the mean, median, and variance of the error.

---

## 2. Running `grid_search.py`

This script runs an exhaustive analysis by crossing all provided values for *M* and *N*. It is ideal for understanding how the polynomial degree *N* affects the error as the sample size *M* changes.

### Configurations
Unlike the previous file, this one does not use fixed pairs, but rather two separate lists:
* `M_VALUES`: List of sample sizes (e.g., `[27, 68, 163, ...]`).
* `N_VALUES`: List of Bernstein polynomial degrees (e.g., `[8, 16, 32, ...]`).
* The remaining configurations (`K_VALUES`, `NUM_SIMULATIONS`, etc.) keep the same meaning as described in `erlangKK.py`. `GLOBAL_Y_LIM_KL_SENSITIVITY` is added to manage the y-axis of the trend plots.

### What it does and Generated Results
The script iterates over each distribution and calculates the metrics for the Cartesian product of `M_VALUES` and `N_VALUES`. 

The results are saved in the `img/YYYYMMDD_grid_search/` folder and include:
1.  **Summary Table (`summary_table_runs{NUM_SIMULATIONS}_grid_search.jpg`)**: Similar to the one in the previous file, but it will contain many more rows (one for each tested *M* and *N* combination).
2.  **Distribution Folders**: Contain the 2x2 comparison plots for every single (*M*, *N*) combination.
3.  **`bph_sensitivity/` Folder**: Contains the trend plots (`bph_trend_K{K}.jpg`). 
    * **How to read the trend plot**: The X-axis represents the polynomial degree (*N*), while the Y-axis (in logarithmic scale) represents the average KL error. Each colored line represents a different sample size (*M*). This plot is essential for visually identifying the existence of a potential "optimal" *N* for a given *M*, or for observing the overfitting phenomenon if *N* becomes too large.

## Required Dependencies

Minimum Python version: 3.13.0

To run the scripts, make sure you have installed:
* `numpy`
* `scipy`
* `matplotlib`

For a detailed list of dependencies, please refer to the `requirements.txt` file.
