# FOLD BOD Model

Python code for the **Fractional-Order Logistic Decay (FOLD)** model of
biochemical oxygen demand (BOD), with the classical first-order model for
comparison and basic wastewater-treatment design calculations.

> Devassy, M. (2026). Improved BOD Modelling Using the FOLD Approach for Wastewater
> Treatment Design. *Water and Environment Journal*. https://doi.org/10.1111/wej.70098

This repository is a **tool for applying the FOLD model to your own BOD data**.
It is not a reproduction of every table in the article.

---

## The model

```
First-order:  BOD(t) = L0 · [1 − exp(−k·t)]
FOLD:         BOD(t) = L0 · [1 − (1 + k·t^α / γ)^(−γ)]
```

| Symbol | Meaning |
|---|---|
| L0 | ultimate BOD (mg/L) |
| k | rate parameter |
| α | fractional-order exponent, 0 < α ≤ 1 (memory / sub-diffusive behaviour) |
| γ | saturation parameter, γ > 0 (smaller γ = stronger saturation) |

FOLD contains two classical models as special cases:
- α = 1, γ → ∞ gives **first-order** kinetics
- α = 1, γ = 1 gives **second-order** kinetics

---

## Use FOLD in 3 steps

**1. Install** (Python 3.9 or later)

```bash
git clone https://github.com/<your-username>/FOLD-BOD-model.git
cd FOLD-BOD-model
pip install -r requirements.txt
```

**2. Put your data in a CSV file** (see `examples/my_data_template.csv`)

```
sample,time_day,bod_mg_L
MySample,1,211
MySample,2,322
MySample,3,383
MySample,4,423
MySample,5,450
```

Any number of samples and time points can be used. Do not include t = 0.

**3. Run**

```bash
python fold_fit.py mydata.csv
```

You get, for each sample:
- first-order and FOLD parameters
- R², RMSE, AIC and BIC for both models
- activated-sludge and aerated-lagoon retention time, volume and blower HP
- a plot of observed BOD with both fitted curves

Results are saved in the `results/` folder (`fold_results.csv` and one PNG per sample).

### Design options

```bash
python fold_fit.py mydata.csv --flow 20000 --effluent 30 --loading 400 --out myresults
```

| Option | Meaning | Default |
|---|---|---|
| `--flow` | design flow (m³/d) | 35000 |
| `--effluent` | target effluent BOD (mg/L) | 20 |
| `--loading` | activated-sludge BOD loading (g BOD/m³/d) | 500 |
| `--out` | output folder | results |
| `--no-plots` | skip figures | off |

### Use inside your own Python code

```python
from fold_bod import fit_fold, fold, goodness_of_fit

t = [1, 2, 3, 4, 5]
y = [211, 322, 383, 423, 450]
L0, k, alpha, gamma = fit_fold(t, y)
print(L0, k, alpha, gamma)          # 535.92  0.5623  1.0  2.2562
pred = fold(t, L0, k, alpha, gamma)
print(goodness_of_fit(y, pred, 4))
```

---

## Example

`examples/example_bod_data.csv` contains five dilution-method BOD series for raw
domestic wastewater from Sibil et al. (2014), as used in the article.

```bash
python fold_fit.py examples/example_bod_data.csv
```

The fitted parameters and fit statistics match Tables 4 and 6 of the article.

---

## Fitting method

Bounded non-linear least squares (`scipy.optimize.least_squares`) with many
starting points, to avoid local minima. Default limits (editable in
`SETTINGS` in `fold_bod.py`):

| Parameter | Lower | Upper |
|---|---|---|
| L0 (FOLD) | max observed BOD | 5/3 × max observed BOD |
| k | 0.00001 | 10 |
| α | 0.05 | 1 |
| γ | 0.01 | 10 |

### Important note on short data series

FOLD has four parameters. With only about five data points, L0 or γ can end up
at a fitting limit. The program flags this ("FOLD parameter(s) at fitting
limit"). Values at a limit are not well determined by the data and should be
interpreted with caution. Longer, denser time series (for example respirometric
data) give more reliable FOLD parameters.

---

## Tests

```bash
python tests/test_fold.py
```

Checks that FOLD reduces to first-order and second-order kinetics, that known
parameters are recovered, and that the example result is reproduced.

---

## Data source for the example

Sibil, R., Berkun, M. and Bekiroglu, S. (2014). The comparison of different
mathematical methods to determine the BOD parameters, a new developed method and
impacts of these parameters variations on the design of WWTPs. *Applied
Mathematical Modelling*, 38(2), 641–658. https://doi.org/10.1016/j.apm.2013.07.013

## How to cite

If you use this code, please cite the article above and this software
(see `CITATION.cff`, or the Zenodo DOI badge once available).

## Licence

MIT — free to use, modify and share with attribution. See `LICENSE`.

## Contact

Manu Devassy, Department of Civil Engineering, Government Engineering College
Thrissur, Kerala, India — manudevassy.2014@gmail.com
