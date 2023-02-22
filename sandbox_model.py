import pandas as pd

df = pd.DataFrame()


##
## STATIONARY TEST
##

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df["CDPROJEKT"].values)
plot_acf(df["CDPROJEKT_ln_diff"].values)


from statsmodels.tsa.stattools import adfuller

result = adfuller(df["CDPROJEKT"].values)
print("p-value(CDPROJEKT):", result[1])
result = adfuller(df["CDPROJEKT_ln_diff"].values)
print("p-value(CDPROJEKT_ln_diff):", result[1])

plot_pacf(df["CDPROJEKT_ln_diff"].values)

##
## COPULAS
##

from copulas.visualization import hist_1d, side_by_side

copula = df[["PKNORLEN_ln_diff"]]
hist_1d(copula)


from copulas.univariate import BetaUnivariate

beta = BetaUnivariate()
beta.fit(copula)
beta._params
copula

from copulas.visualization import compare_1d

sampled = beta.sample(1000)

hist_1d(copula, label="Real")
hist_1d(sampled, label="Synthetic")

cumulative_distribution = beta.cumulative_distribution(copula.values.flatten())
pd.DataFrame(
    {
        "data": copula.values.flatten(),
        "cumulative distribution": cumulative_distribution,
    }
).sort_values("data").set_index("data").plot()

probability_density = beta.pdf(copula.values.flatten())

pd.DataFrame(
    {"data": copula.values.flatten(), "probability_density": probability_density}
).sort_values("data").set_index("data").plot()

## CTP (not good)
values = []
for index, row in ct.iterrows():
    c_tab = []
    for col in row:
        c_tab.append([pol_f(col) for pol_f in l_poln])
    values.append(c_tab)
ctp = np.array(values)
ctp.shape

# scipy.moment
from scipy.stats import moment

moment(df[COMPANY_NAME + "_ln_diff"].values.flatten(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
