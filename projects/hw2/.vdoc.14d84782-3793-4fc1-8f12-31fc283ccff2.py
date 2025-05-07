# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm
import scipy.optimize as opt
import scipy.stats as stats
from scipy.optimize import minimize_scalar

```
#
df = pd.read_csv("blueprinty.csv")  
df.head()
#
#
#
#
# Map numeric to label
df["customer_label"] = df["iscustomer"].map({0: "Non-Customer", 1: "Customer"})

# Create histogram
ax = sns.histplot(data=df, x="patents", hue="customer_label", multiple="dodge", 
                  bins=30, palette={"Non-Customer": "orange", "Customer": "skyblue"})

# Add title and axis labels
plt.title("Histogram of Patents by Customer Status")
plt.xlabel("Number of Patents")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Summary statistics by customer status
df.groupby("customer_label")["patents"].agg(["mean", "std", "count"]).reset_index()

```
#
#
#
#
#
#
#
#
#
#
#
# Region distribution
region_ct = pd.crosstab(df["region"], df["customer_label"], normalize="columns") * 100
region_ct.columns= ["Customer (%)", "Non-Customer (%)"]
region_ct.index.name = "Region"
region_ct = region_ct.reset_index()
print(region_ct.to_string(index=False))
```
#
# Age distribution
age_summary = df.groupby("customer_label")["age"].agg(["mean", "std", "min", "max"]).round(2)
age_summary = age_summary.reset_index()
age_summary.columns = ["Customer Status", "Mean Age", "Std Dev", "Min Age", "Max Age"]
age_summary
```
#
from scipy.stats import gaussian_kde
ages = np.linspace(df["age"].min(), df["age"].max(), 300)
kde_non = gaussian_kde(df.loc[df["customer_label"]=="Non-Customer","age"])(ages)
kde_cust = gaussian_kde(df.loc[df["customer_label"]=="Customer","age"])(ages)

plt.plot(ages, kde_non, label="Non-Customer", color="tab:orange")
plt.plot(ages, kde_cust, label="Customer",    color="tab:blue")
plt.fill_between(ages, kde_non, alpha=0.3, color="tab:orange")
plt.fill_between(ages, kde_cust, alpha=0.3, color="tab:blue")

plt.title("Density of Firm Age by Customer Status")
plt.xlabel("Firm Age")
plt.ylabel("Density")
plt.legend(title="Customer Status")
plt.grid(True)
plt.show()
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
def poisson_loglikelihood(lambda_val, Y):
    return -lambda_val + Y * np.log(lambda_val) - gammaln(Y + 1)

def poisson_loglikelihood_sum(lambda_val, Y):
    return np.sum(-lambda_val + Y * np.log(lambda_val) - gammaln(Y + 1))
```
#
#
#
# Get the observed patents data
observed_patents = df['patents'].values

# Choose a reasonable range of lambda values based on data
lambda_range = np.linspace(0.1, 20, 100)
ll_values = np.zeros(len(lambda_range))

# Calculate log-likelihood for each lambda
for i, lam in enumerate(lambda_range):
    ll_values[i] = np.sum([poisson_loglikelihood(lam, y) for y in observed_patents])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lambda_range, ll_values)
plt.xlabel('Lambda')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood of Poisson Model for Different Lambda Values')
plt.axvline(x=observed_patents.mean(), color='red', linestyle='--', 
            label=f'Sample Mean (λ_MLE = {observed_patents.mean():.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Define the negative log-likelihood (for minimization)
def neg_poisson_loglikelihood_sum(lambda_val, Y):
    return -poisson_loglikelihood_sum(lambda_val, Y)

# Find the MLE using optimization
initial_guess = 1.0  # Starting point for optimization
result = minimize(neg_poisson_loglikelihood_sum, 
                  x0=[1.0], 
                  args=(observed_patents,), 
                  method='L-BFGS-B',
                  bounds=[(1e-6, None)])

print(f"MLE estimate for lambda: {result.x[0]:.4f}")
print(f"Sample mean: {observed_patents.mean():.4f}")
print(f"Are they equal? {np.isclose(result.x[0], observed_patents.mean())}")
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 1) Build design matrix
df["age2"] = df["age"]**2
region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

X_df = pd.concat([
    pd.Series(1, index=df.index, name="intercept"),
    df[["age", "age2", "iscustomer"]],
    region_dummies
], axis=1)

# ensure numeric dtype
X_df = X_df.astype(float)
y = df["patents"].astype(float)

# 2) Fit Poisson GLM
poisson_mod = sm.GLM(y, X_df, family=sm.families.Poisson())
poisson_res = poisson_mod.fit()

# 3) Table of coefficients & SEs
coef_tbl = pd.DataFrame({
    "Estimate":  poisson_res.params,
    "Std. Error": poisson_res.bse,
    "z-value":    poisson_res.tvalues,
    "p-value":    poisson_res.pvalues
})
display(coef_tbl.round(4))
#
#
#
#
#
#
#
# build X0 (all iscustomer=0) and X1 (all iscustomer=1)
X0 = X_df.copy(); X0["iscustomer"] = 0
X1 = X_df.copy(); X1["iscustomer"] = 1

mu0 = np.exp((X0 @ poisson_res.params).values)
mu1 = np.exp((X1 @ poisson_res.params).values)
delta = mu1 - mu0

print("Average Δ patents:", round(delta.mean(), 3))
print("Percent increase:", round(100 * delta.mean() / mu0.mean(), 2), "%")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
def poisson_regression_loglikelihood(beta, Y, X):
    """
    Calculate the log-likelihood for a Poisson regression model
    
    Parameters:
    beta (array-like): Coefficient vector
    Y (array-like): Vector of observed counts
    X (array-like): Design matrix of covariates
    
    Returns:
    float: The negative log-likelihood value (for minimization)
    """
    # Convert inputs to numpy arrays
    beta = np.array(beta, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    
    # Calculate lambda for each observation: λᵢ = exp(Xᵢ'β)
    linear_pred = np.dot(X, beta)
    lambda_i = np.exp(linear_pred)
    
    # Calculate log-likelihood components for each observation
    # log(f(Y|λ)) = -λ + Y*log(λ) - log(Y!)
    # Since log(Y!) is constant for optimization, we can omit it
    ll_components = -lambda_i + Y * np.log(lambda_i)
    
    # Return the negative sum of log-likelihoods (for minimization)
    return -np.sum(ll_components)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
