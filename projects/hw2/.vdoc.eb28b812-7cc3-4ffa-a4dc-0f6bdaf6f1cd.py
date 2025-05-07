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
def poisson_regression_loglikelihood(beta, Y, X):
    beta = np.asarray(beta)
    lambda_i = np.exp(X @ beta)
    ll_components = -lambda_i + Y * np.log(lambda_i) - gammaln(Y + 1)
    return -np.sum(ll_components)
#
#
#
#
#
# Create age-squared variable
df['age_squared'] = df['age'] ** 2

# Get dummy variables for regions
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Combine all variables into the design matrix
X_data = pd.concat([
    pd.DataFrame({'intercept': 1}, index=df.index),
    df[['age', 'age_squared', 'iscustomer']],
    region_dummies
], axis=1)

# Extract X matrix and y vector
X = X_data.values
y = df['patents'].values

# Initial guess for beta
initial_beta = np.zeros(X.shape[1])

# Find the MLE for beta
result_poisson_reg = opt.minimize(
    poisson_regression_loglikelihood, 
    initial_beta, 
    args=(y, X),
    method='BFGS',
    options={'disp': True}
)

# Get the Hessian matrix to calculate standard errors
# We can approximate the Hessian using the Fisher Information Matrix
# For large samples, the covariance matrix is approximately the inverse of the Fisher Information
hessian = sm.tools.hessian_approx(
    lambda params: poisson_regression_loglikelihood(params, y, X), 
    result_poisson_reg.x
)

# Calculate standard errors from the diagonal of the inverse Hessian
cov_matrix = np.linalg.inv(hessian)
std_errors = np.sqrt(np.diag(cov_matrix))

# Create a table of coefficients and standard errors
coef_names = X_data.columns
beta_estimates = result_poisson_reg.x
z_values = beta_estimates / std_errors
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

results_table = pd.DataFrame({
    'Coefficient': beta_estimates,
    'Std. Error': std_errors,
    'z-value': z_values,
    'p-value': p_values
}, index=coef_names)

print(results_table)
#
#
#
#
#
# Create a Poisson model using statsmodels
poisson_model = sm.GLM(
    y,
    X,
    family=sm.families.Poisson()
)

# Fit the model
poisson_results = poisson_model.fit()

# Display the summary
print(poisson_results.summary())
#
#
#
#
#
# Calculate predicted number of patents for each firm if they were not a customer
X_0 = X.copy()
X_0[:, X_data.columns.get_loc('iscustomer')] = 0  # Set iscustomer to 0 for all firms

# Calculate predicted number of patents for each firm if they were a customer
X_1 = X.copy()
X_1[:, X_data.columns.get_loc('iscustomer')] = 1  # Set iscustomer to 1 for all firms

# Calculate predicted counts
lambda_0 = np.exp(X_0 @ result_poisson_reg.x)
lambda_1 = np.exp(X_1 @ result_poisson_reg.x)

# Calculate the difference in predicted counts
diff = lambda_1 - lambda_0

# Calculate average effect
avg_effect = np.mean(diff)
print(f"Average effect of using Blueprinty software: {avg_effect:.4f} additional patents")

# Calculate percentage increase
perc_increase = 100 * avg_effect / np.mean(lambda_0)
print(f"This represents a {perc_increase:.2f}% increase in the number of patents")

# Plot the distribution of the effect
plt.figure(figsize=(10, 6))
plt.hist(diff, bins=30)
plt.axvline(avg_effect, color='red', linestyle='--', 
            label=f'Mean Effect ({avg_effect:.4f} patents)')
plt.xlabel('Increase in Number of Patents')
plt.ylabel('Frequency')
plt.title('Distribution of the Effect of Using Blueprinty Software')
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
