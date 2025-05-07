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
import scipy.stats
from scipy.special import gammaln
import statsmodels.api as sm
import scipy.optimize as sp
import scipy.stats as stats
from scipy.optimize import minimize_scalar,minimize, approx_fprime
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
def poisson_regression_loglikelihood(beta, Y, X):
    beta = np.array(beta, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    linear_pred = np.dot(X, beta)
    lambda_i = np.exp(np.clip(linear_pred, -30, 30))
    ll_components = -lambda_i + Y * np.log(lambda_i)
    return -np.sum(ll_components)
#
#
#
#
#
#
#
# Add squared age and region dummies
df["age_squared"] = df["age"] ** 2
region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

# Combine into design matrix
X_data = pd.concat([
    pd.DataFrame({'intercept': 1}, index=df.index),
    df[['age', 'age_squared', 'iscustomer']],
    region_dummies
], axis=1)

X = X_data.values.astype(np.float64)
y = df['patents'].values.astype(np.float64)
#
#
#
#
#
#
#
glm_model = sm.GLM(y, X, family=sm.families.Poisson())
glm_results = glm_model.fit()
initial_beta = glm_results.params
#
#
#
#
#
#
#

result_poisson_reg = minimize(
    poisson_regression_loglikelihood,
    initial_beta,
    args=(y, X),
    method='BFGS',
    options={'disp': True}
)

beta_estimates = result_poisson_reg.x
#
#
#
#
#
#
#
def hessian(func, x, epsilon=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x1, x2, x3, x4 = x.copy(), x.copy(), x.copy(), x.copy()
            if i == j:
                x1[i] += epsilon
                x2[i] -= epsilon
                hess[i, i] = (func(x1) + func(x2) - 2 * func(x)) / (epsilon ** 2)
            else:
                x1[i] += epsilon; x1[j] += epsilon
                x2[i] += epsilon; x2[j] -= epsilon
                x3[i] -= epsilon; x3[j] += epsilon
                x4[i] -= epsilon; x4[j] -= epsilon
                hess[i, j] = (func(x1) + func(x4) - func(x2) - func(x3)) / (4 * epsilon ** 2)
                hess[j, i] = hess[i, j]
    return hess

H = hessian(lambda b: poisson_regression_loglikelihood(b, y, X), beta_estimates)
cov_matrix = np.linalg.inv(H)
std_errors = np.sqrt(np.diag(cov_matrix))

z_values = beta_estimates / std_errors
p_values = 2 * (1 - np.abs(scipy.stats.norm.cdf(z_values)))

custom_results = pd.DataFrame({
    'Coefficient': beta_estimates,
    'Std. Error': std_errors,
    'z-value': z_values,
    'p-value': p_values
}, index=X_data.columns)

custom_results.round(4)
#
#
#
#
#
#
#
sm_results = pd.DataFrame({
    'Coefficient': glm_results.params,
    'Std. Error': glm_results.bse,
    'z-value': glm_results.tvalues,
    'p-value': glm_results.pvalues
}, index=X_data.columns)

sm_results.round(4)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
X_0 = X.copy()
X_1 = X.copy()
iscustomer_idx = list(X_data.columns).index('iscustomer')
X_0[:, iscustomer_idx] = 0
X_1[:, iscustomer_idx] = 1

lambda_0 = np.exp(np.dot(X_0, beta_estimates))
lambda_1 = np.exp(np.dot(X_1, beta_estimates))

differences = lambda_1 - lambda_0
avg_effect = np.mean(differences)
perc_increase = 100 * avg_effect / np.mean(lambda_0)

print("Average predicted increase in patents:", round(avg_effect, 3), "per firm")
print("Percent increase:", round(perc_increase, 2), "%")
#
#
#
#
#
#
#
#
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(differences, bins=30, alpha=0.7)
plt.axvline(avg_effect, color='red', linestyle='--', 
            label=f'Avg Effect: {avg_effect:.2f} patents')
plt.xlabel('Increase in Predicted Patents')
plt.ylabel('Number of Firms')
plt.title('Effect of Using Blueprinty Software')
plt.legend()
plt.grid(True)
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
# Load the AirBnB data
airbnb_df = pd.read_csv('airbnb.csv',index_col=0)

# Display the first few rows
airbnb_df.head()
#
#
#
# Check the data structure and missing values
airbnb_df.info()
#
#
#
# Summary statistics
airbnb_df.describe()
#
#
#
#
#

# Select relevant columns
model_cols = ['room_type', 'bathrooms', 'bedrooms',
    'price' ,'number_of_reviews',  
    'review_scores_cleanliness', 'review_scores_location',
    'review_scores_value', 'instant_bookable', 
]

# Drop missing values
airbnb_clean = airbnb_df[model_cols].dropna()

# Convert categorical variables into dummy variables
airbnb_clean = pd.get_dummies(airbnb_clean, columns=["room_type", "instant_bookable"], drop_first=True)

# Preview cleaned data
airbnb_clean.head()
#
#
#
#
#
#
#
import statsmodels.api as sm

# Define target and features
y = airbnb_clean["number_of_reviews"]
X = airbnb_clean.drop(columns=["number_of_reviews"])
X = sm.add_constant(X)  # Add intercept

# Fit Poisson regression
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# Display summary
results.summary()
#
#
#
#
