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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
import pandas as pd
import itertools

# Set seed for reproducibility
np.random.seed(123)

# Define attributes
brand = ["N", "P", "H"]  # Netflix, Prime, Hulu
ad = ["Yes", "No"]
price = np.arange(8, 33, 4)

# Generate all possible profiles
profiles = pd.DataFrame(list(itertools.product(brand, ad, price)), columns=["brand", "ad", "price"])
m = len(profiles)

# Assign part-worth utilities (true parameters)
b_util = {"N": 1.0, "P": 0.5, "H": 0.0}
a_util = {"Yes": -0.8, "No": 0.0}
p_util = lambda p: -0.1 * p

# Number of respondents, choice tasks, and alternatives per task
n_peeps = 100
n_tasks = 10
n_alts = 3

# Function to simulate one respondent’s data
def sim_one(id):
    datlist = []

    for t in range(1, n_tasks + 1):
        # Sample 3 alternatives randomly
        sampled_profiles = profiles.sample(n=n_alts).copy()
        sampled_profiles["resp"] = id
        sampled_profiles["task"] = t

        # Compute deterministic portion of utility
        sampled_profiles["v"] = sampled_profiles["brand"].map(b_util) + \
                                sampled_profiles["ad"].map(a_util) + \
                                sampled_profiles["price"].apply(p_util)
        sampled_profiles["v"] = sampled_profiles["v"].round(10)

        # Add Gumbel noise (Type I extreme value)
        sampled_profiles["e"] = -np.log(-np.log(np.random.rand(n_alts)))
        sampled_profiles["u"] = sampled_profiles["v"] + sampled_profiles["e"]

        # Identify chosen alternative
        sampled_profiles["choice"] = (sampled_profiles["u"] == sampled_profiles["u"].max()).astype(int)

        datlist.append(sampled_profiles)

    return pd.concat(datlist)

# Simulate data for all respondents
conjoint_data = pd.concat([sim_one(i) for i in range(1, n_peeps + 1)], ignore_index=True)

# Keep only observable columns
conjoint_data = conjoint_data[["resp", "task", "brand", "ad", "price", "choice"]]

# Show sample
print(conjoint_data.head())
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
# One-hot encode brand and ad
pd.read_csv('projects\hw3\conjoint_data.csv')
df = conjoint_data.copy()
df = pd.get_dummies(df, columns=['brand','ad'], drop_first=False)
# Rename columns for clarity
df.rename(columns={'brand_N':'beta_netflix', 'brand_P':'beta_prime', 'ad_Yes':'beta_ads'}, inplace=True)
# Price is continuous
# Ensure ordering by resp, task
df.sort_values(['resp','task'], inplace=True)

df.head()
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
