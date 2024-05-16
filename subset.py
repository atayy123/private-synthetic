import pandas as pd
import json

# Create a subset of adult dataset (with 5 features)

df = pd.read_csv("data/adult.csv")

with open("data/adult-domain.json") as json_file:
    domain = json.load(json_file)

# let's select the features with fewer bins
# such as: marital-status, relationship, race, sex, income>50K
features = ["marital-status", "relationship", "race", "sex", "income>50K"]

# extract features from csv file
df_subs = df[features]
df_subs.to_csv("data/subset5.csv", index=False)

domain_subs = dict((k, domain[k]) for k in features if k in domain)
with open("data/subset5-domain.json", "w") as domain_json:
    json.dump(domain_subs,domain_json)
