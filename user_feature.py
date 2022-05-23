# Import necessary modules
import pandas as pd
import numpy as np

# Read user profiles into a dataframe
col_names = ['id', 'gender', 'age', 'country', 'date']
profile = pd.read_csv('/home/daitama/Desktop/lastfm-dataset-360K/usersha1-profile.tsv', sep='\t', header=None,
                      names=col_names)
profile = profile.drop(['date'], axis=1)

# Read file containing sampled users into a dataframe
df = pd.read_csv('lastfm_9000_users.csv', na_filter=False)
df = df.drop(['Unnamed: 0'], axis=1)
users = list(np.sort(df.user_id.unique()))

# Initialize an empty dataframe to store user_features
user_features = pd.DataFrame()

# Iterate through each user in sample and extract relevant profile information
for user in users:
    user_features = user_features.append(profile.loc[profile['id'] == user])

# One-hot-encode gender variables
one_hot_gender = pd.get_dummies(user_features['gender'])
user_features = user_features.drop('gender', axis=1)
user_features = user_features.join(one_hot_gender)

# One-hot-encode country variables
one_hot_country = pd.get_dummies(user_features['country'])
user_features = user_features.drop('country', axis=1)
user_features = user_features.join(one_hot_country)

# One-hot-encode age variables
one_hot_age = pd.get_dummies(user_features['age'])
user_features = user_features.drop('age', axis=1)
user_features = user_features.join(one_hot_age)

# Drop unrealisitic ages (outside of range 12-97)
user_features_clean = user_features.drop([-1337.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 11.0, 100.0, 101.0, 102.0, 107.0,
                                          108.0, 109.0], axis=1)

# Export processed user features to CSV
user_features_clean.to_csv('user_features_9000.csv')







