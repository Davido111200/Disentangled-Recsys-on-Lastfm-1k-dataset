# Import necessary modules
import pandas as pd
import numpy as np
import pylast

# Read CSV of user-artist interactions
df = pd.read_csv('lastfm_9000_users.csv', na_filter=False)

df = df.drop(['Unnamed: 0'], axis=1)

# Grab unique users/artist IDS
users = list(np.sort(df.user_id.unique()))
artists = list(df.artist_mbid.unique())
plays = list(df.plays)

# Enter API and account authentication details for Last.fm
API_KEY = "8062ac0fd8bc03b18c551bbd5111babf"
API_SECRET = "2e12eef90b1ca82f7cb2d7bbc0bd33ed"
username = "dai1112"
password_hash = pylast.md5("12345@Bc")

# Connect to API and extract artist info by MBID
network = pylast.LastFMNetwork(api_key = API_KEY, api_secret = API_SECRET,
                               username = username, password_hash = password_hash)
artist = network.get_artist_by_mbid(artists[0])

# Initialize dictionaries to store top 5 tags for each artist
tag1 = {id:0 for id in artists}
tag2 = {id:0 for id in artists}
tag3 = {id:0 for id in artists}
tag4 = {id:0 for id in artists}
tag5 = {id:0 for id in artists}
tags_dict = {id:[] for id in artists}

# Access top tags for each artist through API and store tags in dictionaries
for index in artists:
    try:
        artist = network.get_artist_by_mbid(index)
        toptags = artist.get_top_tags(limit=5)
        for i in toptags:
            tags_dict[index].append(i[0].get_name())
    except:
        tags_dict[index].append('')

# Fill up empty tags for artists with less than 5 tags
for key, item in tags_dict.items():
    while len(item) < 5:
        item.append('')


# Create a dataframe of artist_features from acquired tags
artist_features=pd.DataFrame.from_dict(tags_dict, orient='index')

# One-hot-encode resulting dataframe to be used for LightFM model
artist_features = artist_features.stack().str.get_dummies().sum(level=0)

# Write dataframe to CSV
artist_features.to_csv('artist_features_9000.csv')


