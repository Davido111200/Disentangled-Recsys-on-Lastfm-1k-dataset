import numpy as np
import pandas as pd

#read in full dataset from https://www.dropbox.com/s/wlhkyz8yn51cpnk/360k_users.zip?dl=0
col_names = ["user_id","artist_mbid","artist_name","plays"]
df = pd.read_csv("/home/daitama/Desktop/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", sep="\t", header=None,
                 names=col_names)


def get_users(df, n):
    """
    Input:
    - df (pd.Dataframe): main dataframe to sample from
    - n (int): number of users to sample

    Returns:
    - df_sample (pd.Dataframe): sampled dataframe
    """
    sample_userid = df["user_id"].unique()
    sample_userid = np.random.choice(sample_userid, size=n, replace=False)

    # grab rows with sample user id
    df_sample = df[df.user_id.isin(sample_userid)].reset_index(drop=True)

    return df_sample


# remove faulty user_id
df = df[df.user_id != "sep 20, 2008"]

sizes = [9000, 20000, 40000, 60000, 80000]

for size in sizes:
    get_users(df, size).to_csv("lastfm_" + str(size) + "_users.csv")



