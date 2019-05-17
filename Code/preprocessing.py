# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)


def import_data():

    varol_users = np.loadtxt("Z:\ioptyu2\Desktop\gitDissertation\local\Data\\varol_2017.dat")


    col_name = ["UserID","CreatedAt","CollectedAt","Followings","Followers","Tweets","NameLength","BioLength"]
    bot_users = pd.read_csv("Z:/ioptyu2/Desktop/gitDissertation/local/Data/social_honeypot_icwsm_2011/content_polluters.txt",
                                     sep="\t",
                                     names = col_name)


    legit_users = pd.read_csv("Z:/ioptyu2/Desktop/gitDissertation/local/Data/social_honeypot_icwsm_2011/legitimate_users.txt",
                                      sep="\t",
                                      names = col_name)
    
    return [varol_users,bot_users,legit_users,col_name]

#col_name = ["UserID","TweetID","Tweet","CreatedAt"]
#bot_tweets = pd.read_csv("Z:/ioptyu2/Desktop/gitDissertation/local/Data/social_honeypot_icwsm_2011/content_polluters_tweets.txt",
#                                  sep="\t",
#                                  names = col_name)
