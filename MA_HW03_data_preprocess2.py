#!/usr/bin/env python
# coding: utf-8

# ### Marketing Analytics HW03
# ### Group: zj440 + xh2102 + elm454 + xr2008

# First, we import all needed packages and the two csv files, replacing NA values as "None", and extract some information from the two files.

# In[1]:


import pandas as pd
import json
import numpy as np


# In[2]:


df = pd.read_csv("attribution_allocation_student_data.csv")
df.head()


# In[3]:


df=df.fillna('None')
df.head()


# In[4]:


df_channel = pd.read_csv("channel_spend_student_data.csv")
df_channel.describe()


# In[5]:


df_channel_renamed=df_channel.rename(columns={"Unnamed: 0": "Tier", "0": "Details"})
df_channel_renamed


# We noticed that "Details" column contains useful information, so we use for loop to get things out.

# In[6]:


allocation_method = list()
for i in range(len(df_channel_renamed)):
    r = json.loads(df_channel_renamed["Details"][i].replace("\'", "\""))
    r['tier'] = i+1
    allocation_method.append(r)

df_current_allocation = pd.DataFrame(allocation_method)
df_current_allocation


# In[7]:


df_ttl = pd.merge(df, df_current_allocation,how='left', on='tier')
df_ttl


# Here, we have a dataframe containing integrated information.

# #### Part 1 attribution: 
# Allocate conversions by channel (social, organic_search, referral, email, paid_search, display, direct) and evaluate effectiveness
# 
# • Test 3 methods for allocation
# 
# • Calculate average CAC for each of the channels
# 
# • Discuss observations and potential conclusions from CAC calculations

# #### Method 01: First interaction

# In[8]:


new_df = df[df["convert_TF"]==True]
new_df['touch1'].unique()


# In[9]:


df_1 = new_df['touch1']
df_1.value_counts()


# #### Method 02: Linear

# In[10]:


df_ttl_T = df_ttl[df_ttl["convert_TF"]==True]
df_ttl_T


# In[11]:


import warnings

#To speed things up
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_ttl_T["t1"] = np.where(df_ttl_T['touch1'] == "None", 0, 1)
    df_ttl_T["t2"] = np.where(df_ttl_T['touch2'] == "None", 0, 1)
    df_ttl_T["t3"] = np.where(df_ttl_T['touch3'] == "None", 0, 1)
    df_ttl_T["t4"] = np.where(df_ttl_T['touch4'] == "None", 0, 1)
    df_ttl_T["t5"] = np.where(df_ttl_T['touch5'] == "None", 0, 1)
    df_ttl_T["ttotal"] = df_ttl_T["t1"] + df_ttl_T["t2"] + df_ttl_T["t3"] + df_ttl_T["t4"] + df_ttl_T["t5"]
    df_ttl_T["touch_email_1"] = np.where(df_ttl_T['touch1'] == "email", 1, 0)
    df_ttl_T["touch_email_2"] = np.where(df_ttl_T['touch2'] == "email", 1, 0)
    df_ttl_T["touch_email_3"] = np.where(df_ttl_T['touch3'] == "email", 1, 0)
    df_ttl_T["touch_email_4"] = np.where(df_ttl_T['touch4'] == "email", 1, 0)
    df_ttl_T["touch_email_5"] = np.where(df_ttl_T['touch5'] == "email", 1, 0)
    df_ttl_T["email_total"] = df_ttl_T["touch_email_1"] + df_ttl_T["touch_email_2"] + df_ttl_T["touch_email_3"] + df_ttl_T["touch_email_4"] + df_ttl_T["touch_email_5"]
    df_ttl_T["touch_referral_1"] = np.where(df_ttl_T['touch1'] == "referral", 1, 0)
    df_ttl_T["touch_referral_2"] = np.where(df_ttl_T['touch2'] == "referral", 1, 0)
    df_ttl_T["touch_referral_3"] = np.where(df_ttl_T['touch3'] == "referral", 1, 0)
    df_ttl_T["touch_referral_4"] = np.where(df_ttl_T['touch4'] == "referral", 1, 0)
    df_ttl_T["touch_referral_5"] = np.where(df_ttl_T['touch5'] == "referral", 1, 0)
    df_ttl_T["referral_total"] = df_ttl_T["touch_referral_1"] + df_ttl_T["touch_referral_2"] + df_ttl_T["touch_referral_3"] + df_ttl_T["touch_referral_4"] + df_ttl_T["touch_referral_5"]
    df_ttl_T["touch_paid_search_1"] = np.where(df_ttl_T['touch1'] == "paid_search", 1, 0)
    df_ttl_T["touch_paid_search_2"] = np.where(df_ttl_T['touch2'] == "paid_search", 1, 0)
    df_ttl_T["touch_paid_search_3"] = np.where(df_ttl_T['touch3'] == "paid_search", 1, 0)
    df_ttl_T["touch_paid_search_4"] = np.where(df_ttl_T['touch4'] == "paid_search", 1, 0)
    df_ttl_T["touch_paid_search_5"] = np.where(df_ttl_T['touch5'] == "paid_search", 1, 0)
    df_ttl_T["paid_search_total"] = df_ttl_T["touch_paid_search_1"] + df_ttl_T["touch_paid_search_2"] + df_ttl_T["touch_paid_search_3"] + df_ttl_T["touch_paid_search_4"] + df_ttl_T["touch_paid_search_5"]
    df_ttl_T["touch_direct_1"] = np.where(df_ttl_T['touch1'] == "direct", 1, 0)
    df_ttl_T["touch_direct_2"] = np.where(df_ttl_T['touch2'] == "direct", 1, 0)
    df_ttl_T["touch_direct_3"] = np.where(df_ttl_T['touch3'] == "direct", 1, 0)
    df_ttl_T["touch_direct_4"] = np.where(df_ttl_T['touch4'] == "direct", 1, 0)
    df_ttl_T["touch_direct_5"] = np.where(df_ttl_T['touch5'] == "direct", 1, 0)
    df_ttl_T["direct_total"] = df_ttl_T["touch_direct_1"] + df_ttl_T["touch_direct_2"] + df_ttl_T["touch_direct_3"] + df_ttl_T["touch_direct_4"] + df_ttl_T["touch_direct_5"]
    df_ttl_T["touch_display_1"] = np.where(df_ttl_T['touch1'] == "display", 1, 0)
    df_ttl_T["touch_display_2"] = np.where(df_ttl_T['touch2'] == "display", 1, 0)
    df_ttl_T["touch_display_3"] = np.where(df_ttl_T['touch3'] == "display", 1, 0)
    df_ttl_T["touch_display_4"] = np.where(df_ttl_T['touch4'] == "display", 1, 0)
    df_ttl_T["touch_display_5"] = np.where(df_ttl_T['touch5'] == "display", 1, 0)
    df_ttl_T["display_total"] = df_ttl_T["touch_display_1"] + df_ttl_T["touch_display_2"] + df_ttl_T["touch_display_3"] + df_ttl_T["touch_display_4"] + df_ttl_T["touch_display_5"]
    df_ttl_T["touch_social_1"] = np.where(df_ttl_T['touch1'] == "social", 1, 0)
    df_ttl_T["touch_social_2"] = np.where(df_ttl_T['touch2'] == "social", 1, 0)
    df_ttl_T["touch_social_3"] = np.where(df_ttl_T['touch3'] == "social", 1, 0)
    df_ttl_T["touch_social_4"] = np.where(df_ttl_T['touch4'] == "social", 1, 0)
    df_ttl_T["touch_social_5"] = np.where(df_ttl_T['touch5'] == "social", 1, 0)
    df_ttl_T["social_total"] = df_ttl_T["touch_social_1"] + df_ttl_T["touch_social_2"] + df_ttl_T["touch_social_3"] + df_ttl_T["touch_social_4"] + df_ttl_T["touch_social_5"]
    df_ttl_T["touch_organic_search_1"] = np.where(df_ttl_T['touch1'] == "organic_search", 1, 0)
    df_ttl_T["touch_organic_search_2"] = np.where(df_ttl_T['touch2'] == "organic_search", 1, 0)
    df_ttl_T["touch_organic_search_3"] = np.where(df_ttl_T['touch3'] == "organic_search", 1, 0)
    df_ttl_T["touch_organic_search_4"] = np.where(df_ttl_T['touch4'] == "organic_search", 1, 0)
    df_ttl_T["touch_organic_search_5"] = np.where(df_ttl_T['touch5'] == "organic_search", 1, 0)
    df_ttl_T["organic_search_total"] = df_ttl_T["touch_organic_search_1"] + df_ttl_T["touch_organic_search_2"] + df_ttl_T["touch_organic_search_3"] + df_ttl_T["touch_organic_search_4"] + df_ttl_T["touch_organic_search_5"]
    df_ttl_T["email_total%"] = df_ttl_T["email_total"] / df_ttl_T["ttotal"]
    df_ttl_T["referral_total%"] = df_ttl_T["referral_total"] / df_ttl_T["ttotal"]
    df_ttl_T["paid_search_total%"] = df_ttl_T["paid_search_total"] / df_ttl_T["ttotal"]
    df_ttl_T["direct_total%"] = df_ttl_T["direct_total"] / df_ttl_T["ttotal"]
    df_ttl_T["display_total%"] = df_ttl_T["display_total"] / df_ttl_T["ttotal"]
    df_ttl_T["social_total%"] = df_ttl_T["social_total"] / df_ttl_T["ttotal"]
    df_ttl_T["organic_search_total%"] = df_ttl_T["organic_search_total"] / df_ttl_T["ttotal"]
    
df_ttl_T


# In[12]:


email_total_sum = df_ttl_T["email_total%"].sum()
referral_total_sum = df_ttl_T["referral_total%"].sum()
paid_search_total_sum = df_ttl_T["paid_search_total%"].sum()
direct_total_sum = df_ttl_T["direct_total%"].sum()
display_total_sum = df_ttl_T["display_total%"].sum()
social_total_sum = df_ttl_T["social_total%"].sum()
organic_search_total_sum = df_ttl_T["organic_search_total%"].sum()

total_sums = [["Email total", email_total_sum], ["Referral total", referral_total_sum], ["Paid Search total", paid_search_total_sum], ["Direct total", direct_total_sum], ["Social total", social_total_sum], ["Organic Search total", organic_search_total_sum]]

total_sums.sort(key=lambda x:x[1])
total_sums.reverse()
total_sums


# In[13]:


for i in total_sums:
    print(str(i[0]) + ": " + str(i[1]))


# In[14]:


linear_total_sums_df = pd.DataFrame.from_records(total_sums)
linear_total_sums_df


# #### Method 03: Position Based

# In[15]:


df_ttl_T2 = df_ttl_T.copy()
df_ttl_T2


# In[16]:


#To speed things up. And I get how this is overkill, but I really really wanted to vectorize it and so yeah
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_ttl_T2["count1"] = np.where(df_ttl_T2['ttotal'] == 1, 1, 0)
    df_ttl_T2["count2"] = np.where(df_ttl_T2['ttotal'] == 2, 1, 0)
    df_ttl_T2["count3"] = np.where(df_ttl_T2['ttotal'] == 3, 1, 0)
    df_ttl_T2["count4"] = np.where(df_ttl_T2['ttotal'] == 4, 1, 0)
    df_ttl_T2["count5"] = np.where(df_ttl_T2['ttotal'] == 5, 1, 0)
    #If there is one touch
    df_ttl_T2["touch_email_1"] = np.where(df_ttl_T2['touch1'] == "email", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_email_1"]), df_ttl_T2["touch_email_1"])
    df_ttl_T2["touch_referral_1"] = np.where(df_ttl_T2['touch1'] == "referral", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_referral_1"]), df_ttl_T2["touch_referral_1"])
    df_ttl_T2["touch_paid_search_1"] = np.where(df_ttl_T2['touch1'] == "paid_search", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_paid_search_1"]), df_ttl_T2["touch_paid_search_1"])
    df_ttl_T2["touch_direct_1"] = np.where(df_ttl_T2['touch1'] == "direct", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_direct_1"]), df_ttl_T2["touch_direct_1"])
    df_ttl_T2["touch_display_1"] = np.where(df_ttl_T2['touch1'] == "display", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_display_1"]), df_ttl_T2["touch_display_1"])
    df_ttl_T2["touch_social_1"] = np.where(df_ttl_T2['touch1'] == "social", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_social_1"]), df_ttl_T2["touch_social_1"])
    df_ttl_T2["touch_organic_search_1"] = np.where(df_ttl_T2['touch1'] == "organic_search", np.where(df_ttl_T2["count1"] == 1, 1, df_ttl_T2["touch_organic_search_1"]), df_ttl_T2["touch_organic_search_1"])
    #If there are two touches
    df_ttl_T2["touch_email_1"] = np.where(df_ttl_T2['touch1'] == "email", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_email_1"]), df_ttl_T2["touch_email_1"])
    df_ttl_T2["touch_referral_1"] = np.where(df_ttl_T2['touch1'] == "referral", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_referral_1"]), df_ttl_T2["touch_referral_1"])
    df_ttl_T2["touch_paid_search_1"] = np.where(df_ttl_T2['touch1'] == "paid_search", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_paid_search_1"]), df_ttl_T2["touch_paid_search_1"])
    df_ttl_T2["touch_direct_1"] = np.where(df_ttl_T2['touch1'] == "direct", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_direct_1"]), df_ttl_T2["touch_direct_1"])
    df_ttl_T2["touch_display_1"] = np.where(df_ttl_T2['touch1'] == "display", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_display_1"]), df_ttl_T2["touch_display_1"])
    df_ttl_T2["touch_social_1"] = np.where(df_ttl_T2['touch1'] == "social", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_social_1"]), df_ttl_T2["touch_social_1"])
    df_ttl_T2["touch_organic_search_1"] = np.where(df_ttl_T2['touch1'] == "organic_search", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_organic_search_1"]), df_ttl_T2["touch_organic_search_1"])
    df_ttl_T2["touch_email_2"] = np.where(df_ttl_T2['touch2'] == "email", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_email_2"]), df_ttl_T2["touch_email_2"])
    df_ttl_T2["touch_referral_2"] = np.where(df_ttl_T2['touch2'] == "referral", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_referral_2"]), df_ttl_T2["touch_referral_2"])
    df_ttl_T2["touch_paid_search_2"] = np.where(df_ttl_T2['touch2'] == "paid_search", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_paid_search_2"]), df_ttl_T2["touch_paid_search_2"])
    df_ttl_T2["touch_direct_2"] = np.where(df_ttl_T2['touch2'] == "direct", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_direct_2"]), df_ttl_T2["touch_direct_2"])
    df_ttl_T2["touch_display_2"] = np.where(df_ttl_T2['touch2'] == "display", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_display_2"]), df_ttl_T2["touch_display_2"])
    df_ttl_T2["touch_social_2"] = np.where(df_ttl_T2['touch2'] == "social", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_social_2"]), df_ttl_T2["touch_social_2"])
    df_ttl_T2["touch_organic_search_2"] = np.where(df_ttl_T2['touch2'] == "organic_search", np.where(df_ttl_T2["count2"] == 1, 0.5, df_ttl_T2["touch_organic_search_2"]), df_ttl_T2["touch_organic_search_2"])
    #If there are three touches
    df_ttl_T2["touch_email_1"] = np.where(df_ttl_T2['touch1'] == "email", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_email_1"]), df_ttl_T2["touch_email_1"])
    df_ttl_T2["touch_referral_1"] = np.where(df_ttl_T2['touch1'] == "referral", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_referral_1"]), df_ttl_T2["touch_referral_1"])
    df_ttl_T2["touch_paid_search_1"] = np.where(df_ttl_T2['touch1'] == "paid_search", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_paid_search_1"]), df_ttl_T2["touch_paid_search_1"])
    df_ttl_T2["touch_direct_1"] = np.where(df_ttl_T2['touch1'] == "direct", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_direct_1"]), df_ttl_T2["touch_direct_1"])
    df_ttl_T2["touch_display_1"] = np.where(df_ttl_T2['touch1'] == "display", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_display_1"]), df_ttl_T2["touch_display_1"])
    df_ttl_T2["touch_social_1"] = np.where(df_ttl_T2['touch1'] == "social", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_social_1"]), df_ttl_T2["touch_social_1"])
    df_ttl_T2["touch_organic_search_1"] = np.where(df_ttl_T2['touch1'] == "organic_search", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_organic_search_1"]), df_ttl_T2["touch_organic_search_1"])
    df_ttl_T2["touch_email_2"] = np.where(df_ttl_T2['touch2'] == "email", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_email_2"]), df_ttl_T2["touch_email_2"])
    df_ttl_T2["touch_referral_2"] = np.where(df_ttl_T2['touch2'] == "referral", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_referral_2"]), df_ttl_T2["touch_referral_2"])
    df_ttl_T2["touch_paid_search_2"] = np.where(df_ttl_T2['touch2'] == "paid_search", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_paid_search_2"]), df_ttl_T2["touch_paid_search_2"])
    df_ttl_T2["touch_direct_2"] = np.where(df_ttl_T2['touch2'] == "direct", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_direct_2"]), df_ttl_T2["touch_direct_2"])
    df_ttl_T2["touch_display_2"] = np.where(df_ttl_T2['touch2'] == "display", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_display_2"]), df_ttl_T2["touch_display_2"])
    df_ttl_T2["touch_social_2"] = np.where(df_ttl_T2['touch2'] == "social", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_social_2"]), df_ttl_T2["touch_social_2"])
    df_ttl_T2["touch_organic_search_2"] = np.where(df_ttl_T2['touch2'] == "organic_search", np.where(df_ttl_T2["count3"] == 1, 0.2, df_ttl_T2["touch_organic_search_2"]), df_ttl_T2["touch_organic_search_2"])
    df_ttl_T2["touch_email_3"] = np.where(df_ttl_T2['touch3'] == "email", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_email_3"]), df_ttl_T2["touch_email_3"])
    df_ttl_T2["touch_referral_3"] = np.where(df_ttl_T2['touch3'] == "referral", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_referral_3"]), df_ttl_T2["touch_referral_3"])
    df_ttl_T2["touch_paid_search_3"] = np.where(df_ttl_T2['touch3'] == "paid_search", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_paid_search_3"]), df_ttl_T2["touch_paid_search_3"])
    df_ttl_T2["touch_direct_3"] = np.where(df_ttl_T2['touch3'] == "direct", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_direct_3"]), df_ttl_T2["touch_direct_3"])
    df_ttl_T2["touch_display_3"] = np.where(df_ttl_T2['touch3'] == "display", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_display_3"]), df_ttl_T2["touch_display_3"])
    df_ttl_T2["touch_social_3"] = np.where(df_ttl_T2['touch3'] == "social", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_social_3"]), df_ttl_T2["touch_social_3"])
    df_ttl_T2["touch_organic_search_3"] = np.where(df_ttl_T2['touch3'] == "organic_search", np.where(df_ttl_T2["count3"] == 1, 0.4, df_ttl_T2["touch_organic_search_3"]), df_ttl_T2["touch_organic_search_3"])
    #If there are four touches
    df_ttl_T2["touch_email_1"] = np.where(df_ttl_T2['touch1'] == "email", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_email_1"]), df_ttl_T2["touch_email_1"])
    df_ttl_T2["touch_referral_1"] = np.where(df_ttl_T2['touch1'] == "referral", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_referral_1"]), df_ttl_T2["touch_referral_1"])
    df_ttl_T2["touch_paid_search_1"] = np.where(df_ttl_T2['touch1'] == "paid_search", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_paid_search_1"]), df_ttl_T2["touch_paid_search_1"])
    df_ttl_T2["touch_direct_1"] = np.where(df_ttl_T2['touch1'] == "direct", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_direct_1"]), df_ttl_T2["touch_direct_1"])
    df_ttl_T2["touch_display_1"] = np.where(df_ttl_T2['touch1'] == "display", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_display_1"]), df_ttl_T2["touch_display_1"])
    df_ttl_T2["touch_social_1"] = np.where(df_ttl_T2['touch1'] == "social", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_social_1"]), df_ttl_T2["touch_social_1"])
    df_ttl_T2["touch_organic_search_1"] = np.where(df_ttl_T2['touch1'] == "organic_search", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_organic_search_1"]), df_ttl_T2["touch_organic_search_1"])
    df_ttl_T2["touch_email_2"] = np.where(df_ttl_T2['touch2'] == "email", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_email_2"]), df_ttl_T2["touch_email_2"])
    df_ttl_T2["touch_referral_2"] = np.where(df_ttl_T2['touch2'] == "referral", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_referral_2"]), df_ttl_T2["touch_referral_2"])
    df_ttl_T2["touch_paid_search_2"] = np.where(df_ttl_T2['touch2'] == "paid_search", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_paid_search_2"]), df_ttl_T2["touch_paid_search_2"])
    df_ttl_T2["touch_direct_2"] = np.where(df_ttl_T2['touch2'] == "direct", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_direct_2"]), df_ttl_T2["touch_direct_2"])
    df_ttl_T2["touch_display_2"] = np.where(df_ttl_T2['touch2'] == "display", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_display_2"]), df_ttl_T2["touch_display_2"])
    df_ttl_T2["touch_social_2"] = np.where(df_ttl_T2['touch2'] == "social", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_social_2"]), df_ttl_T2["touch_social_2"])
    df_ttl_T2["touch_organic_search_2"] = np.where(df_ttl_T2['touch2'] == "organic_search", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_organic_search_2"]), df_ttl_T2["touch_organic_search_2"])
    df_ttl_T2["touch_email_3"] = np.where(df_ttl_T2['touch3'] == "email", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_email_3"]), df_ttl_T2["touch_email_3"])
    df_ttl_T2["touch_referral_3"] = np.where(df_ttl_T2['touch3'] == "referral", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_referral_3"]), df_ttl_T2["touch_referral_3"])
    df_ttl_T2["touch_paid_search_3"] = np.where(df_ttl_T2['touch3'] == "paid_search", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_paid_search_3"]), df_ttl_T2["touch_paid_search_3"])
    df_ttl_T2["touch_direct_3"] = np.where(df_ttl_T2['touch3'] == "direct", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_direct_3"]), df_ttl_T2["touch_direct_3"])
    df_ttl_T2["touch_display_3"] = np.where(df_ttl_T2['touch3'] == "display", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_display_3"]), df_ttl_T2["touch_display_3"])
    df_ttl_T2["touch_social_3"] = np.where(df_ttl_T2['touch3'] == "social", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_social_3"]), df_ttl_T2["touch_social_3"])
    df_ttl_T2["touch_organic_search_3"] = np.where(df_ttl_T2['touch3'] == "organi_search", np.where(df_ttl_T2["count4"] == 1, 0.1, df_ttl_T2["touch_organic_search_3"]), df_ttl_T2["touch_organic_search_3"])
    df_ttl_T2["touch_email_4"] = np.where(df_ttl_T2['touch4'] == "email", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_email_4"]), df_ttl_T2["touch_email_4"])
    df_ttl_T2["touch_referral_4"] = np.where(df_ttl_T2['touch4'] == "referral", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_referral_4"]), df_ttl_T2["touch_referral_4"])
    df_ttl_T2["touch_paid_search_4"] = np.where(df_ttl_T2['touch4'] == "paid_search", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_paid_search_4"]), df_ttl_T2["touch_paid_search_4"])
    df_ttl_T2["touch_direct_4"] = np.where(df_ttl_T2['touch4'] == "direct", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_direct_4"]), df_ttl_T2["touch_direct_4"])
    df_ttl_T2["touch_display_4"] = np.where(df_ttl_T2['touch4'] == "display", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_display_4"]), df_ttl_T2["touch_display_4"])
    df_ttl_T2["touch_social_4"] = np.where(df_ttl_T2['touch4'] == "social", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_social_4"]), df_ttl_T2["touch_social_4"])
    df_ttl_T2["touch_organic_search_4"] = np.where(df_ttl_T2['touch4'] == "organic_search", np.where(df_ttl_T2["count4"] == 1, 0.4, df_ttl_T2["touch_organic_search_4"]), df_ttl_T2["touch_organic_search_4"])
    #If there are five touches
    df_ttl_T2["touch_email_1"] = np.where(df_ttl_T2['touch1'] == "email", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_email_1"]), df_ttl_T2["touch_email_1"])
    df_ttl_T2["touch_referral_1"] = np.where(df_ttl_T2['touch1'] == "referral", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_referral_1"]), df_ttl_T2["touch_referral_1"])
    df_ttl_T2["touch_paid_search_1"] = np.where(df_ttl_T2['touch1'] == "paid_search", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_paid_search_1"]), df_ttl_T2["touch_paid_search_1"])
    df_ttl_T2["touch_direct_1"] = np.where(df_ttl_T2['touch1'] == "direct", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_direct_1"]), df_ttl_T2["touch_direct_1"])
    df_ttl_T2["touch_display_1"] = np.where(df_ttl_T2['touch1'] == "display", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_display_1"]), df_ttl_T2["touch_display_1"])
    df_ttl_T2["touch_social_1"] = np.where(df_ttl_T2['touch1'] == "social", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_social_1"]), df_ttl_T2["touch_social_1"])
    df_ttl_T2["touch_organic_search_1"] = np.where(df_ttl_T2['touch1'] == "organic_search", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_organic_search_1"]), df_ttl_T2["touch_organic_search_1"])
    df_ttl_T2["touch_email_2"] = np.where(df_ttl_T2['touch2'] == "email", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_email_2"]), df_ttl_T2["touch_email_2"])
    df_ttl_T2["touch_referral_2"] = np.where(df_ttl_T2['touch2'] == "referral", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_referral_2"]), df_ttl_T2["touch_referral_2"])
    df_ttl_T2["touch_paid_search_2"] = np.where(df_ttl_T2['touch2'] == "paid_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_paid_search_2"]), df_ttl_T2["touch_paid_search_2"])
    df_ttl_T2["touch_direct_2"] = np.where(df_ttl_T2['touch2'] == "direct", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_direct_2"]), df_ttl_T2["touch_direct_2"])
    df_ttl_T2["touch_display_2"] = np.where(df_ttl_T2['touch2'] == "display", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_display_2"]), df_ttl_T2["touch_display_2"])
    df_ttl_T2["touch_social_2"] = np.where(df_ttl_T2['touch2'] == "social", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_social_2"]), df_ttl_T2["touch_social_2"])
    df_ttl_T2["touch_organic_search_2"] = np.where(df_ttl_T2['touch2'] == "organic_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_organic_search_2"]), df_ttl_T2["touch_organic_search_2"])
    df_ttl_T2["touch_email_3"] = np.where(df_ttl_T2['touch3'] == "email", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_email_3"]), df_ttl_T2["touch_email_3"])
    df_ttl_T2["touch_referral_3"] = np.where(df_ttl_T2['touch3'] == "referral", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_referral_3"]), df_ttl_T2["touch_referral_3"])
    df_ttl_T2["touch_paid_search_3"] = np.where(df_ttl_T2['touch3'] == "paid_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_paid_search_3"]), df_ttl_T2["touch_paid_search_3"])
    df_ttl_T2["touch_direct_3"] = np.where(df_ttl_T2['touch3'] == "direct", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_direct_3"]), df_ttl_T2["touch_direct_3"])
    df_ttl_T2["touch_display_3"] = np.where(df_ttl_T2['touch3'] == "display", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_display_3"]), df_ttl_T2["touch_display_3"])
    df_ttl_T2["touch_social_3"] = np.where(df_ttl_T2['touch3'] == "social", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_social_3"]), df_ttl_T2["touch_social_3"])
    df_ttl_T2["touch_organic_search_3"] = np.where(df_ttl_T2['touch3'] == "organic_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_organic_search_3"]), df_ttl_T2["touch_organic_search_3"])
    df_ttl_T2["touch_email_4"] = np.where(df_ttl_T2['touch4'] == "email", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_email_4"]), df_ttl_T2["touch_email_4"])
    df_ttl_T2["touch_referral_4"] = np.where(df_ttl_T2['touch4'] == "referral", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_referral_4"]), df_ttl_T2["touch_referral_4"])
    df_ttl_T2["touch_paid_search_4"] = np.where(df_ttl_T2['touch4'] == "paid_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_paid_search_4"]), df_ttl_T2["touch_paid_search_4"])
    df_ttl_T2["touch_direct_4"] = np.where(df_ttl_T2['touch4'] == "direct", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_direct_4"]), df_ttl_T2["touch_direct_4"])
    df_ttl_T2["touch_display_4"] = np.where(df_ttl_T2['touch4'] == "display", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_display_4"]), df_ttl_T2["touch_display_4"])
    df_ttl_T2["touch_social_4"] = np.where(df_ttl_T2['touch4'] == "social", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_social_4"]), df_ttl_T2["touch_social_4"])
    df_ttl_T2["touch_organic_search_4"] = np.where(df_ttl_T2['touch4'] == "organic_search", np.where(df_ttl_T2["count5"] == 1, 0.2/3, df_ttl_T2["touch_organic_search_4"]), df_ttl_T2["touch_organic_search_4"])
    df_ttl_T2["touch_email_5"] = np.where(df_ttl_T2['touch5'] == "email", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_email_5"]), df_ttl_T2["touch_email_5"])
    df_ttl_T2["touch_referral_5"] = np.where(df_ttl_T2['touch5'] == "referral", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_referral_5"]), df_ttl_T2["touch_referral_5"])
    df_ttl_T2["touch_paid_search_5"] = np.where(df_ttl_T2['touch5'] == "paid_search", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_paid_search_5"]), df_ttl_T2["touch_paid_search_5"])
    df_ttl_T2["touch_direct_5"] = np.where(df_ttl_T2['touch5'] == "direct", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_direct_5"]), df_ttl_T2["touch_direct_5"])
    df_ttl_T2["touch_display_5"] = np.where(df_ttl_T2['touch5'] == "display", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_display_5"]), df_ttl_T2["touch_display_5"])
    df_ttl_T2["touch_social_5"] = np.where(df_ttl_T2['touch5'] == "social", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_social_5"]), df_ttl_T2["touch_social_5"])
    df_ttl_T2["touch_organic_search_5"] = np.where(df_ttl_T2['touch5'] == "organic_search", np.where(df_ttl_T2["count5"] == 1, 0.4, df_ttl_T2["touch_organic_search_5"]), df_ttl_T2["touch_organic_search_5"])
    #Calculate position based channel effectiveness
    df_ttl_T2["email_total"] = df_ttl_T2["touch_email_1"] + df_ttl_T2["touch_email_2"] + df_ttl_T2["touch_email_3"] + df_ttl_T2["touch_email_4"] + df_ttl_T2["touch_email_5"]
    df_ttl_T2["referral_total"] = df_ttl_T2["touch_referral_1"] + df_ttl_T2["touch_referral_2"] + df_ttl_T2["touch_referral_3"] + df_ttl_T2["touch_referral_4"] + df_ttl_T2["touch_referral_5"]
    df_ttl_T2["paid_search_total"] = df_ttl_T2["touch_paid_search_1"] + df_ttl_T2["touch_paid_search_2"] + df_ttl_T2["touch_paid_search_3"] + df_ttl_T2["touch_paid_search_4"] + df_ttl_T2["touch_paid_search_5"]
    df_ttl_T2["direct_total"] = df_ttl_T2["touch_direct_1"] + df_ttl_T2["touch_direct_2"] + df_ttl_T2["touch_direct_3"] + df_ttl_T2["touch_direct_4"] + df_ttl_T2["touch_direct_5"]
    df_ttl_T2["display_total"] = df_ttl_T2["touch_display_1"] + df_ttl_T2["touch_display_2"] + df_ttl_T2["touch_display_3"] + df_ttl_T2["touch_display_4"] + df_ttl_T2["touch_display_5"]
    df_ttl_T2["social_total"] = df_ttl_T2["touch_social_1"] + df_ttl_T2["touch_social_2"] + df_ttl_T2["touch_social_3"] + df_ttl_T2["touch_social_4"] + df_ttl_T2["touch_social_5"]
    df_ttl_T2["organic_search_total"] = df_ttl_T2["touch_organic_search_1"] + df_ttl_T2["touch_organic_search_2"] + df_ttl_T2["touch_organic_search_3"] + df_ttl_T2["touch_organic_search_4"] + df_ttl_T2["touch_organic_search_5"]
#     df_ttl_T2["email_total%"] = df_ttl_T2["email_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["referral_total%"] = df_ttl_T2["referral_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["paid_search_total%"] = df_ttl_T2["paid_search_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["direct_total%"] = df_ttl_T2["direct_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["display_total%"] = df_ttl_T2["display_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["social_total%"] = df_ttl_T2["social_total"] / df_ttl_T2["ttotal"]
#     df_ttl_T2["organic_search_total%"] = df_ttl_T2["organic_search_total"] / df_ttl_T2["ttotal"]
del df_ttl_T2["email_total%"]
del df_ttl_T2["referral_total%"]
del df_ttl_T2["paid_search_total%"]
del df_ttl_T2["direct_total%"]
del df_ttl_T2["display_total%"]
del df_ttl_T2["social_total%"]
del df_ttl_T2["organic_search_total%"]
    
df_ttl_T2


# In[17]:


email_total_sum = df_ttl_T2["email_total"].sum()
referral_total_sum = df_ttl_T2["referral_total"].sum()
paid_search_total_sum = df_ttl_T2["paid_search_total"].sum()
direct_total_sum = df_ttl_T2["direct_total"].sum()
display_total_sum = df_ttl_T2["display_total"].sum()
social_total_sum = df_ttl_T2["social_total"].sum()
organic_search_total_sum = df_ttl_T2["organic_search_total"].sum()

total_sums = [["Email total", email_total_sum], ["Referral total", referral_total_sum], ["Paid Search total", paid_search_total_sum], ["Direct total", direct_total_sum], ["Social total", social_total_sum], ["Organic Search total", organic_search_total_sum]]

total_sums.sort(key=lambda x:x[1])
total_sums.reverse()
total_sums


# In[18]:


for i in total_sums:
    print(str(i[0]) + ": " + str(i[1]))


# In[19]:


position_based_total_sums_df = pd.DataFrame.from_records(total_sums)
position_based_total_sums_df


# ### Time to create a table with the actual values separated by tiers (so 3 tiers each, for each allocation system, across all 7 channels)

# In[20]:


#First method: first interaction
new_df_tier1 = new_df[new_df['tier']==1]
new_df_tier2 = new_df[new_df['tier']==2]
new_df_tier3 = new_df[new_df['tier']==3]


# In[21]:


df_t1 = new_df_tier1['touch1']
df_t1.value_counts()


# In[22]:


df_t2 = new_df_tier2['touch1']
df_t2.value_counts()


# In[23]:


df_t3 = new_df_tier3['touch1']
df_t3.value_counts()


# In[24]:


#Referral | display | social | email | paid_search | organic_search | direct
first_t1_list = [1493, 484, 471, 246, 128, 0, 2]
first_t2_list = [2674, 735, 798, 430, 247, 5, 3]
first_t3_list = [3313, 840, 988, 589, 322, 5, 6]


# In[25]:


#Second method: Linear
df_ttl_T_tier1 = df_ttl_T[df_ttl_T['tier']==1]
df_ttl_T_tier2 = df_ttl_T[df_ttl_T['tier']==2]
df_ttl_T_tier3 = df_ttl_T[df_ttl_T['tier']==3]

email_total_sum_tier1 = df_ttl_T_tier1["email_total%"].sum()
referral_total_sum_tier1 = df_ttl_T_tier1["referral_total%"].sum()
paid_search_total_sum_tier1 = df_ttl_T_tier1["paid_search_total%"].sum()
direct_total_sum_tier1 = df_ttl_T_tier1["direct_total%"].sum()
display_total_sum_tier1 = df_ttl_T_tier1["display_total%"].sum()
social_total_sum_tier1 = df_ttl_T_tier1["social_total%"].sum()
organic_search_total_sum_tier1 = df_ttl_T_tier1["organic_search_total%"].sum()

total_sums_tier1 = [["Email total", email_total_sum_tier1], ["Referral total", referral_total_sum_tier1], ["Paid Search total", paid_search_total_sum_tier1], ["Direct total", direct_total_sum_tier1], ["Social total", social_total_sum_tier1], ["Organic Search total", organic_search_total_sum_tier1]]

total_sums_tier1.sort(key=lambda x:x[1])
total_sums_tier1.reverse()
total_sums_tier1


# In[26]:


email_total_sum_tier2 = df_ttl_T_tier2["email_total%"].sum()
referral_total_sum_tier2 = df_ttl_T_tier2["referral_total%"].sum()
paid_search_total_sum_tier2 = df_ttl_T_tier2["paid_search_total%"].sum()
direct_total_sum_tier2 = df_ttl_T_tier2["direct_total%"].sum()
display_total_sum_tier2 = df_ttl_T_tier2["display_total%"].sum()
social_total_sum_tier2 = df_ttl_T_tier2["social_total%"].sum()
organic_search_total_sum_tier2 = df_ttl_T_tier2["organic_search_total%"].sum()

total_sums_tier2 = [["Email total", email_total_sum_tier2], ["Referral total", referral_total_sum_tier2], ["Paid Search total", paid_search_total_sum_tier2], ["Direct total", direct_total_sum_tier2], ["Social total", social_total_sum_tier2], ["Organic Search total", organic_search_total_sum_tier2]]

total_sums_tier2.sort(key=lambda x:x[1])
total_sums_tier2.reverse()
total_sums_tier2


# In[27]:


email_total_sum_tier3 = df_ttl_T_tier3["email_total%"].sum()
referral_total_sum_tier3 = df_ttl_T_tier3["referral_total%"].sum()
paid_search_total_sum_tier3 = df_ttl_T_tier3["paid_search_total%"].sum()
direct_total_sum_tier3 = df_ttl_T_tier3["direct_total%"].sum()
display_total_sum_tier3 = df_ttl_T_tier3["display_total%"].sum()
social_total_sum_tier3 = df_ttl_T_tier3["social_total%"].sum()
organic_search_total_sum_tier3 = df_ttl_T_tier3["organic_search_total%"].sum()

total_sums_tier3 = [["Email total", email_total_sum_tier3], ["Referral total", referral_total_sum_tier3], ["Paid Search total", paid_search_total_sum_tier3], ["Direct total", direct_total_sum_tier3], ["Social total", social_total_sum_tier3], ["Organic Search total", organic_search_total_sum_tier3]]

total_sums_tier3.sort(key=lambda x:x[1])
total_sums_tier3.reverse()
total_sums_tier3


# In[28]:


total_sums_tier1_list = [referral_total_sum_tier1, display_total_sum_tier1, social_total_sum_tier1, email_total_sum_tier1, paid_search_total_sum_tier1, organic_search_total_sum_tier1, direct_total_sum_tier1]
total_sums_tier2_list = [referral_total_sum_tier2, display_total_sum_tier2, social_total_sum_tier2, email_total_sum_tier2, paid_search_total_sum_tier2, organic_search_total_sum_tier2, direct_total_sum_tier2]
total_sums_tier3_list = [referral_total_sum_tier3, display_total_sum_tier3, social_total_sum_tier3, email_total_sum_tier3, paid_search_total_sum_tier3, organic_search_total_sum_tier3, direct_total_sum_tier3]


# In[29]:


linear_tier1_sums_df = pd.DataFrame.from_records(total_sums_tier1)
linear_tier2_sums_df = pd.DataFrame.from_records(total_sums_tier2)
linear_tier3_sums_df = pd.DataFrame.from_records(total_sums_tier3)
linear_tier1_sums_df


# In[30]:


linear_tier2_sums_df


# In[31]:


linear_tier3_sums_df


# In[32]:


#Method 3: Position based
df_ttl_T2_tier1 = df_ttl_T2[df_ttl_T2['tier']==1]
df_ttl_T2_tier2 = df_ttl_T2[df_ttl_T2['tier']==2]
df_ttl_T2_tier3 = df_ttl_T2[df_ttl_T2['tier']==3]

email_total_sum_tier1 = df_ttl_T2_tier1["email_total"].sum()
referral_total_sum_tier1 = df_ttl_T2_tier1["referral_total"].sum()
paid_search_total_sum_tier1 = df_ttl_T2_tier1["paid_search_total"].sum()
direct_total_sum_tier1 = df_ttl_T2_tier1["direct_total"].sum()
display_total_sum_tier1 = df_ttl_T2_tier1["display_total"].sum()
social_total_sum_tier1 = df_ttl_T2_tier1["social_total"].sum()
organic_search_total_sum_tier1 = df_ttl_T2_tier1["organic_search_total"].sum()

total_sums2_tier1 = [["Email total", email_total_sum_tier1], ["Referral total", referral_total_sum_tier1], ["Paid Search total", paid_search_total_sum_tier1], ["Direct total", direct_total_sum_tier1], ["Social total", social_total_sum_tier1], ["Organic Search total", organic_search_total_sum_tier1]]

total_sums2_tier1.sort(key=lambda x:x[1])
total_sums2_tier1.reverse()
total_sums2_tier1


# In[33]:


email_total_sum_tier2 = df_ttl_T2_tier2["email_total"].sum()
referral_total_sum_tier2 = df_ttl_T2_tier2["referral_total"].sum()
paid_search_total_sum_tier2 = df_ttl_T2_tier2["paid_search_total"].sum()
direct_total_sum_tier2 = df_ttl_T2_tier2["direct_total"].sum()
display_total_sum_tier2 = df_ttl_T2_tier2["display_total"].sum()
social_total_sum_tier2 = df_ttl_T2_tier2["social_total"].sum()
organic_search_total_sum_tier2 = df_ttl_T2_tier2["organic_search_total"].sum()

total_sums2_tier2 = [["Email total", email_total_sum_tier2], ["Referral total", referral_total_sum_tier2], ["Paid Search total", paid_search_total_sum_tier2], ["Direct total", direct_total_sum_tier2], ["Social total", social_total_sum_tier2], ["Organic Search total", organic_search_total_sum_tier2]]

total_sums2_tier2.sort(key=lambda x:x[1])
total_sums2_tier2.reverse()
total_sums2_tier2


# In[34]:


email_total_sum_tier3 = df_ttl_T2_tier3["email_total"].sum()
referral_total_sum_tier3 = df_ttl_T2_tier3["referral_total"].sum()
paid_search_total_sum_tier3 = df_ttl_T2_tier3["paid_search_total"].sum()
direct_total_sum_tier3 = df_ttl_T2_tier3["direct_total"].sum()
display_total_sum_tier3 = df_ttl_T2_tier3["display_total"].sum()
social_total_sum_tier3 = df_ttl_T2_tier3["social_total"].sum()
organic_search_total_sum_tier3 = df_ttl_T2_tier3["organic_search_total"].sum()

total_sums2_tier3 = [["Email total", email_total_sum_tier3], ["Referral total", referral_total_sum_tier3], ["Paid Search total", paid_search_total_sum_tier3], ["Direct total", direct_total_sum_tier3], ["Social total", social_total_sum_tier3], ["Organic Search total", organic_search_total_sum_tier3]]

total_sums2_tier3.sort(key=lambda x:x[1])
total_sums2_tier3.reverse()
total_sums2_tier3


# In[35]:


total_sums2_tier1_list = [referral_total_sum_tier1, display_total_sum_tier1, social_total_sum_tier1, email_total_sum_tier1, paid_search_total_sum_tier1, organic_search_total_sum_tier1, direct_total_sum_tier1]
total_sums2_tier2_list = [referral_total_sum_tier2, display_total_sum_tier2, social_total_sum_tier2, email_total_sum_tier2, paid_search_total_sum_tier2, organic_search_total_sum_tier2, direct_total_sum_tier2]
total_sums2_tier3_list = [referral_total_sum_tier3, display_total_sum_tier3, social_total_sum_tier3, email_total_sum_tier3, paid_search_total_sum_tier3, organic_search_total_sum_tier3, direct_total_sum_tier3]


# In[36]:


linear_tier1_sums2_df = pd.DataFrame.from_records(total_sums2_tier1)
linear_tier2_sums2_df = pd.DataFrame.from_records(total_sums2_tier2)
linear_tier3_sums2_df = pd.DataFrame.from_records(total_sums2_tier3)
linear_tier1_sums2_df


# In[37]:


linear_tier2_sums2_df


# In[38]:


linear_tier3_sums2_df


# In[39]:


data = {'Channel': ["Referral total", "Display total", "Social total", "Email total", "Paid Search total", "Organic Search total", "Direct total"]}

final_df = pd.DataFrame(data)


# In[40]:


#'Linear Tier 1', 'Linear Tier 2', 'Linear Tier 3', 'Position based Tier 1', 'Position based Tier 2', 'Position based Tier 3'] 

final_df["First Interaction Tier 1"] = first_t1_list
final_df["First Interaction Tier 2"] = first_t2_list
final_df["First Interaction Tier 3"] = first_t3_list
final_df["Linear Tier 1"] = total_sums_tier1_list
final_df["Linear Tier 2"] = total_sums_tier2_list
final_df["Linear Tier 3"] = total_sums_tier3_list
final_df["Position-based Tier 1"] = total_sums2_tier1_list
final_df["Position-based Tier 2"] = total_sums2_tier2_list
final_df["Position-based Tier 3"] = total_sums2_tier3_list
final_df


# In[42]:


final_df.to_csv("final_df_allocation.csv")


# In[ ]:





# In[41]:


selection = df_ttl_T2.iloc[:,15:35]
print(selection)


# In[44]:


df_ttl_T2["ttotal"].value_counts()


# In[ ]:





# In[ ]:




