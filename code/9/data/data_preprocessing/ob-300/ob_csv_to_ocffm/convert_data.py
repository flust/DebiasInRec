#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import pandas as pd

# Filter by click number
#click_number=1000
click_number=3000

df_click = pd.read_csv("clicks_train.csv")
df_click_pos = df_click.loc[df_click['clicked'] > 0]
df_events = pd.read_csv("cv_events.csv")
df_meta = pd.read_csv("documents_meta.csv")
df_ad = pd.read_csv("promoted_content.csv")

df_ad = pd.merge(df_ad, df_meta, how='left', on='document_id')

df_click_ad = pd.merge(df_click_pos, df_ad, on='ad_id', how='left')

myfilter = df_click_ad['source_id'].value_counts().reset_index()[0:300]['index']


# In[2]:


df_item = df_ad[ df_ad['source_id'].isin(myfilter) ]

df_item = df_item.drop(columns = ['ad_id', 'document_id'])

df_item = df_item.groupby(by=['source_id']).agg(set)

def my_merge( i_set ):
    tmp = [ str(x) for x in i_set]
    tmp = set(tmp)
    tmp = "|".join(tmp)
    return tmp
    
df_item = df_item.applymap(my_merge)

df_item = df_item.reset_index().reset_index()

df_item.to_csv("item.csv".format(click_number), index=False)


# In[3]:


df_context = df_click_ad[ df_click_ad['source_id'].isin(myfilter) ]
df_context = pd.merge(df_context, df_item[['source_id', 'index']], on='source_id', how='left')
df_context = df_context.drop(columns=['document_id', 'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', 'publish_time', 'ad_id'])

df_context = pd.merge(df_context, df_events, on='display_id', how='left')
df_context = pd.merge(df_context, df_meta, on='document_id', how='left')

df_context = df_context.rename(columns={'index': 'label'})

df_context.head()

df_context.to_csv("context.csv".format(click_number), index=False)

