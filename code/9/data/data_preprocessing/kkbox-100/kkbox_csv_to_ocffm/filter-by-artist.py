#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

top_num=10

df = pd.read_csv("train.shuf.csv")
df.target.astype(int)
dfs_info = pd.read_csv("songs.csv")
dfu_info = pd.read_csv("members.csv")


# In[2]:


dfs_info.head()

dfs_info['artist_name'] = dfs_info['artist_name'].apply(lambda x: str(x).strip('|').split('|'))

dfs_info.head()

df_context = pd.merge(df, dfs_info, on='song_id', how='left')

df_context_explode = df_context.explode('artist_name')

artist_counts = df_context_explode['artist_name'].value_counts()

filter_num=101
artist_counts[1:filter_num].plot(kind='bar')
artist_counts[1:filter_num].sum()
myfilter = artist_counts[1:filter_num].index
myfilter


# In[3]:


dfs_info_explode = dfs_info.explode('artist_name')

dfs_info_filter = dfs_info_explode[ dfs_info_explode.artist_name.isin(myfilter) ]

item = dfs_info_filter[ ['genre_ids', 'artist_name' ] ]

item['genre_ids'] = item['genre_ids'].apply( lambda x : str(x).split('|'))

item = item.explode('genre_ids')

genre_gby = item.groupby('artist_name')['genre_ids'].apply(list)

item = genre_gby.reset_index()

def merge(mylist):
    return "|".join(mylist)

item['genre_ids'] = item['genre_ids'].apply(set)
item['genre_ids'] = item['genre_ids'].apply(merge)

item.to_csv("item.csv", index = False)

item.head()


# In[4]:


df_context_filter = df_context_explode[ df_context_explode['artist_name'].isin(myfilter) ]

item_label = item.reset_index()
item_label = item_label.rename(columns={'index': 'label'}).drop(columns=['genre_ids'])

df_context_filter = pd.merge(df_context_filter, item_label, on='artist_name', how='left')

context = df_context_filter.drop(columns=['artist_name', 'composer', 'song_length', 'genre_ids', 'language', 'lyricist', 'target'])
context.head()


# In[5]:


dset = {}
context['his'] = np.nan
context['his'] = context.his.apply(str)
for i, row in context.iterrows():
    user_id = row['msno']
    song_id = str(row['song_id'])
    if user_id in dset:
        context.at[i, 'his'] = '|'.join(dset[user_id][-50:])
        dset[user_id].append(song_id)
    else:
        dset[user_id] = [song_id]
    if i % 100000 == 0:
        print(i)

context.tail()


# In[6]:


context = pd.merge(context, dfu_info, on='msno', how='left')
context.head()
context = context.drop(columns=['registered_via', 'registration_init_time', 'expiration_date', 'bd'])


# In[7]:


context.to_csv("context.csv", index = False)


# In[ ]:




