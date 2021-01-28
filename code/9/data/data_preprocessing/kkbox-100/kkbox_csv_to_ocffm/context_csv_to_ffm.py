#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! /usr/bin/python3
import csv

# Field 1, 2
field1 = ['msno', 'city', 'gender', 'source_system_tab', 'source_screen_name', 'source_type', 'his']
all_fields = [field1]

feat_dict = {}
all_idx = [-1]

def add_feat(key, value, field):
    global all_idx
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    all_idx[field] += 1
    feat_dict[real_key] = all_idx[field]
    return all_idx[field]

def make_tuple(feat_list,field):
    feat_str = ["{}:{}".format(idx, val) for idx, val in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))


# In[2]:


of = open('context.ffm', 'w')
rf = open('context.csv')
for line in csv.DictReader(rf, delimiter=','):
    # Lable
    #output = line['level_0'].strip().replace("|", ",")
    output = ",".join( ["{}:1:1".format(adid) for adid in line['label'].strip().strip("|").split("|")] )
    #Feature
    for i, field_i in enumerate(all_fields):
        feat_idx_list = []
        for key in field_i:
            if line[key] == "":
                continue
            values = line[key].strip('|').split("|")
            for val in values:
                feat_idx_list.append( ( add_feat(key, val.strip(), i), 1.0/float(len(values)) ) )
        output = "{} {}".format(output," ".join(make_tuple(feat_idx_list, i)))

    print( output, file=of )

print(all_idx)


# In[3]:


of.close()


# In[ ]:




