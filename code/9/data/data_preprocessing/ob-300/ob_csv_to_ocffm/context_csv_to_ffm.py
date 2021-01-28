#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! /usr/bin/python3
from __future__ import print_function
import csv

# Field 1, 2
keys1 = ['platform', 'geo_location', 'source_id', 'publisher_id', 'document_id']

feat_dict = {}
incr_num1 = -1

def add_feat(key, value, field):
    global incr_num1
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    if field == 0:
        incr_num1 += 1
        feat_idx = incr_num1
    feat_dict[real_key] = feat_idx
    return feat_idx

def make_tuple(feat_list,field):
    feat_str = ["%d:1" % i for i in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))


def handle_geo(geo_str):
    items = geo_str.split('>')
    if len(items) == 1:
        if items[0].isdigit():
            return [add_feat('code', items[0], 0)]
        else:
            return [add_feat('country', items[0], 0)]
    if len(items) == 2:
        if items[-1].isdigit():
            return [add_feat('country', items[0], 0), add_feat('code', items[1], 0)]
        else:
            return [add_feat('country', items[0], 0), add_feat('state', items[1], 0)]
    if len(items) == 3:
        return [add_feat('country', items[0], 0), add_feat('state', items[1], 0), add_feat('code', items[2], 0)]


def convert2ffm( o_f, i_f ):
    svm_f = open(o_f, 'w')
    for line in csv.DictReader(open(i_f), delimiter=','):
        # Key1
        feat_idx_list = []
        output = "{}:1:1".format(line['label'])
        for key in keys1:
            if line[key] == "":
                continue
            if key == 'geo_location':
                feat_idx_list += handle_geo(line['geo_location'])
            else:
                feat_idx_list.append(add_feat(key, line[key], 0))
        output = "{} {}".format(output, " ".join(make_tuple(feat_idx_list,0)))
        print(output,file=svm_f)




# In[2]:


convert2ffm('context.ffm', 'context.csv')

