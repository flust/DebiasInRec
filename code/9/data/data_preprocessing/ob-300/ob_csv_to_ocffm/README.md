# Data preprocess 

Overall usage

1. Link clicks_train.csv, cv_events, promote_content.csv document_meta.csv
2. ./generate_ffm.sh

===========================

1. filter.py

* Input: clicks_train.csv, cv_events.csv, promote_content.csv
* Output: click_filter.csv, events_filter.csv, ad_filter.csv
* Purpose: Filter by number of clicks count

2. add_label.py

* Input: ad_filter.csv, events_filter.csv, click_filter.csv, document_meta.csv
* Output: events_filter_label.csv, ad_filter.csv

4. Convert events_filter_label.csv to data format

* context_ffm.py, context_fm.py, context_mf.py
* Input: events_filter_label.csv 
* Output: ob_all.ffm

5. Convert ad_filter.csv to data format

* item_ffm.py, item_fm.py, item_mf.py
* Input: ad_filter.csv 
* Output: item.{ffm,fm,mf}
