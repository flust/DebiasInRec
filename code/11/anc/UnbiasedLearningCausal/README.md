# Debiased Learning for the Causal Effect of Recommender

This is our implementation for the paper ["Unbiased Learning for the Causal Effect of Recommender" (RecSys 2020)](https://doi.org/10.1145/3383313.3412261).

It includes our proposed method *DLCE* and several baselines (*BPR*, *ULBPR*, etc.)
It also covers the codes for generating semi-synthetic datasets for the experiments.

## Requirement
The source codes are mostly written in Python.
Only preprocessing the base data requires R.
The libraries requred are listed below.
The versions in parenthesis are our environment for the experiment.

* Python (3.7.0)
* numpy (1.15.1)
* pandas (0.23.4)
* R (3.4.4)
* data.table (1.12.0)

## Usage
The basic usage of our code is as follows. 
1. Download the dataset from the Dunnhumby dataset site.
1. Preprocess the dataset with *preprocess_dunnhumby.R*.
1. Generate a semi-synthetic dataset with *prepare_data.py*.
1. Run the experiment with *param_search.py*.


### 1. Download the base dataset
We use ***The Complete Journey*** dataset provided by [Dunnhumby](https://www.dunnhumby.com/careers/engineering/sourcefiles).
We really thank Dunnhumby for making this valuable dataset publicly available.
Download the data and locate them under the directory ***./UnbiasedLearningCausal/data/raw***.

### 2. Preprocess the base dataset (step 1 of Subsection 5.1 in the paper)

```bash:sample
cd ./UnbiasedLearningCausal
Rscript preprocess_dunnhumby.R
```
It should yield the files named ***cnt_logs.csv*** in both ***.data/preprocessed/dunn_cat_mailer_10_10_1_1/*** and ***.data/preprocessed/dunn_mailer_10_10_1_1/***.
The former is category-level granularity and the latter is product-level granularity.
The obtained files look like below.
|idx_user|idx_item|num_visit|num_treatment|num_outcome|num_treated_outcome|
|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0|66|11|0|0|
|0|1|66|1|0|0|
|0|2|66|11|0|0|
|0|3|66|11|0|0|
|0|4|66|6|1|0|
|0|5|66|5|5|1|
First two columns represent the pair of user-item.
*num_visit* is the number of visit of the user (***&Sigma; V<sub>ut</sub>***).
*num_treatment* is the number of recommendations (***&Sigma; Z<sub>uit</sub>***).
*num_outcome* is the number of purchases (***&Sigma; Y<sub>uit</sub>***).
*num_treated_outcome* is the number of purchases with recommendation (***&Sigma; Z<sub>uit</sub>Y<sub>uit</sub>***).
These are sufficient information for calculating purchase probabilities with and without recommendations for each user-item pair.

### 3. Generate a semi-synthetic dataset (steps 2-4 of Subsection 5.1 in the paper)

To generate ***Category-Original*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_cat_mailer_10_10_1_1 -tlt 10 -tlv 1 -tle 10 -rp 0.4 -mas original -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40*.

To generate ***Category-Personalized (&beta;=2.0)*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_cat_mailer_10_10_1_1 -tlt 10 -tlv 1 -tle 10 -rp 0.4 -mas rank -sf 2.0 -nr 210 -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210*.

To generate ***Product-Original*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_mailer_10_10_1_1 -tlt 10 -tlv 1 -tle 10 -rp 0.9 -mas original -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_mailer_10_10_1_1/original_rp0.90*.

To generate ***Product-Personalized (&beta;=2.0)*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_mailer_10_10_1_1 -tlt 10 -tlv 1 -tle 10 -rp 0.9 -mas rank -sf 2.0 -nr 991 -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_mailer_10_10_1_1/rank_rp0.90_sf2.00_nr991*.

Regarding some arguments,
- **-rp (rate_prior)** specifies the weight of prior ***w*** in Subsection 5.1.
- **-sf (scale_factor)** specifies ***&beta;***, so you can obtain datasets with varied uneveness of propensities by changing this value.
- **-nr (num_rec)** specifies the average number of recommendations for users and it determins ***&alpha;***.

After the excecution, we obtain three files: *data_train.csv, data_vali_csv, data_test.csv*.
These are data for training, validation, and test as the names suggest.

The *data_train.csv* and *data_test.csv* look like below.
|idx_user|idx_item|treated|outcome|propensity|causal_effect|idx_time|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0|0|0|0.1543|0|0|
|0|1|0|0|0.0154|0|0|
|0|2|0|0|0.1553|0|0|
|0|3|0|0|0.1565|0|0|
|0|4|0|0|0.0937|0|0|
|0|5|0|0|0.0774|0|0|
Note that *data_vali.csv* includes some additional columns for debugging.


### 4. Run the experiment
*param_search.py* conducts experiments with several conditions of hyper-parameters.
The results are saved in **<dataset_folder>/result/<YYYYMMDD_hhmmss_<model_name>_<experiment_name>_tlt10.csv**.
After the parameter tuning, we evaluate on test data by adding argument **-p test**.

See an example below.
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -rp 0.4 -mas rank -sf 2.0 -nr 210 -tm DLMF -cs num_loop:30+interval_eval:20000000+train_metric:AR_logi+dim_factor:200+learn_rate:0.003+naive:False+only_treated:False+capping:0.1:0.01:0.001+reg_factor:0.3:0.03:0.003:0.0003 -ne DLCE_tune_cap_and_reg
```
This runs hyper parameter exploration of DLCE with combinations of capping threshold **&chi; in {0.1, 0.01, 0.001}** and regularization coefficient **&gamma; in {0.3, 0.03, 0.003, 0.0003}** on *Category-Personalized (&beta;=2.0)* dataset.
(This exploration range is simplified for explanation purpose and is different from that described in the paper.)

Here arguments: **-d, -rp, -mas, -sf, -nr**, should be the same with the arguments for corresponding data generation.
Namely, for *Product-Personalized (&beta;=2.0)* dataset, they become 
```
-d dunn_mailer_10_10_1_1 -rp 0.9 -mas rank -sf 2.0 -nr 991
```
**-cs (cond_search)** specifies the hyper-parameter settings.
Parameters are seperated by **+** and explored values are seperated by **:**.

Some notes on parameters,
- **interval_eval** is the number of training iterations between each evaluation and **num_loop** is the number of evaluations. So multiplying them yields total number of training iterations.
- **train_metric** is the training objective and it determins the method together with **-tm** argument. For example, **-DLMF** and **train_metric:AR_logi** is *DLCE* with upper bound loss in Eq. (29), **-DLMF** and **train_metric:AR_sig** is *DLCE* with approximation loss in Eq. (28), **-ULMF** and **train_metric:AUC** is *ULBPR*, **-ULMF** and **train_metric:logloss** is ULRMF, **-LMF** and **train_metric:AUC** is *BPR*.
- **naive:True** select *BLCE* that uses naive estimate in Eq. (12). **naive:False** for *DLCE* and *DLTO*.
- **only_treated:True** select *DLTO* that learns for treated outcomes. **only_treated:False** for *DLCE* and *BLCE*.
- **-ne** describes the name of experiment that is included in the result file name. 

The result file includes several metrics and hyperparameters.
**CPrec_10** is **CP@10**, **CPrec_100** is **CP@100**, and **CDCG_100000** is **CDCG**.
Note that the numbers after *_* denote the number of recommendation for evaluation (top-N in the ranking.)
For CDCG, we evaluate the whole ranking, hence the number is intentinally set to be larger than the number of items.

After tuning the parameter, do the test evaluation as following.
```bash:sample
python param_search.py -p test -ce False -d dunn_cat_mailer_10_10_1_1 -rp 0.4 -mas rank -sf 2.0 -nr 210 -tm DLMF -cs num_loop:1+interval_eval:120000000+train_metric:AR_logi+dim_factor:200+learn_rate:0.003+naive:False+only_treated:False+capping:0.3+reg_factor:0.003 -ne DLCE_test
```
If we set **-ce True**, it also calculates IPS-based estimates of evaluation metrics that were used for the analysis in Subsection 5.4.


Other examples.

Run *ULBPR* on ***Category-Original*** dataset:
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -rp 0.4 -mas original -tm ULMF -cs num_loop:30+interval_eval:20000000+train_metric:AUC+dim_factor:200+learn_rate:0.01+alpha:0.0:0.2:0.4+reg_factor:0.3:0.03:0.003:0.0003 -ne ULBPR_tune_alpha_and_reg
```
Note that ULBPR has an unique hyper paramter &alpha; (different from &alpha; in data generation), that is the probability of regarding NR-NP (not recommended and not purchased) items as positives. Please refer to ["Uplift-based Evaluation and Optimization of Recommenders" (RecSys 2019)](https://dl.acm.org/doi/10.1145/3298689.3347018) for the detail.

Run *CausE* (CausE-Prod version) on ***Category-Original*** dataset:
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -rp 0.4 -mas original -tm CausEProd -cs num_loop:30+interval_eval:20000000+train_metric:logloss+dim_factor:200+learn_rate:0.01+reg_causal:0.1:0.01:0.001+reg_factor:0.3:0.03:0.003:0.0003 -ne ULBPR_tune_alpha_and_reg
```
Note that CausE has an unique hyper paramter, regularization between tasks (*reg_causal* in our code), that penalizes the difference in item latent factors between treatment and control conditions. Please refer to ["Causal Embedding for Recommendation" (RecSys 2018)](https://dl.acm.org/doi/10.1145/3240323.3240360) for the detail.
We reimplemented CausE to rank items by the causal effect since the original CausE aimed for the prediction of treated outcomes.

Run *BPR* on ***Product-Personalized (&beta;=2.0)*** dataset:
```bash:sample
python param_search.py -d dunn_mailer_10_10_1_1 -rp 0.9 -mas rank -sf 2.0 -nr 991 -tm LMF -cs num_loop:30+interval_eval:20000000+train_metric:AUC+dim_factor:200+learn_rate:0.01+reg_factor:0.3:0.03:0.003:0.0003 -ne BPR_tune_reg
```

Run *Pop* on ***Product-Original*** dataset:
```bash:sample
python param_search.py -d dunn_mailer_10_10_1_1 -rp 0.9 -mas original -tm PopularBase -cs num_loop:1+interval_eval:1 -ne Pop
```
Note that *Pop* does not require model training, but our implementation treates calculation of popularity rankings as a sort of training, hence the code requires num_loop and interval_eval.

## Tuned Hyper-parameters to reproduce the paper.
In case the best parameters are different among evaluation metrics, we chose the best one for each metric.
In practice, the most important metric would differ depending on the actual system and the model should be tuned for the most important metric.
For example, if the number of recommendation is fixed to 10, CP@10 would be appropriate; if the users can view recomendated items as much as they want, CDCG would be appropriate.

Below are tuned hypearparameters for ***Category-Original*** dataset.
|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.01|100|0.03|-|-|-|
|BPR|CP@100|0.01|80|0.001|-|-|-|
|BPR|CDCG|0.01|80|0.01|-|-|-|
|ULBPR|CP@10|0.01|300|0.0001|-|0.2|-|
|ULBPR|CP@100|0.01|300|0.003|-|0.0|-|
|ULBPR|CDCG|0.01|300|0.0003|-|0.0|-|
|ULRMF|CP@10|0.01|300|0.003|-|0.0|-|
|ULRMF|CP@100|0.01|300|0.003|-|0.0|-|
|ULRMF|CDCG|0.01|300|0.003|-|0.0|-|
|CausE|CP@10|0.01|1100|0.001|-|-|0.0001|
|CausE|CP@100|0.01|1000|0.001|-|-|0.001|
|CausE|CDCG|0.01|1000|0.001|-|-|0.001|
|BLCE|CP@10|0.003|400|0.01|-|-|-|
|BLCE|CP@100|0.003|300|0.01|-|-|-|
|BLCE|CDCG|0.003|300|0.01|-|-|-|
|DLTO|CP@10|0.003|300|0.03|0.01|-|-|
|DLTO|CP@100|0.003|300|0.01|0.01|-|-|
|DLTO|CDCG|0.003|300|0.03|0.01|-|-|
|DLCE|CP@10|0.003|640|0.01|0.01|-|-|
|DLCE|CP@100|0.003|460|0.01|0.01|-|-|
|DLCE|CDCG|0.003|540|0.01|0.01|-|-|

python param_search.py -p test -ce False -d dunn_cat_mailer_10_10_1_1 -rp 0.4 -mas rank -sf 2.0 -nr 210 -tm DLMF -cs num_loop:1+interval_eval:120000000+train_metric:AR_logi+dim_factor:200+learn_rate:0.003+naive:False+only_treated:False+capping:0.3+reg_factor:0.003 -ne DLCE_test

Below are tuned hyparparameters for ***Category-Personalized (&beta;=2.0)*** dataset.
|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.01|40|0.01|-|-|-|
|BPR|CP@100|0.01|40|0.01|-|-|-|
|BPR|CDCG|0.01|40|0.01|-|-|-|
|ULBPR|CP@10|0.03|400|0.003|-|0.0|-|
|ULBPR|CP@100|0.03|400|0.003|-|0.0|-|
|ULBPR|CDCG|0.03|400|0.003|-|0.0|-|
|ULRMF|CP@10|0.01|280|0.003|-|0.0|-|
|ULRMF|CP@100|0.01|530|0.01|-|0.0|-|
|ULRMF|CDCG|0.01|530|0.01|-|0.0|-|
|CausE|CP@10|0.01|1500|0.001|-|-|0.001|
|CausE|CP@100|0.01|1500|0.001|-|-|0.01|
|CausE|CDCG|0.01|1500|0.001|-|-|0.001|
|BLCE|CP@10|0.003|200|0.03|-|-|-|
|BLCE|CP@100|0.003|200|0.03|-|-|-|
|BLCE|CDCG|0.003|200|0.03|-|-|-|
|DLTO|CP@10|0.003|20|0.01|0.3|-|-|
|DLTO|CP@100|0.003|10|0.01|0.3|-|-|
|DLTO|CDCG|0.003|20|0.01|0.3|-|-|
|DLCE|CP@10|0.003|120|0.003|0.3|-|-|
|DLCE|CP@100|0.003|90|0.01|0.3|-|-|
|DLCE|CDCG|0.003|120|0.003|0.3|-|-|

Below are tuned hypearparameters for ***Product-Original*** dataset.
|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.01|30|0.003|-|-|-|
|BPR|CP@100|0.01|50|0.01|-|-|-|
|BPR|CDCG|0.01|30|0.003|-|-|-|
|ULBPR|CP@10|0.01|240|0.01|-|0.0|-|
|ULBPR|CP@100|0.01|240|0.01|-|0.0|-|
|ULBPR|CDCG|0.01|260|0.03|-|0.0|-|
|ULRMF|CP@10|0.03|400|0.01|-|0.2|-|
|ULRMF|CP@100|0.03|650|0.01|-|0.0|-|
|ULRMF|CDCG|0.03|650|0.01|-|0.0|-|
|CausE|CP@10|0.01|440|0.03|-|-|0.01|
|CausE|CP@100|0.01|500|0.01|-|-|0.01|
|CausE|CDCG|0.01|500|0.01|-|-|0.01|
|BLCE|CP@10|0.003|400|0.03|-|-|-|
|BLCE|CP@100|0.003|400|0.03|-|-|-|
|BLCE|CDCG|0.003|400|0.03|-|-|-|
|DLTO|CP@10|0.003|80|0.1|0.01|-|-|
|DLTO|CP@100|0.003|150|0.3|0.01|-|-|
|DLTO|CDCG|0.003|150|0.3|0.01|-|-|
|DLCE|CP@10|0.003|280|0.03|0.003|-|-|
|DLCE|CP@100|0.003|600|0.1|0.003|-|-|
|DLCE|CDCG|0.003|600|0.1|0.003|-|-|


Below are tuned hypearparameters for ***Product-Personalized (&beta;=2.0)*** dataset.
|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.01|300|0.01|-|-|-|
|BPR|CP@100|0.01|300|0.01|-|-|-|
|BPR|CDCG|0.01|300|0.01|-|-|-|
|ULBPR|CP@10|0.03|200|0.003|-|0.0|-|
|ULBPR|CP@100|0.03|200|0.003|-|0.0|-|
|ULBPR|CDCG|0.03|200|0.03|-|0.0|-|
|ULRMF|CP@10|0.03|360|0.003|-|0.0|-|
|ULRMF|CP@100|0.03|340|0.01|-|0.0|-|
|ULRMF|CDCG|0.03|340|0.01|-|0.0|-|
|CausE|CP@10|0.01|800|0.003|-|-|0.1|
|CausE|CP@100|0.01|1400|0.001|-|-|0.01|
|CausE|CDCG|0.01|1400|0.001|-|-|0.01|
|BLCE|CP@10|0.001|120|0.03|-|-|-|
|BLCE|CP@100|0.001|40|0.03|-|-|-|
|BLCE|CDCG|0.001|40|0.03|-|-|-|
|DLTO|CP@10|0.001|220|0.01|0.3|-|-|
|DLTO|CP@100|0.001|80|0.01|0.3|-|-|
|DLTO|CDCG|0.001|80|0.01|0.3|-|-|
|DLCE|CP@10|0.001|400|0.01|0.3|-|-|
|DLCE|CP@100|0.001|200|0.01|0.3|-|-|
|DLCE|CDCG|0.001|200|0.01|0.3|-|-|