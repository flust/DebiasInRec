# pos-bias-exp-code

## System requirement

- memory 128G
- GPU

## Install dependencies

- [mkl](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html), modify 'init.sh' according to your installation path.
- cuda 9.0 or higher
- anaconda python 3.7 version

## Create conda env for dl exps

```shell
conda env create -f pos.yml
conda activate py3.7
git clone --recursive https://github.com/johncreed/pos-bias-exp-code.git
```

## Experiments

### Prepare raw data (should be do first!)

Download data from [Outbrain](https://www.kaggle.com/c/outbrain-click-prediction) and [KKbox](https://www.kaggle.com/c/kkbox-music-recommendation-challenge).
- **KKbox**: Put KKBOX's members.csv, songs.csv and train.csv to "data/data_preprocessing/kkbox-100/kkbox_csv_to_ocffm/".
- **Outbrain**: Put Outbrain's clicks_train.csv, cv_events.csv, promote_content.csv and document_meta.csv to "data/data_preprocessing/ob.300/ob_csv_to_ocffm".

### Toy example

Before running the all-in-one script, it's recommended to run the toy example first to check if the env and denpendencies are set up properly.
```shell
./toy-example.sh
```
In the end, it should be output something as below:
```
method,ll-pos,auc-pos,ll-alpha,auc-alpha
FFM-Ideal,0.662159,0.473809,0.662134,0.363423
FFM-Greedy-A,0.683852,0.389617,0.683836,0.674310
FFM-Greedy-P,0.683749,0.390451,0.683732,0.729846
FFM-Random,0.663728,0.386832,0.663704,0.216443
FFM-Greedy,0.683511,0.338186,0.683490,0.642328
FFM-EE(0.01),0.679597,0.387599,0.679578,0.750850
FFM-EE(0.1),0.673465,0.408462,0.673445,0.688388
FFM-CF(0.01),0.692,0.628,0.692,0.698
FFM-CF(0.01),0.18,0.432,0.179,0.197
====================
TFM-Greedy,0.564,0.679,--,--
TFM-EE(0.01),0.558,0.616,--,--
TFM-EE(0.1),0.509,0.536,--,--
TFM-Random,0.164,0.501,--,--
TFM-CF(0.01),0.692,0.541,--,--
TFM-CF(0.1),0.207,0.506,--,--
TFM-RG,0.193,0.316,--,--
TFM-GR,0.56,0.41,--,--
```

And in directory "toy_exps_res/", there are files like:
- toy-example_mbias.pdf
- toy-example_pbias.pdf
- toy-example_rbias.pdf
- toy-example.res

Otherwise, there is something wrong with your env and denpendencies, please check the screen output.

### All-in-one 

After dependencies installation, setting up the env and data preparation, please run
```shell
./exp.sh
```
All results are in directory "exps_res/".
