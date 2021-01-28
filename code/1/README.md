# Improving Ad Click Prediction by Considering Non-displayed Events 


## Reproduce usage

Go to yh.data/split-data folder. Downloads Yahoo music R3 and extract files to that folder. (Yahoo! Music ratings for User Selected and Randomly Selected songs, version 1.0 (1.2 MB) at https://webscope.sandbox.yahoo.com/catalog.php?datatype=r )
- Install intel mkl https://software.intel.com/en-us/get-started-with-mkl-for-linux
- source source-blas.sh with intel mkl path
```shelll
vim source-blas.sh
# change 'intel_mkl' path variable
```
- Alias python as python2.xx
```shell
alias python=python2.xx
```
- ./exp.sh

## Concept of Each Folder

### data generation

We first need to convert Yahoo Music data.

```shell
cd yh.data
vim README.md
```

### preprocess

After we have converted data, some files should be generated before training.

```shell
cd preprocess
vim README.md
```

### code

After having neccesary files and **converted** data, we can train and conduct performance evaluation by grid.sh & do-test.sh

```shell
cd code
vim README.md
```




