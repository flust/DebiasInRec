# Preprocess

There are several way to exploit the random policy. We will consider four different ways.

## Initialization

- Config **init.sh** file
- Link necessary files
```shell
./init
```

## Calculate inverse propensity score (IPS) (calc-IPS)

```shell
cd calc-IPS
./calc.sh
```

## Calculate imputation constant (calc-r)

- constant from random policy
```shell
cd calc-r
./calc.sh
```
- constant from bias policy
```shell
cd calc-r
./bias.sh
```

## Calculate imputation itemwise constant (calc-item-r)

- constant from random policy
```shell
cd calc-r
./calc.sh
```
- constant from bias policy
```shell
cd calc-r
./bias.sh
```

## Generate impuation model (grid-and-save-imputation-model)

Get best imputation model parameter
```shell
cd save-imputation-model
./grid.sh
```

We can sort the logs files to get the best parameter. Then, please config save-model.sh

```shell
vim save-model.sh # config: k, l and t 
./save-model.sh
```

