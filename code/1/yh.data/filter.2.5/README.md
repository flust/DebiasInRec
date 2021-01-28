# Filter yh.tr to yh.f.tr

## Modify yh.tr
Input: yh.tr 
Output: yh.all.tr.ps
```shell
ln -s ../split-data/yh.tr ./
awk '{printf "%s\t1\n", $0}' yh.tr > yh.all.tr.ps
```

## split.py file
Input: yh.all.tr.ps
Output: yh.all.tr.{0..4}.ps.tr yh.all.tr.{0..4}.ps.te
```shell
python split.py yh.all.tr.ps
```

## grid.py
Input: *.tr *.te
Output: *.logs
```shell
cd grid-code
make
cd ..
ln -s grid-code/train ./ 
./grid.sh
```

## merge.py
Input: *.logs
Output: *.merges 
```shell
./merge_logs.sh
head *.best_score
```
Example:
> ==> 0.0001.best_score <==
> 200	1.6058
>
> ==> 0.001.best_score <==
> 200	1.3182
>
> ==> 0.01.best_score <==
> 62	1.2482
>
> ==> 0.1.best_score <==
> 34	1.2082
>
> ==> 1.best_score <==
> 1	1.5184

Select best paramter with lowest mse score. In the demo case, we have lowest mse  1.2082 with l = 0.1 and iteration 34.

## Filter yh.all.tr.ps
Config filter.sh with previous selected parameters. Ex: l = 0.1 and  t = 34
```shell
cd ocmf
vim filter.sh # Config best parameters
make
./filter.sh
cp yh.f.tr ../../split-data/
```
