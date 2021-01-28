# Run experience

## Initialization

Link some preprocess files and data to folders.
```shell
./init.sh
```

## General usage

- Go to the corresponding folder. For example,
```shell
cd new-complex
```

- Grid parameter
```shell
./grid.sh
```

- Find best parameter
``` shell
cat logs/* | sort -grk 28 
```

- Do perfoamance tesing
```shell
vim do-test.sh # config corresponding parameter
./do-test.sh
```

