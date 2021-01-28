if [ -f "ydata-ymusic-rating-study-v1_0-test.txt" ]
then
    awk '{printf "%s\t1\n", $0}' ydata-ymusic-rating-study-v1_0-test.txt > yh.all.te
    python split.py
else
    echo "The orignal test set ydata-ymusic-rating-study-v1_0-test.txt is contained in R3 - Yahoo! Music ratings for User Selected and Randomly Selected songs, version 1.0 (1.2 MB) is required". 
    echo "It can be obtained from https://webscope.sandbox.yahoo.com/catalog.php?datatype=r"
fi

if [ -f "ydata-ymusic-rating-study-v1_0-train.txt" ]
then
    awk '{printf "%s\t1\n", $0}' ydata-ymusic-rating-study-v1_0-train.txt > yh.tr
    echo "Done!"
else
    echo "The original training set ydata-ymusic-rating-study-v1_0-train.txt is contained in R3 - Yahoo! Music ratings for User Selected and Randomly Selected songs, version 1.0 (1.2 MB) is required". 
    echo "It can be obtained from https://webscope.sandbox.yahoo.com/catalog.php?datatype=r"
fi
