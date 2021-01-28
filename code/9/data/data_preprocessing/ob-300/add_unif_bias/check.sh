
bias_ratio(){
    echo $1 $2
    numa=`awk '{print $1} ' $1 | grep ':1:' |  wc -l& `
    numb=`awk '{print $1} ' $2 | grep ':1:' |  wc -l& ` 
    wait 
    echo " $numa / $numb " | bc -l
}

bias_ratio det.ffm det.ffm.pos.0.5.unif.bias
bias_ratio random.ffm random.ffm.pos.0.5.unif.bias
