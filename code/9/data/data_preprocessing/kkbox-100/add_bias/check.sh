#! /bin/bash

check(){
    echo $1
    for i in {1..10}
    do
        awk '{print $1}' $1 | awk -F , -v pos=$i '{print $pos}' | grep ':1:' | wc -l
    done

}


check $1 
