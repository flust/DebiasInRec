#!/bin/bash

list_solve_type(){
echo "=== Choose solve type ===
0: ob, ffm-ffm      |
------------------"
echo -n "sovle (0): "
read solve_type
}

set_ns(){
echo -n "Do --ns ? [y/n]: "
read ns_bool
if [[ $ns_bool =~ y ]]
then
  ns='--ns'
else
  ns=''
fi
}

te_or_va(){
echo -n "va or te ? [0/1]: "
read ns_bool
if [[ $ns_bool =~ 0 ]]
then
  tr_data='tr'
  te_data='va'
else
  tr_data='trva'
  te_data='te'
fi
}

set_up_solve_type(){
  echo -n "latent vector size(k): "
  read k
  
  te_or_va

  case $solve_type in
    0)
      # Ext & logs_pth
      tr_ext=ocffm
      i_ext=ocffm
      # Data
      name=ob
      tr=ob.${tr_data}.${tr_ext}
      te=ob.${te_data}.sub.${tr_ext}
      item=item.${i_ext}
      # Var
      set_ns
      t=51
      ;;
    *)
      echo "No match"
      exit
  esac
  ext=${tr_ext}-${i_ext}${ns}

  if [[ ${te_data} =~ va ]]
  then
    logs_pth=logs/${name}.${k}/${ext}
  else
    logs_pth=logs/${name}.${k}.${te_data}/${ext}
  fi
}


choose_w_list(){
  # 2^0 ~ 2^-11
  w_all=(1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 6.103515625e-05 3.0517578125e-05 1.52587890625e-05)
  w_train=()

  # Print w_all
  for i in ${!w_all[@]};
  do
    printf "%s: %s(2^-%s)\n" "$i" "${w_all[$i]}" "$i"
  done

  # Create w_train
  while true;
  do
    echo -n "select idx: "
    read idx
    if [ $idx -eq -1 ]
    then
      break
    fi
    w_train+=(${w_all[${idx}]})
  done
}

choose_wn_list(){
  # 2^0 ~ 2^-11
  wn_all=(1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 6.103515625e-05 3.0517578125e-05 1.52587890625e-05)
  wn_train=()

  # Print w_all
  for i in ${!wn_all[@]};
  do
    printf "%s: %s(2^-%s)\n" "$i" "${wn_all[$i]}" "$i"
  done

  # Create wn_train
  while true;
  do
    echo -n "select idx: "
    read idx
    if [ $idx -eq -1 ]
    then
      break
    fi
    wn_train+=(${wn_all[${idx}]})
  done
}

choose_l_list(){
  # 2^1 ~ 2^4
  l_all=(1 4 16)
  l_train=()

  # Print l_all
  for i in ${!l_all[@]};
  do
    printf "%s: %s\n" "$i" "${l_all[$i]}" 
  done

  # Create l_train
  while true;
  do
    echo -n "select idx: "
    read idx
    if [ $idx -eq -1 ]
    then
      break
    fi
    l_train+=(${l_all[${idx}]})
  done
}

choose_r_list(){
  # 2^1 ~ 2^4
  r_all=(-1 -5 -10 -15)
  r_train=()

  # Print l_all
  for i in ${!r_all[@]};
  do
    printf "%s: %s\n" "$i" "${r_all[$i]}" 
  done

  # Create l_train
  while true;
  do
    echo -n "select idx: "
    read idx
    if [ $idx -eq -1 ]
    then
      break
    fi
    r_train+=(${r_all[${idx}]})
  done
}

task(){
  for w in ${w_train[@]} 
  do
      for wn in ${wn_train[@]} 
      do
          for r in ${r_train[@]}
          do
            for l in ${l_train[@]}
            do
              echo "./train -k $k -l $l -t ${t} -r $r -w $w -wn $wn $ns -c ${c} -p ${te} ${item} ${tr} > $logs_pth/${tr}.$l.$r.$w.$wn.${ext}"
            done
          done
      done
  done
}

list_param(){
echo "===Parameter list==="
echo "data: $name
tr: $tr
va: $te
item: $item
k: $k
t: $t
ns: $ns
w list: [ ${w_train[@]} ]
wn list: [ ${wn_train[@]} ]
l list: [ ${l_train[@]} ]
r list: [ ${r_train[@]} ]
log_path: ${logs_pth}"
}

grid(){
# Empty .task_file.tmp
echo -n "" > .task_file.tmp

while true
do
  # Choose solve type
  clear
  list_solve_type
  set_up_solve_type
  
  # Set w range
  clear
  echo "===Set w range, -1 to exit==="
  choose_w_list
  echo "++++++++++++++++++++++++++"
  
  # Set wn range
  clear
  echo "===Set wn range, -1 to exit==="
  choose_wn_list
  echo "++++++++++++++++++++++++++"
  
  # Set r range
  clear
  echo "===Set r range, -1 to exit==="
  choose_r_list
  echo "++++++++++++++++++++++++++"
  
  # Set l range
  clear
  echo "===Set l range, -1 to exit==="
  choose_l_list
  echo "++++++++++++++++++++++++++"

  # Set -c option
  clear
  echo "===Set num core -c option==="
  echo -n "-c = "
  read c
  
  # logs_pth
  
  # Check the right command
  clear
  list_param
  echo "===All cmd to run==="
  task
  echo -n "Save current setting to .task_file.tmp? [y/n] "
  read std
  if [[ $std =~ y ]]
  then
    task >> .task_file.tmp
    # Create logs_pth
    mkdir -p $logs_pth
  fi
  

  # Start or not
  echo -n "Continue? [y/n] "
  read std
  if [[ $std =~ n ]]
  then
    break
  fi
done

# Check all command
clear
echo "===All run settings==="
cat .task_file.tmp
echo "====================="

# Number of parameter set do in once.
echo -n "Number of param run at once: "
read num_core
echo "++++++++++++++++++++++++++"

echo -n "Start ? [y/n] "
read std
if [[ $std =~ y ]]
then
  echo "run"
  cat .task_file.tmp | xargs -d '\n' -P $num_core -I {} sh -c {} &
else
  echo "no run"
fi

}

grid
