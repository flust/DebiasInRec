# Config
de='det.ffm'
rd='random.ffm'
gr='greedy_random.ffm'
rg='random_greedy.ffm'

python ab_bias.py $rd&
python ab_bias.py $de&
python ab_bias.py $gr&
python ab_bias.py $rg&
wait


