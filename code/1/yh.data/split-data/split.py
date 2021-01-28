a = open('yh.all.te')

b = open('yh.va', 'w')
c = open('yh.te', 'w')
d = open('yh.bayes', 'w')

from random import random



for i in a:
    dice = random()
    if dice < 0.05:
        d.write(i)
    elif dice < 0.1:
        b.write(i)
    else:
        c.write(i)


    
