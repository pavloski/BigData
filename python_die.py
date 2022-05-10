import random
mydict = dict()
for i in range (1,100):
    a = random.randint(1,6)
    mydict[a] = mydict.get(a,0) +1

print(sorted(mydict))