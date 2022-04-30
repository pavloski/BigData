hFile = open('intro.txt')
dDictionary = dict()
for line in hFile:
    line = line.rstrip()
    words = line.split()
    for word in words:
        dDictionary[word] = dDictionary.get(word, 0) + 1


print(sorted(((v, k) for k, v in dDictionary.items()), reverse = True)[:3])
for v,k in sorted(((v, k) for k, v in dDictionary.items()), reverse = True)[:3]:
    print (k,v)