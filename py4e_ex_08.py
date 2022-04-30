file = 'mbox-short.txt'
hFile = open(file)
for line in hFile:
    line = line.rstrip()
    words = line.split()
    if len(words) > 3 or words:
        if words[0] != 'From':
            continue
        print(words[2])

