import urllib.request, urllib.parse, urllib.error

fhand = urllib.request.urlopen('http://www.dr-chuck.com/page2.htm')

i=0
for line in fhand:
    print(line.decode().strip())
