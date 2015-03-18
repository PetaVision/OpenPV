import urllib
import re
response1 = urllib.urlopen("https://svn.code.sf.net/p/petavision/code/trunk/src/layers/")
response2 = urllib.urlopen("https://svn.code.sf.net/p/petavision/code/trunk/src/connections/")
llines = response1.readlines()
clines = response2.readlines()
layers = []
conns = []

for i in llines:
    j = re.search("(?<=href).+", i)
    if j:
        k = re.search(".+(?=cpp)", j.group())
        if k:
            l = re.search("\w+", k.group())
            if l:
                layers.append(l.group())
for i in clines:
    j = re.search("(?<=href).+", i)
    if j:
        k = re.search(".+(?=cpp)", j.group())
        if k:
            l = re.search("\w+", k.group())
            if l:
                conns.append(l.group())



print(layers)
print(conns)

f = open("layers.txt", "w")
for i in layers:
    f.write(i + "\n")
f.close()  
f = open("connections.txt", "w")
for i in conns:
    f.write(i + "\n")
f.close()
