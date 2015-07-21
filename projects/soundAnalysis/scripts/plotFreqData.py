import matplotlib.pyplot as plt

txtfile = "freqData.txt"
maxfreq = 4000;

dataFile = open(txtfile, 'r')

data = dataFile.readlines()
data = [float(singledata[:-1]) for singledata in data]
data = [singledata for singledata in data if singledata < maxfreq]
#copy list
prevdata = data[:]
prevdata.pop(0)
data.pop()
newdata = [nextVal - now  for now, nextVal in zip(data, prevdata)]

plt.plot(data, newdata)
plt.show()
