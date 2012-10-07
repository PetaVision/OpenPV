filename = "/Users/slundquist/Desktop/LCALIF_31_31_0.txt"
progress_step = 1000

f = open(filename, 'r')
lines = f.readlines()
f.close()

for line in lines:
   print line
