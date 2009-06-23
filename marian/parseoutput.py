import numpy

#canon = '776400013652413777'
max_weight = 1.0

# read mom string
f = open('log.dat','r')
for line in f.readlines():
   ls = line.split()
   if ls[0] == 'mom:':
	bin_string = ls[1]
	print "mom string: %s" % bin_string

# build canon string

canon = ''
for j in range(len(bin_string)):
	if j==0:
	   #n = int(bin_string[1]) + int(bin_string[0])*2
	   n=int(bin_string[0:2],2)
	   canon = canon+str(n)
	   #print "n = %d canon = %s" % (n,canon)
	   #raw_input()
	   continue
        if j== len(bin_string):
	   #n = int(bin_string[-1]) + int(bin_string[-2])*2
	   n=int(bin_string[-2:-1],2)
	   canon = canon + str(n)
	   #print "n = %d canon = %s" % (n,canon)
	   #raw_input()
           continue
	#n = int(bin_string[j+1]) + int(bin_string[j])*2 + int(bin_string[j])*4
	n=int(bin_string[j-1:j+2],2)
	canon = canon + str(n)
	#print "n = %d canon = %s" % (n,canon)
	#raw_input()
		
print "mom   string: %s" % bin_string
print "canon string: %s" % canon
raw_input('type a character to continue')


# open matlab file
m_dir  = '/nh/home/manghel/petavision/MLAB_Anghel/'
m_file = 'weights_evol_features.m'
m_out  =  open(m_dir + 'wef.m','w')
m_out.write('fname = {')



for j in range(len(canon)):
    
    f = open('log.dat','r')
    #fout = open('3d_plot_' + str(j) + '_all.txt','w')
    if canon[j]=='0':	
      fname = str(j)+'_'+canon[j]+'_000.txt'
      fout = open(fname,'w')
      m_out.write("'" + str(j)+'_'+canon[j]+'_000.txt' + "', ")	
    else:
      fname = str(j)+'_'+canon[j]+'_'+numpy.binary_repr(int(canon[j]),3)+'.txt'
      fout = open(fname,'w')
      m_out.write("'" + str(j)+'_'+canon[j]+'_'+numpy.binary_repr(int(canon[j]),3)+'.txt' + "', ")

    print "neuron %d write to %s" % (j,fname)

    #foutx = open('3d_plot_' + str(j) + '_x.txt','w')
    #fouty = open('3d_plot_' + str(j) + '_y.txt','w')
    #foutz = open('3d_plot_' + str(j) + '_z.txt','w')
    for line in f.readlines():
        if line[0] != 't' and line[0] != 'w' and line[0] != 'm': # skip mom too
            ls = line.split()
            n = int(ls[0])
            if n == j:
                #if ls[1] != 'z':
                    #c = ls[1]
                w00 = float(ls[2])/max_weight
                w01 = float(ls[3])/max_weight
                w02 = float(ls[4])/max_weight
                w10 = float(ls[5])/max_weight
                w11 = float(ls[6])/max_weight
                w12 = float(ls[7])/max_weight
                a = {'1': [w00, w01, w12],
                    '2': [w00, w11, w02],
                    '3':[w00, w11, w12],
                    '4':[w10, w01, w02],
                    '5':[w10, w01, w12],
                    '6':[w10, w11, w02],
                    '0':[w00, w01, w02],
                    '7':[w10, w11, w12]}

                #foutx.write(str(a[canon[n]][0])+'\n')
                #fouty.write(str(a[canon[n]][1])+'\n')
                #foutz.write(str(a[canon[n]][2])+'\n')
                fout.write(str(a[canon[n]][0])+'\t'+ str(a[canon[n]][1])+'\t'+ str(a[canon[n]][2]) +'\n')
                #print str(a[canon[n]][0])+'\t'+ str(a[canon[n]][1])+'\t'+ str(a[canon[n]][2]) +'\n'
		#raw_input()
    f.close()
    fout.close()
    #foutx.close()
    #fouty.close()
    #foutz.close()




m_out.write("};")

# print the rest of the matlab file

for line in open(m_dir + m_file,'r'):
  	m_out.write(line)

m_out.close()