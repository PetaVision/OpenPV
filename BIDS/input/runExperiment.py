from os import system

#60
print "Lateral Interactions, SNR = 60"
system("cp filenames_60.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI60")
print "No Lateral Interactions, SNR = 60"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI60")

#80
print "Lateral Interactions, SNR = 80"
system("cp filenames_80.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI80")
print "No Lateral Interactions, SNR = 80"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI80")

#90
print "Lateral Interactions, SNR = 90"
system("cp filenames_90.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI90")
print "No Lateral Interactions, SNR = 90"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI90")
