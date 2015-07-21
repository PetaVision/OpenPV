from os import system

#2.5
print "Lateral Interactions, SNR = 2.5"
system("cp filenames_2_5.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI2.5")
print "No Lateral Interactions, SNR = 2.5"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI2.5")

#5
print "Lateral Interactions, SNR = 5"
system("cp filenames_5.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI5")
print "No Lateral Interactions, SNR = 5"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI5")

#10
print "Lateral Interactions, SNR = 10"
system("cp filenames_10.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI10")
print "No Lateral Interactions, SNR = 10"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI10")

#20
print "Lateral Interactions, SNR = 20"
system("cp filenames_20.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI20")
print "No Lateral Interactions, SNR = 20"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI20")

#40
print "Lateral Interactions, SNR = 40"
system("cp filenames_40.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI40")
print "No Lateral Interactions, SNR = 40"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI40")

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

#100
print "Lateral Interactions, SNR = 100"
system("cp filenames_100.txt filenames.txt")
system("../Debug/BIDS -p params_LI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputLI100")
print "No Lateral Interactions, SNR = 100"
system("../Debug/BIDS -p params_NLI.pv 1> ../log.txt")
system("mv ../log.txt ../output")
system("mv ../output ../outputNLI100")
