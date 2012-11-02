from collections import OrderedDict
#*************************#
#          PARAMS         #
#*************************#
#Paths
#workspaceDir  = "/Users/slundquist/workspace"
#filenames     = [("sheng","/Users/slundquist/Desktop/ptLIF.txt")]
#filenames     = [("sheng","/Users/slundquist/Desktop/retONtoLif.txt")]

workspaceDir  = "/Users/dpaiton/Documents/Work/Lanl/workspace" #Dylan Mac
checkpointDir = workspaceDir+"/iHouse/checkpoints/Checkpoint2001000"

#filenames = [("label","path")]
filenames    = [
        ("OnVer",checkpointDir+"/retONtoLifVer.txt"),
        ("OnHor",checkpointDir+"/retONtoLifHor.txt"),
        ("OnDia",checkpointDir+"/retONtoLifDia.txt"),
        ("OffVer",checkpointDir+"/retOFFtoLifVer.txt"),
        ("OffHor",checkpointDir+"/retOFFtoLifHor.txt"),
        ("OffDia",checkpointDir+"/retOFFtoLifDia.txt")]
#filenames     = [
#        ("OnVer",checkpointDir+"/retONtoLifVer.txt")]

rootFigOutDir = checkpointDir+"/analysis/probeFigs/"
rootFigName   = 'weights'

#Values for range of frames
startTime   = 2000000
endTime     = 2001000  #End must be under number of lines in file

#Which plots
timePlot    = True 
weightMap   = True 


#Other flags
numTCBins   = 2     #number of bins for time course plot
doLegend    = False #if True, time graph will have a legend
dispFigs    = False #if True, display figures. Otherwise, print them to file.

#Data structure for scale, and data array to store all the data
data = OrderedDict()
#Made time for data
#TIME MUST EXIST AND BE FIRST IN THIS LIST
data['t']                     = []

####
####OJA STDP CONN
####
#data['prOjaTr*']    = []
#data['prStdpTr*']   = []
#######
#data['prOjaTr_0_0']    = []
#data['prOjaTr_0_1']    = []
#data['prOjaTr_0_2']    = []
#data['prOjaTr_0_3']    = []
#data['prOjaTr_0_4']    = []
#data['prOjaTr_0_5']    = []
#data['prOjaTr_0_6']    = []
#data['prOjaTr_0_7']    = []
#data['prOjaTr_0_8']    = []
#data['prOjaTr_0_9']    = []
#data['prOjaTr_0_10']    = []
#data['prOjaTr_0_11']    = []
#data['prOjaTr_0_12']    = []
#data['prOjaTr_0_13']    = []
#data['prOjaTr_0_14']    = []
#data['prOjaTr_0_15']    = []
#data['prOjaTr_0_16']    = []
#data['prOjaTr_0_17']    = []
#data['prOjaTr_0_18']    = []
#data['prOjaTr_0_19']    = []
#data['prOjaTr_0_20']    = []
#data['prOjaTr_0_21']    = []
#data['prOjaTr_0_22']    = []
#data['prOjaTr_0_23']    = []
#data['prOjaTr_0_24']    = []
#######
#data['poOjaTr']     = []
#data['poStdpTr']    = []
#data['poIntTr']     = []
#######
#data['ampLTD']      = []
#######
#data['weight_0_0']  = []
#data['weight_0_1']  = []
#data['weight_0_2']  = []
#data['weight_0_3']  = []
#data['weight_0_4']  = []
#data['weight_0_5']  = []
#data['weight_0_6']  = []
#data['weight_0_7']  = []
#data['weight_0_8']  = []
#data['weight_0_9']  = []
#data['weight_0_10'] = []
#data['weight_0_11'] = []
#data['weight_0_12'] = []
#data['weight_0_13'] = []
#data['weight_0_14'] = []
#data['weight_0_15'] = []
#data['weight_0_16'] = []
#data['weight_0_17'] = []
#data['weight_0_18'] = []
#data['weight_0_19'] = []
#data['weight_0_20'] = []
#data['weight_0_21'] = []
#data['weight_0_22'] = []
#data['weight_0_23'] = []
#data['weight_0_24'] = []
#######
data['weight*']     = []

####
####lif layer
####
#data['v']                    = []
#data['vth']                  = []
#data['a']                    = []

#set scales for plots. Key must be the same as what is in the data dictionary
scale = {}
#scale['weight_0_0']  = 100
#scale['weight_0_1']  = 100
#scale['weight_0_2']  = 100
#scale['weight_0_3']  = 100
#scale['weight_0_4']  = 100
#scale['weight_0_5']  = 100
#scale['weight_0_6']  = 100
#scale['weight_0_7']  = 100
#scale['weight_0_8']  = 100
#scale['weight_0_9']  = 100
#scale['weight_0_10'] = 100
#scale['weight_0_11'] = 100
#scale['weight_0_12'] = 100
#scale['weight_0_13'] = 100
#scale['weight_0_14'] = 100
#scale['weight_0_15'] = 100
#scale['weight_0_16'] = 100
#scale['weight_0_17'] = 100
#scale['weight_0_18'] = 100
#scale['weight_0_19'] = 100
#scale['weight_0_20'] = 100
#scale['weight_0_21'] = 100
#scale['weight_0_22'] = 100
#scale['weight_0_23'] = 100
#scale['weight_0_24'] = 100
#scale['weight_4_0']  = 100
#scale['weight_4_1']  = 100
#scale['weight_4_2']  = 100
#scale['weight_4_3']  = 100
#scale['weight_4_4']  = 100
#scale['weight_4_5']  = 100
#scale['weight_4_6']  = 100
#scale['weight_4_7']  = 100
#scale['weight_4_8']  = 100
#scale['weight_4_9']  = 100
#scale['weight_4_10'] = 100
#scale['weight_4_11'] = 100
#scale['weight_4_12'] = 100
#scale['weight_4_13'] = 100
#scale['weight_4_14'] = 100
#scale['weight_4_15'] = 100
#scale['weight_4_16'] = 100
#scale['weight_4_17'] = 100
#scale['weight_4_18'] = 100
#scale['weight_4_19'] = 100
#scale['weight_4_20'] = 100
#scale['weight_4_21'] = 100
#scale['weight_4_22'] = 100
#scale['weight_4_23'] = 100
#scale['weight_4_24'] = 100
