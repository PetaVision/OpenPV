To run the params file, you will need an pvpFiles/ directory in the
input/initializationFiles/ directory, with files from the following sources:

From /nh/compneuro/Data/PASCAL_VOC/PASCAL_S1X16_1536_DeepX3_ICA/VOC2007_landscape10_S1_Movie3/
GroundTruth.pvp
GroundTruthReconS1.pvp

From /nh/compneuro/Data/PASCAL_VOC/PASCAL_S1X16_1536_DeepX3_ICA/VOC2007_landscape10_S1_Movie3/Checkpoints/Checkpoint79580
S1ToImageReconS1Error_W.pvp
S1MaxPooled1X1ToGroundTruthReconS1Error_W.pvp
BiasS1ToGroundTruthReconS1Error_W.pvp

To create the confidenceTableS1.bin file used by the ConvertFromTable layer:
cd input/initializationFiles
octave --eval 'createTable;'
