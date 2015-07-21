/*
 * HeliTileMovie.cpp
 * Author: slundquist
 */

#include "HeliTileMovie.hpp"
#include <assert.h>
#include <dirent.h>

namespace PV {

HeliTileMovie::HeliTileMovie()
{
   initialize_base();
}

HeliTileMovie::HeliTileMovie(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

HeliTileMovie::~HeliTileMovie(){
   if(targetCSV) targetCSV.close();
   if(distractorCSV) distractorCSV.close();
}

int HeliTileMovie::initialize_base(){
   targetCSVfilename = NULL;
   distractorCSVfilename = NULL;
   timestampFilename = NULL;
   csvXScale = 1;
   csvYScale = 1;
   csvXOffset= 0;
   csvYOffset = 0;

   allPos.clear();
   allPosIdx = 0;
   groundTruth.clear();
   newFrameNeeded = true;
}

int HeliTileMovie::initialize(const char * name, HyPerCol * hc) {
   //TODO make only root process do this
   int status = Movie::initialize(name, hc);

   if (parent->columnId()==0) {
      //Open the 2 csv files
      targetCSV.open(targetCSVfilename, std::ifstream::in);
      if (!targetCSV.is_open()){
         std::cout << "Unable to open file " << targetCSVfilename << "\n";
         exit(EXIT_FAILURE);
      }
      distractorCSV.open(distractorCSVfilename, std::ifstream::in);
      if (!distractorCSV.is_open()){
         std::cout << "Unable to open file " << distractorCSVfilename << "\n";
         exit(EXIT_FAILURE);
      }
      //Read one line from both files for the header
      getline(targetCSV, targetString);
      getline(distractorCSV, distractorString);

      //Read first line
      getline (targetCSV,targetString);
      getline(distractorCSV, distractorString);

      //Open the timestamp file
      timestampFile.open(timestampFilename, std::ifstream::in);
      if (!timestampFile.is_open()){
         std::cout << "Unable to open file " << timestampFilename << "\n";
         exit(EXIT_FAILURE);
      }
      //Read first line
      getline (timestampFile,timestampString);
      //std::cout << "First line: " << timestampString << "\n";
      //Get display period
      float firsttime = getTimestampTime(timestampString);
      getline (timestampFile,timestampString);
      //Get display period
      timestampDisplayPeriod = getTimestampTime(timestampString) - firsttime;
      //Reset filepointer
      timestampFile.seekg (0, timestampFile.beg);
      getline (timestampFile,timestampString);
      //std::cout << "Regrabbing first line: " << timestampString << "\n";
   }
   return status;
}

int HeliTileMovie::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = Movie::ioParamsFillGroup(ioFlag);
   ioParam_TargetCSV(ioFlag);
   ioParam_DistractorCSV(ioFlag);
   ioParam_CSVScale(ioFlag);
   ioParam_CSVOffset(ioFlag);
   ioParam_TimestampFile(ioFlag);
   return status;
}

int HeliTileMovie::ioParam_offsets(enum ParamsIOFlag ioFlag){
   //Does nothing, since this layer is explicitly setting offsets
   return PV_SUCCESS;
}

void HeliTileMovie::ioParam_TimestampFile(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "timestampFile", &timestampFilename);
}

void HeliTileMovie::ioParam_TargetCSV(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "targetCSV", &targetCSVfilename);
}

void HeliTileMovie::ioParam_DistractorCSV(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "distractorCSV", &distractorCSVfilename);
}

void HeliTileMovie::ioParam_CSVScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "csvXScale", &csvXScale, csvXScale);
   parent->ioParamValue(ioFlag, name, "csvYScale", &csvYScale, csvYScale);
}

void HeliTileMovie::ioParam_CSVOffset(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "csvXOffset", &csvXOffset, csvXOffset);
   parent->ioParamValue(ioFlag, name, "csvYOffset", &csvYOffset, csvYOffset);
}

int HeliTileMovie::getCSVFrame(std::string csvString){
   unsigned found = csvString.find_first_of(",");
   std::string stringFrame = csvString.substr(0, found); 
   int csvFrame = atoi(stringFrame.c_str());
   return csvFrame;
}

float HeliTileMovie::getTimestampTime(std::string timestampString){
   std::stringstream lineStream(timestampString);
   std::string cell;
   //First value is frame, don't need
   std::getline(lineStream, cell, ',');
   //Second value is timestamp time, what we need
   std::getline(lineStream, cell, ',');
   return atoi(cell.c_str());
}

std::string HeliTileMovie::getTimestampClipFrame(std::string timestampString){
   std::stringstream lineStream(timestampString);
   std::string cell;
   //First value is frame, don't need
   std::getline(lineStream, cell, ',');
   //Second value is timestamp time, don't need
   std::getline(lineStream, cell, ',');
   //Third value is timestamp time, this is waht we split
   std::getline(lineStream, cell, ',');
   //std::cout << "cell val: " << cell << "\n";
   int found = cell.find_last_of("/");
   if(found == -1){
      std::cout << "Formatting error\n";
      exit(-1);
   }
   std::string outStr = cell.substr(found-3, 3) + "_" + cell.substr(found+1, 5);
   return outStr;
}

std::string HeliTileMovie::getCSVClipFrame(std::string csvString){
   std::stringstream lineStream(csvString);
   std::string cell;
   //Need 14th value
   for(int i = 0; i < 14; i++){
      std::getline(lineStream, cell, ',');
   }
   //Need to parse out clip and frame
   std::stringstream lineStream2(cell);
   std::getline(lineStream2, cell, '_');
   std::string clip = cell;
   std::getline(lineStream2, cell, '_');
   std::string outStr = clip + "_" + cell;
   return outStr;
}

std::pair <int, int> HeliTileMovie::getCSVOffset(std::string csvString){
   //const PVLayerLoc * loc = &imageLoc; 
   //Get scale
   std::stringstream lineStream(csvString);
   std::string cell;
   //First value is csv frame, don't need
   std::getline(lineStream, cell, ',');
   //Get next 2 values, first is x, second is y
   std::getline(lineStream, cell, ',');
   //Truncating to int
   int xpos = (atoi(cell.c_str())-csvXOffset) * csvXScale;
   std::getline(lineStream, cell, ',');
   int ypos = (atoi(cell.c_str())-csvYOffset) * csvYScale;
   //std::cout << "coords: (" << xpos << "," << ypos << ") loc: (" << loc->nx << "," << loc->ny << "\n";
   return std::pair<int, int>(xpos, ypos);
}

int HeliTileMovie::checkpointWrite(const char * cpDir) {
   Movie::checkpointWrite(cpDir);
   parent->writeScalarToFile(cpDir, getName(), "targetCsvState", (long)targetCSV.tellg());
   parent->writeScalarToFile(cpDir, getName(), "distractorCsvState", (long)distractorCSV.tellg());
   parent->writeScalarToFile(cpDir, getName(), "timestampState", (long)timestampFile.tellg());
}


int HeliTileMovie::checkpointRead(const char * cpDir, double * timeptr) {
   Movie::checkpointRead(cpDir, timeptr);
   long targetPos, distractorPos, timestampPos;
   targetPos = 0;
   distractorPos = 0;
   timestampPos = 0;
   parent->readScalarFromFile(cpDir, getName(), "targetCsvState", &targetPos, targetPos);
   parent->readScalarFromFile(cpDir, getName(), "distractorCsvState", &distractorPos, distractorPos);
   parent->readScalarFromFile(cpDir, getName(), "timestampState", &timestampPos, timestampPos);
   targetCSV.seekg(targetPos);
   distractorCSV.seekg(distractorPos);
   timestampFile.seekg(timestampPos);
}

int HeliTileMovie::updateState(double timef, double dt) {
   //TODO need to put rewinding on all mlp stuff
   if (parent->columnId()==0) {
      //Only working on readpvpfiles
      assert(readPvpFile);

      //For some reason, the filename is off by 1. TODO check this
      //Advance frames until the timestamp file matches with the csv filename
      if(allPosIdx >= allPos.size()){

         std::vector <std::pair <int, int> > targetPos;
         std::vector <std::pair <int, int> > distractorPos;

         //Need to match csv filename to timestamp filename
         //It's the targets that matter, so only grab the target
         std::string targetClipFrame = getCSVClipFrame(targetString);
         std::string distractorClipFrame = getCSVClipFrame(distractorString);
         std::string timestampClipFrame = getTimestampClipFrame(timestampString);

         //Spin target clips until it matches with timestampClipFrame
         while(targetClipFrame < timestampClipFrame){
            getline(targetCSV, targetString);
            targetClipFrame = getCSVClipFrame(targetString);
         }
         //Spin distractor clips until it matches with targetClipFrame 
         while(distractorClipFrame < targetClipFrame){
            getline(distractorCSV, distractorString);
            distractorClipFrame = getCSVClipFrame(distractorString);
         }
         //Spin timestamps until it matches exactly with targets
         //Update timestamp lines csvClipFrame
         //While the strings are unequal
         while(targetClipFrame > timestampClipFrame){
            getline (timestampFile, timestampString);
            if(timestampFile.eof()){
               std::cout << "EOF reached\n";
               exit(1);
            }
            timestampClipFrame = getTimestampClipFrame(timestampString);
         }

         assert(targetClipFrame == timestampClipFrame);

         //timestamp line is now the correct line that matches the instance of the csv clip
         float timestampTime = getTimestampTime(timestampString);
         float targetTimestampTime = timestampTime + timestampDisplayPeriod - 1;
         //Need to match timestamp file's time to the pvp file time
         //Advance frames until we reach the target timestamp time
         int lastFrameNum = frameNumber;
         while(getPvpFileTime() < targetTimestampTime){
            std::cout << "pvpFileTime: " << getPvpFileTime() << " targetTimestampTime: " << targetTimestampTime << "\n";
            updateFrameNum(1);
            //See if the frame numbers rolled back
            if(frameNumber < lastFrameNum){
               std::cout << "Target Timestamp of " << targetTimestampTime << " does not exist\n";
               exit(EXIT_FAILURE);
            }
            //Update image
            readImage(filename);
         }
         //now at the right frame
         targetPos.clear();
         distractorPos.clear();
         allPos.clear();
         groundTruth.clear();
         //TODO there's a chance we can skip over the clip we're looking for, so need to check that, especially if there's no cars in the clip
         ////Spin distractors and cars until it matches the timestamp clip frame
         //while(targetClipFrame.compare(timestampClipFrame) != 0){
         //   getline(targetCSV, targetString);
         //   targetClipFrame = getCSVClipFrame(targetString);
         //}
         //while(distractorClipFrame.compare(timestampClipFrame) != 0){
         //   getline(distractorCSV, distractorString);
         //   distractorClipFrame = getCSVClipFrame(distractorString);
         //}
         
         //Need to read all bounding boxes of the frames 
         while(targetClipFrame.compare(timestampClipFrame) == 0){
            //std::cout << "layer " << name << ": targetString << "\n";
            targetPos.push_back(getCSVOffset(targetString));
            getline(targetCSV, targetString);
            targetClipFrame = getCSVClipFrame(targetString);
         }
         while(distractorClipFrame.compare(timestampClipFrame) == 0){
            //std::cout << distractorString << "\n";
            distractorPos.push_back(getCSVOffset(distractorString));
            getline(distractorCSV, distractorString);
            distractorClipFrame = getCSVClipFrame(distractorString);
         }
         std::cout.flush();

         //Alternate between target and distractor
         //2 distractors per car
         unsigned altLen = targetPos.size() < distractorPos.size() ? targetPos.size() : distractorPos.size();
         for(unsigned i = 0; i < altLen; i++){
            allPos.push_back(targetPos[i]);
            groundTruth.push_back(1);
            allPos.push_back(distractorPos[i]);
            groundTruth.push_back(-1);
         }
         //Add rest of distractors/targets
         for(unsigned i = altLen; i < targetPos.size(); i++){
            allPos.push_back(targetPos[i]);
            groundTruth.push_back(1);
         }
         for(unsigned i = altLen; i < distractorPos.size(); i++){
            allPos.push_back(distractorPos[i]);
            groundTruth.push_back(-1);
         }
         //Clear temp vectors
         targetPos.clear();
         distractorPos.clear();
         //Reset index
         allPosIdx = 0;
      }

      //std::cout << "layer " << name << " run time: " << timef << " run Pos: (" << allPos[allPosIdx].first << "," << allPos[allPosIdx].second << ")  Gt: " << groundTruth[allPosIdx] << "\n";
      assert(allPosIdx < allPos.size());

      GDALColorInterp * colorbandtypes = NULL;
      int status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "Movie %s: Error getting image info \"%s\"\n", name, filename);
         abort();
      }
      //Read image
      readImage(filename, allPos[allPosIdx].first, allPos[allPosIdx].second, colorbandtypes);
      allPosIdx++;
   }
   return PV_SUCCESS;
}
}
