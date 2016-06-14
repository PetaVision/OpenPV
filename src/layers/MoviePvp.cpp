/*
 * MoviePvp.cpp
 *
 *  Created on: July 14, 2015 
 *      Author: slundquist 
 */

#include "MoviePvp.hpp"
#include "../include/default_params.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>

namespace PV {

MoviePvp::MoviePvp() {
   initialize_base();
}

MoviePvp::MoviePvp(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int MoviePvp::initialize_base() {
   movieOutputPath = NULL;
   startFrameIndex = NULL;
   skipFrameIndex = NULL;
   paramsStartFrameIndex = NULL;
   paramsSkipFrameIndex = NULL;
   numStartFrame = 0;
   numSkipFrame = 0;
   echoFramePathnameFlag = false;
   displayPeriod = DISPLAY_PERIOD;
   frameNumbers = NULL;
   writeFrameToTimestamp = true;
   timestampFile = NULL;
   flipOnTimescaleError = true;
   resetToStartOnLoop = false;
   batchMethod = NULL;
   return PV_SUCCESS;
}

int MoviePvp::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = ImagePvp::readStateFromCheckpoint(cpDir, timeptr);
   status = readFrameNumStateFromCheckpoint(cpDir);
   return status;
}

int MoviePvp::readFrameNumStateFromCheckpoint(const char * cpDir) {
   int status = PV_SUCCESS;
   parent->readArrayFromFile(cpDir, getName(), "FrameNumState", frameNumbers, parent->getNBatch());
   return status;
}

int MoviePvp::checkpointRead(const char * cpDir, double * timef){
   int status = ImagePvp::checkpointRead(cpDir, timef);

   // should this be moved to readStateFromCheckpoint?
   if (writeFrameToTimestamp) {
      long timestampFilePos = 0L;
      parent->readScalarFromFile(cpDir, getName(), "TimestampState", &timestampFilePos, timestampFilePos);
      if (timestampFile) {
         assert(parent->columnId()==0);
         if (PV_fseek(timestampFile, timestampFilePos, SEEK_SET) != 0) {
            fprintf(stderr, "MovieLayer::checkpointRead error: unable to recover initial file position in timestamp file for layer %s: %s\n", name, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }

   return status;
}

int MoviePvp::checkpointWrite(const char * cpDir){
   int status = ImagePvp::checkpointWrite(cpDir);

   parent->writeArrayToFile(cpDir, getName(), "FrameNumState", frameNumbers, parent->getNBatch());

   //Only do a checkpoint TimestampState if there exists a timestamp file
   if (timestampFile) {
      long timestampFilePos = getPV_StreamFilepos(timestampFile);
      parent->writeScalarToFile(cpDir, getName(), "TimestampState", timestampFilePos);
   }

   return status;
}

//
/*
 * Notes:
 * - writeImages, offsetX, offsetY are initialized by Image::initialize()
 */
int MoviePvp::initialize(const char * name, HyPerCol * hc) {
   int status = ImagePvp::initialize(name, hc);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Image::initialize failed on Movie layer \"%s\".  Exiting.\n", name);
      exit(PV_FAILURE);
   }

   //Update on first timestep
   setNextUpdateTime(parent->simulationTime() + hc->getDeltaTime());

   // set output path for movie frames
   if(writeImages){
      status = parent->ensureDirExists(movieOutputPath);
   }

   if(writeFrameToTimestamp){
      std::string timestampFilename = std::string(parent->getOutputPath());
      timestampFilename += "/timestamps/";
      parent->ensureDirExists(timestampFilename.c_str());
      timestampFilename += name;
      timestampFilename += ".txt";
      if(getParent()->icCommunicator()->commRank()==0){
          //If checkpoint read is set, append, otherwise, clobber
          if(getParent()->getCheckpointReadFlag()){
             struct stat statbuf;
             if (PV_stat(timestampFilename.c_str(), &statbuf) != 0) {
                fprintf(stderr, "%s \"%s\" warning: timestamp file \"%s\" unable to be found.  Creating new file.\n",
                      getKeyword(), name, timestampFilename.c_str());
                timestampFile = PV::PV_fopen(timestampFilename.c_str(), "w", parent->getVerifyWrites());
             }
             else {
                timestampFile = PV::PV_fopen(timestampFilename.c_str(), "r+", false/*verifyWrites*/);
             }
          }
          else{
             timestampFile = PV::PV_fopen(timestampFilename.c_str(), "w", parent->getVerifyWrites());
          }
          assert(timestampFile);
      }
   }
   return PV_SUCCESS;
}

int MoviePvp::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ImagePvp::ioParamsFillGroup(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_batchMethod(ioFlag);
   ioParam_start_frame_index(ioFlag);
   ioParam_skip_frame_index(ioFlag);
   ioParam_movieOutputPath(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);
   ioParam_flipOnTimescaleError(ioFlag);
   ioParam_resetToStartOnLoop(ioFlag);
   return status;
}

void MoviePvp::ioParam_pvpFrameIdx(enum ParamsIOFlag ioFlag) {
   // Image uses frameNumber to pick the frame of a pvp file, but
   // Movie uses start_frame_index to pick the starting frame.
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "pvpFrameIdx");
   }
}

void MoviePvp::ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "flipOnTimescaleError", &flipOnTimescaleError, flipOnTimescaleError);
}

void MoviePvp::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, displayPeriod);
}

void MoviePvp::ioParam_batchMethod(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "bySpecified");
   if(strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "bySpecified") == 0){
      //Correct
   }
   else{
      pvError() << "Movie layer " << name << " batchMethod not recognized. Options are \"byImage\", \"byMovie\", and \"bySpecified\"\n";
   }
}

void MoviePvp::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamArray(ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &numStartFrame);
}

void MoviePvp::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamArray(ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &numSkipFrame);
}

void MoviePvp::ioParam_movieOutputPath(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeImages"));
   if (writeImages){
      parent->ioParamString(ioFlag, name, "movieOutputPath", &movieOutputPath, parent->getOutputPath());
   }
}

void MoviePvp::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeFrameToTimestamp", &writeFrameToTimestamp, writeFrameToTimestamp);
}

void MoviePvp::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "resetToStartOnLoop", &resetToStartOnLoop, resetToStartOnLoop);
}

MoviePvp::~MoviePvp()
{
   if (getParent()->icCommunicator()->commRank()==0 && timestampFile != NULL && timestampFile->isfile) {
       PV_fclose(timestampFile);
   }
   free(movieOutputPath);
   free(paramsStartFrameIndex);
   free(startFrameIndex);
   free(paramsSkipFrameIndex);
   free(skipFrameIndex);
   free(frameNumbers);
   free(batchMethod);
}

int MoviePvp::allocateDataStructures() {

   //Get file information
   Communicator* comm = parent->icCommunicator();

   startFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(startFrameIndex);
   skipFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(skipFrameIndex);

   int nbatch = parent->getNBatch();
   assert(batchMethod);

   if(strcmp(batchMethod, "byImage") == 0){
      //No skip here allowed
      if(numSkipFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"" << batchMethod << "\" sets skip_frame_index, do not specify.\n";
      }

      int offset = 0;
      //Default value
      if(numStartFrame == 0){
      }
      //Uniform start array
      else if(numStartFrame == 1){
         offset = *paramsStartFrameIndex;
      }
      else{
         pvError() << "Movie layer " << name << " batchMethod of \"" << batchMethod << "\" requires 0 or 1 start_frame_index values\n";
      }
      //Allocate and default
      //Not done in allocate, as Image Allocate needs this parameter to be set
      int kb0 = getLayerLoc()->kb0;
      int nbatchGlobal = getLayerLoc()->nbatchGlobal;
      for(int b = 0; b < nbatch; b++){ 
         startFrameIndex[b] = offset + kb0 + b;
         skipFrameIndex[b] = nbatchGlobal;
      }
   }
   else if (strcmp(batchMethod, "byMovie") == 0){
      //No skip here allowed
      if(numSkipFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"byImage\" sets skip_frame_index, do not specify.\n";
      }
      
      int offset = 0;
      //Default value
      if(numStartFrame== 0){
      }
      //Uniform start array
      else if(numStartFrame== 1){
         offset = *paramsStartFrameIndex;
      }
      else{
         pvError() << "Movie layer " << name << " batchMethod of \"byMovie\" requires 0 or 1 start_frame_index values\n";
      }

      int nbatchGlobal = getLayerLoc()->nbatchGlobal;
      int kb0 = getLayerLoc()->kb0;

      int framesPerBatch = floor(fileNumFrames/nbatchGlobal);
      if(framesPerBatch < 1){
         framesPerBatch = 1;
      }
      for(int b = 0; b < nbatch; b++){ 
         //+1 for 1 indexed
         startFrameIndex[b] = offset + ((b+kb0)*framesPerBatch);
         skipFrameIndex[b] = 1;
      }
   }
   else if(strcmp(batchMethod, "bySpecified") == 0){
      if(numStartFrame != nbatch && numStartFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatch << " start_frame_index values\n";
      }
      if(numSkipFrame != nbatch && numSkipFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatch << " skip_frame_index values\n";
      }
      for(int b = 0; b < nbatch; b++){ 
         if(numStartFrame == 0){
            //+1 for 1 indexed
            startFrameIndex[b] = 0;
         }
         else{
            startFrameIndex[b] = paramsStartFrameIndex[b];
         }
         if(numSkipFrame == 0){
            skipFrameIndex[b] = 1;
         }
         else{
            skipFrameIndex[b] = paramsSkipFrameIndex[b];
         }
      }
   }
   else{
      //This should never excute, as this check was done in the reading of this parameter
      assert(0);
   }

   //Allocate and set frameNumbers
   frameNumbers = (int*) malloc(parent->getNBatch() * sizeof(int));
   for(int b = 0; b < nbatch; b++){
      frameNumbers[b] = startFrameIndex[b];
   }



   int status = ImagePvp::allocateDataStructures();
   assert(status == PV_SUCCESS);

   return status;
}

PVLayerLoc MoviePvp::getImageLoc()
{
   return imageLoc;
   // imageLoc contains size information of the image file being loaded;
   // clayer->loc contains size information of the layer, which may
   // be smaller than the whole image.  To get information on the layer, use
   // getLayerLoc().  --pete 2011-07-10
}

double MoviePvp::getDeltaUpdateTime(){
   //If jittering, update every timestep
   if( jitterFlag ){
      return parent->getDeltaTime();
   }
   return displayPeriod;
}

  // ensure that timeScale == 1 if new frame being loaded on current time step
  // (note: this used to add getDeltaTime to simulationTime, but calcTimeScale is now called after simulationTime is incremented. -pfs 2015-11-05)
  
double MoviePvp::calcTimeScale(int batchIdx){
    if(needUpdate(parent->simulationTime(), parent->getDeltaTime())){
      return parent->getTimeScaleMin(); 
    }
    else{
      return HyPerLayer::calcTimeScale(batchIdx);
    }
  }

int MoviePvp::updateState(double time, double dt)
{
   updateImage(time, dt);
   return PV_SUCCESS;
}

//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int MoviePvp::retrieveData(double timef, double dt, int batchIndex)
{
   //TODO this function needs to decide how to read into batches
   int status = PV_SUCCESS;
   if(timef>parent->getStartTime()){
      updateFrameNum(skipFrameIndex[batchIndex], batchIndex);
   }

   //Using member varibles here
   status = readPvp(inputPath, frameNumbers[batchIndex]);
   assert(status == PV_SUCCESS);
   return status;
}


/**
 * - Update the image buffers
 * - If the time is a multiple of biasChangetime then the position of the bias (biasX, biasY) changes.
 * - With probability persistenceProb the offset position (offsetX, offsetY) remains unchanged.
 * - Otherwise, with probability (1-persistenceProb) the offset position performs a random walk
 *   around the bias position (biasX, biasY).
 *
 * - If the time is a multiple of displayPeriod then load the next image.
 * - If nf=1 then the image is converted to grayscale during the call to read(filename, offsetX, offsetY).
 *   If nf>1 then the image is loaded with color information preserved.
 * - Return true if buffers have changed
 */
bool MoviePvp::updateImage(double time, double dt)
{
   InterColComm * icComm = getParent()->icCommunicator();

            if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
               int status = getFrame(time, dt);
               if( status != PV_SUCCESS ) {
                  fprintf(stderr, "Movie %s: Error reading file \"%s\"\n", name, inputPath);
                  abort();
               }
            }
      if(writePosition && icComm->commRank()==0){
         fprintf(fp_pos->fp,"%f %s: \n",time,inputPath);
      }
      //Set frame number (member variable in Image)
      //Write to timestamp file here when updated
      if( icComm->commRank()==0 ) {
          //Only write if the parameter is set
          if(timestampFile){
             std::ostringstream outStrStream;
             outStrStream.precision(15);
             int kb0 = getLayerLoc()->kb0;
             for(int b = 0; b < parent->getNBatch(); b++){
                outStrStream << time << "," << b+kb0 << "," << frameNumbers[b] << "," << inputPath << "\n";
             }
             size_t len = outStrStream.str().length();
             int status = PV_fwrite(outStrStream.str().c_str(), sizeof(char), len, timestampFile)==len ? PV_SUCCESS : PV_FAILURE;
             if (status != PV_SUCCESS) {
                fprintf(stderr, "%s \"%s\" error: Movie::updateState failed to write to timestamp file.\n", getKeyword(), name);
                exit(EXIT_FAILURE);
             }
             //Flush buffer
             fflush(timestampFile->fp);
          }
      }

   return true;
}

int MoviePvp::outputState(double timed, bool last)
{
   if (writeImages) {
      char basicFilename[PV_PATH_MAX + 1];
      for(int b = 0; b < parent->getNBatch(); b++){
         snprintf(basicFilename, PV_PATH_MAX, "%s/%s_%d_%.2f.%s", movieOutputPath, name, b, timed, writeImagesExtension);
         writeImage(basicFilename, b);
      }
   }

   int status = PV_SUCCESS;
   status = HyPerLayer::outputState(timed, last);

   return status;
}

/**
 * A function only called if readPvpFile is set
 * Will update frameNumber
 */
//This function takes care of rewinding for pvp files
int MoviePvp::updateFrameNum(int n_skip, int batchIdx){
   //assert(readPvpFile);
   InterColComm * icComm = getParent()->icCommunicator();
   int numskip = n_skip < 1 ? 1 : n_skip;
   for(int i_skip = 0; i_skip < numskip; i_skip++){
      int status = updateFrameNum(batchIdx);
      if(status == PV_BREAK){
         break;
      }
   }
   return PV_SUCCESS;
}

int MoviePvp::updateFrameNum(int batchIdx) {
   frameNumbers[batchIdx] += 1;
   //numFrames only set if pvp file
   if(frameNumbers[batchIdx] >= fileNumFrames){
      if(parent->columnId()==0){
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, inputPath );
      }
      if(resetToStartOnLoop){
         frameNumbers[batchIdx] = startFrameIndex[batchIdx];
         return PV_BREAK;
      }
      else{
         frameNumbers[batchIdx] = 0;
      }
   }
   return PV_SUCCESS;
}

BaseObject * createMoviePvp(char const * name, HyPerCol * hc) {
   return hc ? new MoviePvp(name, hc) : NULL;
}

}
