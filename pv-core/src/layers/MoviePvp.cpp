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
//#include <iostream>

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
   //filenamestream = NULL;
   displayPeriod = DISPLAY_PERIOD;
   //readPvpFile = false;
   //fileOfFileNames = NULL;
   frameNumbers = NULL;
   //fileNumFrames = 0;
   //fileNumBatches = 0;
   writeFrameToTimestamp = true;
   timestampFile = NULL;
   flipOnTimescaleError = true;
   resetToStartOnLoop = false;
   initFlag = false;
   batchMethod = NULL;
   //updateThisTimestep = false;
   // newImageFlag = false;
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

   //if (!readPvpFile) {
   //   int startFrame = frameNumber;
   //   if (parent->columnId()==0) {
   //      PV_fseek(filenamestream, 0L, SEEK_SET);
   //      frameNumber = 0;
   //   }
   //   if (filename != NULL) free(filename);
   //   filename = strdup(getNextFileName(startFrame)); // getNextFileName() will increment frameNumber by startFrame;
   //   if (parent->columnId()==0) assert(frameNumber==startFrame);
   //   if (parent->columnId()==0) {
   //      printf("%s \"%s\" checkpointRead set frameNumber to %d and filename to \"%s\"\n",
   //            getKeyword(), name, frameNumber, filename);
   //   }
   //}
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

   //PVParams * params = hc->parameters();

   //assert(!params->presentAndNotBeenRead(name, "randomMovie")); // randomMovie should have been set in ioParams
   //if (randomMovie) return status; // Nothing else to be done until data buffer is allocated, in allocateDataStructures


   ////If not pvp file, open fileOfFileNames 
   //assert(!params->presentAndNotBeenRead(name, "readPvpFile")); // readPvpFile should have been set in ioParams
   //if( getParent()->icCommunicator()->commRank()==0 && !readPvpFile) {
   //   filenamestream = PV_fopen(fileOfFileNames, "r", false/*verifyWrites*/);
   //   if( filenamestream == NULL ) {
   //      fprintf(stderr, "Movie::initialize error opening \"%s\": %s\n", fileOfFileNames, strerror(errno));
   //      abort();
   //   }
   //}

   //if (startFrameIndex <= 1){
   //   frameNumber = 0;
   //}
   //else{
   //   frameNumber = startFrameIndex - 1;
   //}
   ////Set filename as param
   //Grab number of frames from header
   
   //PV_Stream * pvstream = NULL;
   //if (getParent()->icCommunicator()->commRank()==0) {
   //   pvstream = PV::PV_fopen(inputPath, "rb", false/*verifyWrites*/);
   //}
   //int numParams = NUM_PAR_BYTE_PARAMS;
   //int params[numParams];
   //pvp_read_header(pvstream, getParent()->icCommunicator(), params, &numParams);
   //PV::PV_fclose(pvstream); pvstream = NULL;
   //fileNumFrames = params[INDEX_NBANDS]; 
   //fileNumBatches = params[INDEX_NBATCH];

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
   //ioParam_randomMovie(ioFlag);
   //ioParam_randomMovieProb(ioFlag);
   //ioParam_readPvpFile(ioFlag);
   //ioParam_echoFramePathnameFlag(ioFlag);
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
      std::cout << "Movie layer " << name << " batchMethod not recognized. Options are \"byImage\", \"byMovie\", and \"bySpecified\"\n";
      exit(-1);
   }
}

//void MoviePvp::ioParam_randomMovie(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "randomMovie", &randomMovie, 0/*default value*/);
//}
//
//void MoviePvp::ioParam_randomMovieProb(enum ParamsIOFlag ioFlag) {
//   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
//   if (randomMovie) {
//      parent->ioParamValue(ioFlag, name, "randomMovieProb", &randomMovieProb, 0.05f);
//   }
//}

//void MoviePvp::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
//   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
//   if (!randomMovie) {
//      assert(!parent->parameters()->presentAndNotBeenRead(name, "readPvpFile"));
//      if (!readPvpFile) {
//         parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &echoFramePathnameFlag, false/*default value*/);
//      }
//   }
//}

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
   //if (imageData != NULL) {
   //   delete imageData;
   //   imageData = NULL;
   //}
   //if (getParent()->icCommunicator()->commRank()==0 && filenamestream != NULL && filenamestream->isfile) {
   //   PV_fclose(filenamestream);
   //}
   //free(fileOfFileNames); fileOfFileNames = NULL;
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
         std::cout << "Movie layer " << name << " batchMethod of \"" << batchMethod << "\" sets skip_frame_index, do not specify.\n"; 
         exit(-1);
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
         std::cout << "Movie layer " << name << " batchMethod of \"" << batchMethod << "\" requires 0 or 1 start_frame_index values\n"; 
         exit(-1);
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
         std::cout << "Movie layer " << name << " batchMethod of \"byImage\" sets skip_frame_index, do not specify.\n"; 
         exit(-1);
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
         std::cout << "Movie layer " << name << " batchMethod of \"byMovie\" requires 0 or 1 start_frame_index values\n"; 
         exit(-1);
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
         std::cout << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatch << " start_frame_index values\n"; 
         exit(-1);
      }
      if(numSkipFrame != nbatch && numSkipFrame != 0){
         std::cout << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatch << " skip_frame_index values\n"; 
         exit(-1);
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

pvdata_t * MoviePvp::getImageBuffer()
{
   //   return imageData;
   return data;
}

PVLayerLoc MoviePvp::getImageLoc()
{
   return imageLoc;
   //   return clayer->loc;
   // imageLoc contains size information of the image file being loaded;
   // clayer->loc contains size information of the layer, which may
   // be smaller than the whole image.  To get information on the layer, use
   // getLayerLoc().  --pete 2011-07-10
}

double MoviePvp::getDeltaUpdateTime(){
   //If jitter or randomMovie, update every timestep
   if( jitterFlag ){
      return parent->getDeltaTime();
   }
   //if(randomMovie){
   //   return parent->getDeltaTime();
   //}
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
int MoviePvp::retrieveData(double timef, double dt)
{
   bool init = false;
   //TODO this function needs to decide how to read into batches
   int status = PV_SUCCESS;
   for(int b = 0; b < parent->getNBatch(); b++){
      if(!initFlag){
         init = true;
      }
      else{
         updateFrameNum(skipFrameIndex[b], b);
      }

      //Using member varibles here
      status = readPvp(inputPath, frameNumbers[b], b, offsets[0], offsets[1], offsetAnchor);
      assert(status == PV_SUCCESS);
   }
   if(init){
      initFlag = true;
   }
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
   //if( jitterFlag ) {
   //   jitter();
   //} // jitterFlag

   InterColComm * icComm = getParent()->icCommunicator();

   //if(randomMovie){
   //   randomFrame();
   //   //Moved to updateStateWrapper
   //   //lastUpdateTime = time;
   //} else {
      //TODO: Fix movie layer to take with batches. This is commented out for compile
      //if(!flipOnTimescaleError && (parent->getTimeScale() > 0 && parent->getTimeScale() < parent->getTimeScaleMin())){
      //   if (parent->icCommunicator()->commRank()==0) {
      //      std::cout << "timeScale of " << parent->getTimeScale() << " is less than timeScaleMin of " << parent->getTimeScaleMin() << ", Movie is keeping the same frame\n";
      //   }
      //}
      //else{
         //if(!readPvpFile){
         //   //Only do this if it's not the first update timestep
         //   //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
         //   //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).  
         //   if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
         //      //If the error is too high, keep the same frame
         //      if (filename != NULL) free(filename);
         //      filename = strdup(getNextFileName(skipFrameIndex));
         //   }
         //   assert(filename != NULL);
         //}
         //else{
            //Only do this if it's not the first update timestep
            //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
            //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).  
            if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
               int status = getFrame(time, dt);
               if( status != PV_SUCCESS ) {
                  fprintf(stderr, "Movie %s: Error reading file \"%s\"\n", name, inputPath);
                  abort();
               }
            }
         //}
      //}
      
      

      if(writePosition && icComm->commRank()==0){
         fprintf(fp_pos->fp,"%f %s: \n",time,inputPath);
      }
      //nextDisplayTime removed, now using nextUpdateTime in HyPerLayer
      //while (time >= nextDisplayTime) {
      //   nextDisplayTime += displayPeriod;
      //}
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

//int MoviePvp::copyReducedImagePortion()
//{
//   const PVLayerLoc * loc = getLayerLoc();
//
//   const int nx = loc->nx;
//   const int ny = loc->ny;
//
//   const int nx0 = imageLoc.nx;
//   const int ny0 = imageLoc.ny;
//
//   assert(nx0 <= nx);
//   assert(ny0 <= ny);
//
//   const int i0 = nx/2 - nx0/2;
//   const int j0 = ny/2 - ny0/2;
//
//   int ii = 0;
//   for (int j = j0; j < j0+ny0; j++) {
//      for (int i = i0; i < i0+nx0; i++) {
//         imageData[ii++] = data[i+nx*j];
//      }
//   }
//
//   return 0;
//}

///**
// * This creates a random image patch (frame) that is used to perform a reverse correlation analysis
// * as the input signal propagates up the visual system's hierarchy.
// * NOTE: Check Image::toGrayScale() method which was the inspiration for this routine
// */
//int MoviePvp::randomFrame()
//{
//   assert(randomMovie); // randomMovieProb was set only if randomMovie is true
//   for (int kex = 0; kex < clayer->numExtended; kex++) {
//      double p = randState->uniformRandom();
//      data[kex] = (p < randomMovieProb) ? 1: 0;
//   }
//   return 0;
//}

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
