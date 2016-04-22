/*
 * Movie.cpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#include "Movie.hpp"
#include "../include/default_params.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>
//#include <iostream>

namespace PV {

#ifdef PV_USE_GDAL

Movie::Movie() {
   initialize_base();
}

Movie::Movie(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

Movie::~Movie()
{
   if (getParent()->icCommunicator()->commRank()==0 && filenamestream != NULL && filenamestream->isfile) {
      PV_fclose(filenamestream);
   }
   if (getParent()->icCommunicator()->commRank()==0 && timestampFile != NULL && timestampFile->isfile) {
       PV_fclose(timestampFile);
   }
   if(movieOutputPath){
      free(movieOutputPath);
   }
   if(framePath){
      for(int b = 0; b < parent->getNBatch(); b++){
         if(framePath[b]){
            free(framePath[b]);
         }
      }
      free(framePath);
   }
   if(startFrameIndex){
      free(startFrameIndex);
   }
   if(skipFrameIndex){
      free(skipFrameIndex);
   }
   if(paramsStartFrameIndex){
      free(paramsStartFrameIndex);
   }
   if(paramsSkipFrameIndex){
      free(paramsSkipFrameIndex);
   }
   if(batchMethod){
      free(batchMethod);
   }
   if(batchPos){
      free(batchPos);
   }
   if(frameNumbers){
      free(frameNumbers);
   }
}


int Movie::initialize_base() {
   movieOutputPath = NULL;
   echoFramePathnameFlag = false;
   filenamestream = NULL;
   displayPeriod = DISPLAY_PERIOD;
   framePath = NULL;
   numFrames = 0;
   frameNumbers = NULL;
   writeFrameToTimestamp = true;
   timestampFile = NULL;
   flipOnTimescaleError = true;
   resetToStartOnLoop = false;
   startFrameIndex = NULL;
   skipFrameIndex = NULL;
   paramsStartFrameIndex = NULL;
   paramsSkipFrameIndex = NULL;
   numStartFrame = 0;
   numSkipFrame = 0;

   batchPos = NULL;
   batchMethod = NULL;
   //randomMovie commented out Jul 22, 2015
   //randomMovie = false;
   //updateThisTimestep = false;
   // newImageFlag = false;
   initFlag = false;

   return PV_SUCCESS;
}

int Movie::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = Image::readStateFromCheckpoint(cpDir, timeptr);
   status = readFrameNumStateFromCheckpoint(cpDir);
   return status;
}

int Movie::readFrameNumStateFromCheckpoint(const char * cpDir) {
   int status = PV_SUCCESS;
   int nbatch = parent->getNBatch();

   parent->readArrayFromFile(cpDir, getName(), "FilenamePos", batchPos, nbatch);
   parent->readArrayFromFile(cpDir, getName(), "FrameNumbers", frameNumbers, parent->getNBatch());


   //for(int b = 0; b < nbatch; b++){
   //   int startFrame = frameNumber[b];
   //   if (parent->columnId()==0) {
   //      PV_fseek(filenamestream, 0L, SEEK_SET);
   //      frameNumber = 0;
   //   }
   //   if (framePath[b] != NULL) free(framePath[b]);

   //   //Navigates to one frame before, as getFrame will increment getNextFileName 
   //   if(startFrame - skipFrameIndex[b] > 0){
   //      framePath[b] = strdup(getNextFileName(startFrame-skipFrameIndex[b])); // getNextFileName() will increment frameNumber by startFrame;
   //      initFlag = true;
   //   }


   //   if (parent->columnId()==0) assert(frameNumber[b]==startFrame);
   //   if (parent->columnId()==0) {
   //      printf("%s \"%s\" checkpointRead set frameNumber to %d and filename to \"%s\"\n",
   //            getKeyword(), name, frameNumber[b], framePath[b]);
   //   }
   //}
   return status;
}

int Movie::checkpointRead(const char * cpDir, double * timef){
   int status = Image::checkpointRead(cpDir, timef);

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

int Movie::checkpointWrite(const char * cpDir){
   int status = Image::checkpointWrite(cpDir);

   parent->writeArrayToFile(cpDir, getName(), "FilenamePos", batchPos, parent->getNBatch());
   parent->writeArrayToFile(cpDir, getName(), "FrameNumbers", frameNumbers, parent->getNBatch());

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
int Movie::initialize(const char * name, HyPerCol * hc) { int status = Image::initialize(name, hc);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Image::initialize failed on Movie layer \"%s\".  Exiting.\n", name);
      exit(PV_FAILURE);
   }

   //Update on first timestep
   setNextUpdateTime(parent->simulationTime() + hc->getDeltaTime());

   PVParams * params = hc->parameters();

   //assert(!params->presentAndNotBeenRead(name, "randomMovie")); // randomMovie should have been set in ioParams
   //if (randomMovie) return status; // Nothing else to be done until data buffer is allocated, in allocateDataStructures


   //If not pvp file, open fileOfFileNames 
   //assert(!params->presentAndNotBeenRead(name, "readPvpFile")); // readPvpFile should have been set in ioParams
   if (hc->columnId()==0) {
      filenamestream = PV_fopen(inputPath, "r", false/*verifyWrites*/);
      if( filenamestream == NULL ) {
         fprintf(stderr, "Movie::initialize error opening \"%s\": %s\n", inputPath, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }

   //if (!randomMovie) {
   //   //frameNumber handled here
   //   //imageFilename = strdup(getNextFileName(startFrameIndex));
   //   //assert(imageFilename != NULL);
   //}

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

int Movie::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = Image::ioParamsFillGroup(ioFlag);
   //ioParam_imageListPath(ioFlag);
   ioParam_displayPeriod(ioFlag);
   //ioParam_randomMovie(ioFlag);
   //ioParam_randomMovieProb(ioFlag);
   //ioParam_readPvpFile(ioFlag);
   ioParam_echoFramePathnameFlag(ioFlag);
   ioParam_batchMethod(ioFlag);
   ioParam_start_frame_index(ioFlag);
   ioParam_skip_frame_index(ioFlag);
   ioParam_movieOutputPath(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);
   ioParam_flipOnTimescaleError(ioFlag);
   ioParam_resetToStartOnLoop(ioFlag);
   return status;
}

//void Movie::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
//   if (ioFlag == PARAMS_IO_READ) {
//      imageFilename = NULL;
//      parent->parameters()->handleUnnecessaryStringParameter(name, "imageList");
//   }
//}

//void Movie::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
//   // Image uses frameNumber to pick the frame of a pvp file, but
//   // Movie uses start_frame_index to pick the starting frame.
//   if (ioFlag == PARAMS_IO_READ) {
//      filename = NULL;
//      parent->parameters()->handleUnnecessaryParameter(name, "frameNumber");
//   }
//}

//void Movie::ioParam_imageListPath(enum ParamsIOFlag ioFlag) {
//   parent->ioParamStringRequired(ioFlag, name, "imageListPath", &fileOfFileNames);
//}
//
void Movie::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   //Do not use Image's ioParam_writeStep
   BaseInput::ioParam_writeStep(ioFlag);
}

void Movie::ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "flipOnTimescaleError", &flipOnTimescaleError, flipOnTimescaleError);
}

void Movie::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, displayPeriod);
}

//void Movie::ioParam_randomMovie(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "randomMovie", &randomMovie, 0/*default value*/);
//}
//
//void Movie::ioParam_randomMovieProb(enum ParamsIOFlag ioFlag) {
//   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
//   if (randomMovie) {
//      parent->ioParamValue(ioFlag, name, "randomMovieProb", &randomMovieProb, 0.05f);
//   }
//}

//void Movie::ioParam_readPvpFile(enum ParamsIOFlag ioFlag) {
//   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
//   if (!randomMovie) {
//      parent->ioParamValue(ioFlag, name, "readPvpFile", &readPvpFile, false/*default value*/);
//   }
//}

void Movie::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
   //assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   //if (!randomMovie) {
      parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &echoFramePathnameFlag, false/*default value*/);
   //}
}

void Movie::ioParam_batchMethod(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "bySpecified");
   if(strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "bySpecified") == 0){
      //Correct
   }
   else{
      std::cout << "Movie layer " << name << " batchMethod not recognized. Options are \"byImage\", \"byMovie\", and \"bySpecified\"\n";
      exit(-1);
   }
}

void Movie::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   //Read parameter
   this->getParent()->ioParamArray(ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &numStartFrame);
   //if(numStartFrame == 0){
   //   numStartFrameIndex = 1;
   //   *paramsStartFrameIndex = 0;
   //}
}

void Movie::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   //Read parameter
   this->getParent()->ioParamArray(ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &numSkipFrame);
   //if(numSkipFrame == 0){
   //   *paramsSkipFrameIndex = 1;
   //}
}

void Movie::ioParam_movieOutputPath(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeImages"));
   if (writeImages){
      parent->ioParamString(ioFlag, name, "movieOutputPath", &movieOutputPath, parent->getOutputPath());
   }
}

void Movie::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeFrameToTimestamp", &writeFrameToTimestamp, writeFrameToTimestamp);
}

void Movie::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "resetToStartOnLoop", &resetToStartOnLoop, resetToStartOnLoop);
}

int Movie::allocateDataStructures() {


   //Allocate framePaths here before image, since allocate call will call getFrame

   if(parent->icCommunicator()->commRank()==0){
      framePath = (char**) malloc(parent->getNBatch() * sizeof(char*));
      assert(framePath);
      for(int b = 0; b < parent->getNBatch(); b++){
         framePath[b] = NULL;
      }
   }
   
   batchPos = (long*) malloc(parent->getNBatch() * sizeof(long));
   if(batchPos==NULL) {
      fprintf(stderr, "%s \"%s\" error allocating memory for batchPos (batch size %d): %s\n",
            name, getKeyword(), parent->getNBatch(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      batchPos[b] = 0L;
   }
   frameNumbers = (int*) calloc(parent->getNBatch(), sizeof(int));
   if (frameNumbers==NULL) {
      fprintf(stderr, "%s \"%s\" error allocating memory for frameNumbers (batch size %d): %s\n",
            name, getKeyword(), parent->getNBatch(), strerror(errno));
      exit(EXIT_FAILURE);
   }

   //Calculate file positions for beginning of each frame
   numFrames = getNumFrames();
   std::cout << "File " << inputPath << " contains " << numFrames << " frames\n";

   startFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(startFrameIndex);
   skipFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(skipFrameIndex);

   int nbatch = parent->getNBatch();
   assert(batchMethod);

   if(strcmp(batchMethod, "byImage") == 0){
      //No skip here allowed
      if(numSkipFrame != 0){
         std::cout << "Movie layer " << name << " batchMethod of \"byImage\" sets skip_frame_index, do not specify.\n"; 
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
         std::cout << "Movie layer " << name << " batchMethod of \"byImage\" requires 0 or 1 start_frame_index values\n"; 
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

      int framesPerBatch = floor(numFrames/nbatchGlobal);
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
         std::cout << "Movie layer " << name << " batchMethod of \"bySpecified\" requires " << nbatch << " start_frame_index values\n"; 
         exit(-1);
      }
      if(numSkipFrame != nbatch && numSkipFrame != 0){
         std::cout << "Movie layer " << name << " batchMethod of \"bySpecified\" requires " << nbatch << " skip_frame_index values\n"; 
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
   if (parent->columnId()==0) {
      for (int b=0; b<parent->getNBatch(); b++) {
         frameNumbers[b] = -1;
      }
   }

   //Call Image allocate, which will call getFrame
   int status = Image::allocateDataStructures();

   //if (!randomMovie) {
      //assert(!parent->parameters()->presentAndNotBeenRead(name, "start_frame_index"));
      //assert(!parent->parameters()->presentAndNotBeenRead(name, "skip_frame_index"));

      //assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      //if (!autoResizeFlag){
      //   constrainOffsets();  // ensure that offsets keep loc within image bounds
      //}

      // status = readImage(filename, getOffsetX(), getOffsetY()); // readImage already called by Image::allocateDataStructures(), above
      //assert(status == PV_SUCCESS);
   //}
   //else {
   //   if (randState==NULL) {
   //      initRandState();
   //   }
   //   status = randomFrame();
   //}

   return status;
}

pvdata_t * Movie::getImageBuffer()
{
   //   return imageData;
   return data;
}

PVLayerLoc Movie::getImageLoc()
{
   return imageLoc;
   //   return clayer->loc;
   // imageLoc contains size information of the image file being loaded;
   // clayer->loc contains size information of the layer, which may
   // be smaller than the whole image.  To get information on the layer, use
   // getLayerLoc().  --pete 2011-07-10
}

double Movie::getDeltaUpdateTime(){
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
  
double Movie::calcTimeScale(int batchIdx){
    if(needUpdate(parent->simulationTime(), parent->getDeltaTime())){
      return parent->getTimeScaleMin(); 
    }
    else{
      return HyPerLayer::calcTimeScale(batchIdx);
    }
  }

int Movie::updateState(double time, double dt)
{
   updateImage(time, dt);
   return PV_SUCCESS;
}


//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int Movie::retrieveData(double timef, double dt)
{
   int status = PV_SUCCESS;
   bool init = false;
   for(int b = 0; b < parent->getNBatch(); b++){
      if(parent->icCommunicator()->commRank() == 0){
         if(framePath[b]!= NULL) free(framePath[b]);
         if(!initFlag){
            framePath[b] = strdup(getNextFileName(startFrameIndex[b]+1, b));
            init = true;
         }
         else{
            framePath[b] = strdup(getNextFileName(skipFrameIndex[b], b));
         }
         std::cout << "Reading frame " << framePath[b] << " into batch " << b << " at time " << timef << "\n";
         status = readImage(framePath[b], b, offsets[0], offsets[1], offsetAnchor);
      }
      else{
         status = readImage(NULL, b, offsets[0], offsets[1], offsetAnchor);
      }

      if( status != PV_SUCCESS ) {
         fprintf(stderr, "Movie %s: Error reading file \"%s\"\n", name, framePath[b]);
         abort();
      }
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
bool Movie::updateImage(double time, double dt)
{
   if( jitterFlag ) {
      jitter();
   } // jitterFlag

   InterColComm * icComm = getParent()->icCommunicator();

      //TODO: Fix movie layer to take with batches. This is commented out for compile
      //if(!flipOnTimescaleError && (parent->getTimeScale() > 0 && parent->getTimeScale() < parent->getTimeScaleMin())){
      //   if (parent->icCommunicator()->commRank()==0) {
      //      std::cout << "timeScale of " << parent->getTimeScale() << " is less than timeScaleMin of " << parent->getTimeScaleMin() << ", Movie is keeping the same frame\n";
      //   }
      //}
      //else{
         //Only do this if it's not the first update timestep
         //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
         //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).  
      if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
         int status = getFrame(time, dt);
         assert(status == PV_SUCCESS);
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
                outStrStream << time << "," << b+kb0 << "," << frameNumbers[b] << "," << framePath[b] << "\n";
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
   //} // randomMovie

   return true;
}

/**
 * When we play a random frame - in order to perform a reverse correlation analysis -
 * we call writeActivitySparse(time) in order to write the "activity" in the image.
 *
 */
int Movie::outputState(double timed, bool last)
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

//int Movie::copyReducedImagePortion()
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
//int Movie::randomFrame()
//{
//   assert(randomMovie); // randomMovieProb was set only if randomMovie is true
//   for (int kex = 0; kex < clayer->numExtended; kex++) {
//      double p = randState->uniformRandom();
//      data[kex] = (p < randomMovieProb) ? 1: 0;
//   }
//   return 0;
//}

///**
// * A function only called if readPvpFile is set
// * Will update frameNumber
// */
////This function takes care of rewinding for pvp files
//int Movie::updateFrameNum(int n_skip){
//   //assert(readPvpFile);
//   InterColComm * icComm = getParent()->icCommunicator();
//   int numskip = n_skip < 1 ? 1 : n_skip;
//   for(int i_skip = 0; i_skip < numskip; i_skip++){
//      int status = updateFrameNum();
//      if(status == PV_BREAK){
//         break;
//      }
//   }
//   return PV_SUCCESS;
//}

//int Movie::updateFrameNum() {
//   frameNumber += 1;
//   //numFrames only set if pvp file
//   if(frameNumber >= numFrames){
//      if(parent->columnId()==0){
//         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
//      }
//      if(resetToStartOnLoop){
//         frameNumber = startFrameIndex-1;
//         return PV_BREAK;
//      }
//      else{
//         frameNumber = 0;
//      }
//   }
//   return PV_SUCCESS;
//}

// advance by n_skip lines through file of filenames, always advancing at least one line
const char * Movie::getNextFileName(int n_skip, int batchIdx) {
   InterColComm * icComm = getParent()->icCommunicator();
   assert(icComm->commRank() == 0);
   const char* outFilename = NULL;
   int numskip = n_skip < 1 ? 1 : n_skip;
   for (int i_skip = 0; i_skip < numskip; i_skip++){
      outFilename = advanceFileName(batchIdx);
   }
   if (echoFramePathnameFlag){
      printf("%f, %d: %s\n", parent->simulationTime(), batchIdx, outFilename);
   }
   return outFilename;
}

//This function will reset the file position of the open file
int Movie::getNumFrames(){
   int count = 0;
   if(parent->columnId()==0){
      int c;
      PV_fseek(filenamestream, 0L, SEEK_SET);
      while((c = fgetc(filenamestream->fp)) != EOF) {
         count++;
         ungetc(c, filenamestream->fp);
         //Here, we're using batch 0, but we're resetting the batch pos of it at the end
         advanceFileName(0);
      }
      PV_fseek(filenamestream, 0L, SEEK_SET);
      batchPos[0] = 0L;
      frameNumbers[0] = -1;
   }
   MPI_Bcast(&count, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
   return count;
}

//This function takes care of rewinding for frame files
const char * Movie::advanceFileName(int batchIdx) {
   // IMPORTANT!! This function should only be called by getNextFileName(int), and only by the root process
   assert(parent->columnId()==0);

   //Restore position of batch Idx
   PV_fseek(filenamestream, batchPos[batchIdx], SEEK_SET);

   int c;
   size_t maxlen = PV_PATH_MAX;
   bool reset = false;

   // Ignore blank lines
   bool hasrewound = false;
   bool lineisblank = true;
   while(lineisblank) {
      // if at end of file (EOF), rewind
      if ((c = fgetc(filenamestream->fp)) == EOF) {
         PV_fseek(filenamestream, 0L, SEEK_SET);
         frameNumbers[0] = -1;
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, inputPath);
         if (hasrewound) {
            fprintf(stderr, "Movie %s: filenamestream \"%s\" does not have any non-blank lines.\n", name, filenamestream->name);
            exit(EXIT_FAILURE);
         }
         hasrewound = true;
         reset = true;
      }
      else {
         ungetc(c, filenamestream->fp);
      }

      //Always do at least once
      int loopCount = 0;
      do{
         char * path = fgets(inputfile, maxlen, filenamestream->fp);
         if (path != NULL) {
            filenamestream->filepos += strlen(path);
            path[PV_PATH_MAX-1] = '\0';
            size_t len = strlen(path);
            if (len > 0) {
               if (path[len-1] == '\n') {
                  path[len-1] = '\0';
                  len--;
               }
            }
            for (size_t n=0; n<len; n++) {
               if (!isblank(path[n])) {
                  frameNumbers[batchIdx]++;
                  lineisblank = false;
                  break;
               }
            }
            loopCount++;
         }
      }while(resetToStartOnLoop && reset && loopCount < startFrameIndex[batchIdx]);

      assert(strlen(inputfile)>(size_t) 0);
      // assert(inputfile && strlen(inputfile)>(size_t) 0);
      // current version of clang generates a warning since inputfile is a member variable declared as an array and therefore always non-null.
      // Keeping the line in case inputfile is changed to be malloc'd instead of declared as an array.
      char * expandedpath = expandLeadingTilde(inputfile);
      if (strlen(expandedpath)>=PV_PATH_MAX) {
         fprintf(stderr, "Movie \"%s\": input line \"%s\" from imageListPath is too long.\n", name, expandedpath);
         exit(EXIT_FAILURE);
      }
      strncpy(inputfile, expandedpath, PV_PATH_MAX);
      free(expandedpath);
   }
   //Save batch position
   batchPos[batchIdx] = getPV_StreamFilepos(filenamestream);
   return inputfile;
}

//const char * Movie::getCurrentImage(){
//   return inputfile;
//}

#else // PV_USE_GDAL
Movie::Movie(const char * name, HyPerCol * hc) {
   if (hc->columnId()==0) {
      fprintf(stderr, "Movie class requires compiling with PV_USE_GDAL set\n");
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
Movie::Movie() {}
#endif // PV_USE_GDAL

BaseObject * createMovie(char const * name, HyPerCol * hc) {
   return hc ? new Movie(name, hc) : NULL;
}

} // ends namespace PV block
