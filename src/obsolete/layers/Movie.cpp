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
   if (getParent()->getCommunicator()->commRank()==0 && filenamestream != NULL && filenamestream->isfile) {
      PV_fclose(filenamestream);
   }
   if (getParent()->getCommunicator()->commRank()==0 && timestampFile != NULL && timestampFile->isfile) {
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
            pvError().printf("MovieLayer::checkpointRead error: unable to recover initial file position in timestamp file for layer %s: %s\n", name, strerror(errno));
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

/*
 * Notes:
 * - writeImages, offsetX, offsetY are initialized by Image::initialize()
 */
int Movie::initialize(const char * name, HyPerCol * hc) {
   int status = Image::initialize(name, hc);
   if (status != PV_SUCCESS) {
      pvError().printf("Image::initialize failed on Movie layer \"%s\".  Exiting.\n", name);
   }

   //Update on first timestep
   setNextUpdateTime(parent->simulationTime() + hc->getDeltaTime());

   PVParams * params = hc->parameters();


   //If not pvp file, open fileOfFileNames 
   if (hc->columnId()==0) {
      filenamestream = PV_fopen(inputPath, "r", false/*verifyWrites*/);
      if( filenamestream == NULL ) {
         pvError().printf("Movie::initialize error opening \"%s\": %s\n", inputPath, strerror(errno));
      }
   }

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
      if(getParent()->getCommunicator()->commRank()==0){
          //If checkpoint read is set, append, otherwise, clobber
          if(getParent()->getCheckpointReadFlag()){
             struct stat statbuf;
             if (PV_stat(timestampFilename.c_str(), &statbuf) != 0) {
                pvWarn().printf("%s: timestamp file \"%s\" unable to be found.  Creating new file.\n",
                      getDescription_c(), timestampFilename.c_str());
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
   ioParam_displayPeriod(ioFlag);
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

void Movie::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &echoFramePathnameFlag, false/*default value*/);
}

void Movie::ioParam_batchMethod(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "bySpecified");
   if(strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "bySpecified") == 0){
      //Correct
   }
   else{
      pvError() << "Movie layer " << name << " batchMethod not recognized. Options are \"byImage\", \"byMovie\", and \"bySpecified\"\n";
   }
}

void Movie::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamArray(ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &numStartFrame);
}

void Movie::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamArray(ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &numSkipFrame);
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

   if(parent->getCommunicator()->commRank()==0){
      framePath = (char**) malloc(parent->getNBatch() * sizeof(char*));
      assert(framePath);
      for(int b = 0; b < parent->getNBatch(); b++){
         framePath[b] = NULL;
      }
   }
   
   batchPos = (long*) malloc(parent->getNBatch() * sizeof(long));
   if(batchPos==NULL) {
      pvError().printf("%s: unable to allocate memory for batchPos (batch size %d): %s\n",
            getDescription_c(), parent->getNBatch(), strerror(errno));
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      batchPos[b] = 0L;
   }
   frameNumbers = (int*) calloc(parent->getNBatch(), sizeof(int));
   if (frameNumbers==NULL) {
      pvError().printf("%s: unable to allocate memory for frameNumbers (batch size %d): %s\n",
            getDescription_c(), parent->getNBatch(), strerror(errno));
   }

   //Calculate file positions for beginning of each frame
   numFrames = getNumFrames();
   pvInfo() << "File " << inputPath << " contains " << numFrames << " frames\n";

   startFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(startFrameIndex);
   skipFrameIndex = (int*)calloc(parent->getNBatch(), sizeof(int));
   assert(skipFrameIndex);

   int nbatch = parent->getNBatch();
   assert(batchMethod);

   if(strcmp(batchMethod, "byImage") == 0){
      //No skip here allowed
      if(numSkipFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"byImage\" sets skip_frame_index, do not specify.\n";
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
         pvError() << "Movie layer " << name << " batchMethod of \"byImage\" requires 0 or 1 start_frame_index values\n";
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
      int nbatchGlobal = parent->getNBatchGlobal();
      int commBatch = parent->commBatch();
      int numBatchPerProc = parent->numCommBatches();

      if(numStartFrame != nbatchGlobal && numStartFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatchGlobal << " start_frame_index values\n";
      }
      if(numSkipFrame != nbatchGlobal && numSkipFrame != 0){
         pvError() << "Movie layer " << name << " batchMethod of \"bySpecified\" requires 0 or " << nbatchGlobal << " skip_frame_index values\n";
      }

      //Each process grabs it's own set of start/skipFrameIndex from the parameters
      int startCommIdx = commBatch*numBatchPerProc;
      int endCommIdx = startCommIdx + nbatch;
      int b = 0;
      for(int pb = startCommIdx; pb < endCommIdx; pb++){ 
         if(numStartFrame == 0){
            //+1 for 1 indexed
            startFrameIndex[b] = 0;
         }
         else{
            startFrameIndex[b] = paramsStartFrameIndex[pb];
         }
         if(numSkipFrame == 0){
            skipFrameIndex[b] = 1;
         }
         else{
            skipFrameIndex[b] = paramsSkipFrameIndex[pb];
         }
         b++;
      }
      //Sanity check
      assert(b == nbatch);
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

   return status;
}

PVLayerLoc Movie::getImageLoc()
{
   return imageLoc;
   // imageLoc contains size information of the image file being loaded;
   // clayer->loc contains size information of the layer, which may
   // be smaller than the whole image.  To get information on the layer, use
   // getLayerLoc().  --pete 2011-07-10
}

double Movie::getDeltaUpdateTime(){
   //If jitter , update every timestep
   if( jitterFlag ){
      return parent->getDeltaTime();
   }
   return displayPeriod;
}

#ifdef OBSOLETE // Marked obsolete Aug 18, 2016. Handling the adaptive timestep has been moved to ColumnEnergyProbe.
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
#endif // OBSOLETE // Marked obsolete Aug 18, 2016. Handling the adaptive timestep has been moved to ColumnEnergyProbe.

int Movie::updateState(double time, double dt)
{
   updateImage(time, dt);
   return PV_SUCCESS;
}


//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int Movie::retrieveData(double timef, double dt, int batchIdx)
{
   int status = PV_SUCCESS;
   if(framePath[batchIdx]!= NULL) free(framePath[batchIdx]);
   if(timef==parent->getStartTime()){
      framePath[batchIdx] = strdup(getNextFileName(startFrameIndex[batchIdx]+1, batchIdx));
   }
   else{
      framePath[batchIdx] = strdup(getNextFileName(skipFrameIndex[batchIdx], batchIdx));
   }
   pvInfo() << "Reading frame " << framePath[batchIdx] << " into batch " << batchIdx << " at time " << timef << "\n";
   status = readImage(framePath[batchIdx]);
   if( status != PV_SUCCESS ) {
      pvError().printf("Movie %s: Error reading file \"%s\"\n", name, framePath[batchIdx]);
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

   Communicator * icComm = getParent()->getCommunicator();

   //TODO: Fix movie layer to take with batches. This is commented out for compile
   //if(!flipOnTimescaleError && (parent->getTimeScale() > 0 && parent->getTimeScale() < parent->getTimeScaleMin())){
   //   if (parent->icCommunicator()->commRank()==0) {
   //      pvWarn() << "timeScale of " << parent->getTimeScale() << " is less than timeScaleMin of " << parent->getTimeScaleMin() << ", Movie is keeping the same frame\n";
   //   }
   //}
   //else
   {
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
               pvError().printf("%s: Movie::updateState failed to write to timestamp file.\n", getDescription_c());
            }
            //Flush buffer
            fflush(timestampFile->fp);
         }
      }
   } // else-clause of if(!flipOnTimescaleError && (parent->getTimeScale() > 0 && parent->getTimeScale() < parent->getTimeScaleMin()))

   return true;
}

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

// advance by n_skip lines through file of filenames, always advancing at least one line
const char * Movie::getNextFileName(int n_skip, int batchIdx) {
   Communicator * icComm = getParent()->getCommunicator();
   assert(icComm->commRank() == 0);
   const char* outFilename = NULL;
   int numskip = n_skip < 1 ? 1 : n_skip;
   for (int i_skip = 0; i_skip < numskip; i_skip++){
      outFilename = advanceFileName(batchIdx);
   }
   if (echoFramePathnameFlag){
      pvInfo().printf("%s: t=%f, batch element %d: loading %s\n", getDescription_c(), parent->simulationTime(), batchIdx, outFilename);
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
   MPI_Bcast(&count, 1, MPI_INT, 0, parent->getCommunicator()->communicator());
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
         pvInfo().printf("Movie %s: EOF reached, rewinding file \"%s\"\n", name, inputPath);
         if (hasrewound) {
            pvError().printf("Movie %s: filenamestream \"%s\" does not have any non-blank lines.\n", name, filenamestream->name);
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
   }
   //Save batch position
   batchPos[batchIdx] = getPV_StreamFilepos(filenamestream);
   return inputfile;
}

#else // PV_USE_GDAL
Movie::Movie(const char * name, HyPerCol * hc) {
   if (hc->columnId()==0) {
      pvErrorNoExit().printf("Movie \"%s\": Movie class requires compiling with PV_USE_GDAL set\n", name);
   }
   MPI_Barrier(hc->getCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
Movie::Movie() {}
#endif // PV_USE_GDAL

} // ends namespace PV block
