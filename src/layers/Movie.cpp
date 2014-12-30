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

Movie::Movie() {
   initialize_base();
}

Movie::Movie(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int Movie::initialize_base() {
   movieOutputPath = NULL;
   skipFrameIndex = 0;
   echoFramePathnameFlag = false;
   filenamestream = NULL;
   displayPeriod = DISPLAY_PERIOD;
   readPvpFile = false;
   fileOfFileNames = NULL;
   frameNumber = 0;
   numFrames = 0;
   writeFrameToTimestamp = true;
   timestampFile = NULL;
   flipOnTimescaleError = true;
   resetToStartOnLoop = false;
   //updateThisTimestep = false;
   // newImageFlag = false;
   return PV_SUCCESS;
}

int Movie::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = Image::readStateFromCheckpoint(cpDir, timeptr);
   status = readFrameNumStateFromCheckpoint(cpDir);
   return status;
}

int Movie::readFrameNumStateFromCheckpoint(const char * cpDir) {
   int status = PV_SUCCESS;
   parent->readScalarFromFile(cpDir, getName(), "FrameNumState", &frameNumber, frameNumber);

   if (!readPvpFile) {
      int startFrame = frameNumber;
      if (parent->columnId()==0) {
         PV_fseek(filenamestream, 0L, SEEK_SET);
         frameNumber = 0;
      }
      if (filename != NULL) free(filename);
      filename = strdup(getNextFileName(startFrame)); // getNextFileName() will increment frameNumber by startFrame;
      if (parent->columnId()==0) assert(frameNumber==startFrame);
      if (parent->columnId()==0) {
         printf("%s \"%s\" checkpointRead set frameNumber to %d and filename to \"%s\"\n",
               parent->parameters()->groupKeywordFromName(name), name, frameNumber, filename);
      }
   }
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

   parent->writeScalarToFile(cpDir, getName(), "FrameNumState", frameNumber);

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
int Movie::initialize(const char * name, HyPerCol * hc) {
   int status = Image::initialize(name, hc);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Image::initialize failed on Movie layer \"%s\".  Exiting.\n", name);
      exit(PV_FAILURE);
   }

   //Update on first timestep
   setNextUpdateTime(parent->simulationTime() + hc->getDeltaTime());

   PVParams * params = hc->parameters();

   assert(!params->presentAndNotBeenRead(name, "randomMovie")); // randomMovie should have been set in ioParams
   if (randomMovie) return status; // Nothing else to be done until data buffer is allocated, in allocateDataStructures


   //If not pvp file, open fileOfFileNames 
   assert(!params->presentAndNotBeenRead(name, "readPvpFile")); // readPvpFile should have been set in ioParams
   if( getParent()->icCommunicator()->commRank()==0 && !readPvpFile) {
      filenamestream = PV_fopen(fileOfFileNames, "r", false/*verifyWrites*/);
      if( filenamestream == NULL ) {
         fprintf(stderr, "Movie::initialize error opening \"%s\": %s\n", fileOfFileNames, strerror(errno));
         abort();
      }
   }

   if (!randomMovie) {
      if(readPvpFile){
         if (startFrameIndex <= 1){
            frameNumber = 0;
         }
         else{
            frameNumber = startFrameIndex - 1;
         }
         //Set filename as param
         filename = strdup(fileOfFileNames);
         assert(filename != NULL);
         //Grab number of frames from header
         PV_Stream * pvstream = NULL;
         if (getParent()->icCommunicator()->commRank()==0) {
            pvstream = PV::PV_fopen(filename, "rb", false/*verifyWrites*/);
         }
         int numParams = NUM_PAR_BYTE_PARAMS;
         int params[numParams];
         pvp_read_header(pvstream, getParent()->icCommunicator(), params, &numParams);
         PV::PV_fclose(pvstream); pvstream = NULL;
         numFrames = params[INDEX_NBANDS];
      }
      else{
         //frameNumber handled here
         filename = strdup(getNextFileName(startFrameIndex));
         assert(filename != NULL);
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
      if(getParent()->icCommunicator()->commRank()==0){
          //If checkpoint read is set, append, otherwise, clobber
          if(getParent()->getCheckpointReadFlag()){
             struct stat statbuf;
             if (PV_stat(timestampFilename.c_str(), &statbuf) != 0) {
                fprintf(stderr, "%s \"%s\" warning: timestamp file \"%s\" unable to be found.  Creating new file.\n",
                      parent->parameters()->groupKeywordFromName(name), name, timestampFilename.c_str());
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
   ioParam_imageListPath(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_randomMovie(ioFlag);
   ioParam_randomMovieProb(ioFlag);
   ioParam_readPvpFile(ioFlag);
   ioParam_echoFramePathnameFlag(ioFlag);
   ioParam_start_frame_index(ioFlag);
   ioParam_skip_frame_index(ioFlag);
   ioParam_movieOutputPath(ioFlag);
   ioParam_writeFrameToTimestamp(ioFlag);
   ioParam_flipOnTimescaleError(ioFlag);
   ioParam_resetToStartOnLoop(ioFlag);
   return status;
}

void Movie::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      filename = NULL;
      parent->parameters()->handleUnnecessaryStringParameter(name, "imageList");
   }
}

void Movie::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   // Image uses frameNumber to pick the frame of a pvp file, but
   // Movie uses start_frame_index to pick the starting frame.
   if (ioFlag == PARAMS_IO_READ) {
      filename = NULL;
      parent->parameters()->handleUnnecessaryParameter(name, "frameNumber");
   }
}

void Movie::ioParam_imageListPath(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "imageListPath", &fileOfFileNames);
}

void Movie::ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "flipOnTimescaleError", &flipOnTimescaleError, flipOnTimescaleError);
}

void Movie::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, displayPeriod);
}

void Movie::ioParam_randomMovie(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "randomMovie", &randomMovie, 0/*default value*/);
}

void Movie::ioParam_randomMovieProb(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   if (randomMovie) {
      parent->ioParamValue(ioFlag, name, "randomMovieProb", &randomMovieProb, 0.05f);
   }
}

void Movie::ioParam_readPvpFile(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   if (!randomMovie) {
      parent->ioParamValue(ioFlag, name, "readPvpFile", &readPvpFile, false/*default value*/);
   }
}

void Movie::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   if (!randomMovie) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "readPvpFile"));
      if (!readPvpFile) {
         parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &echoFramePathnameFlag, false/*default value*/);
      }
   }
}

void Movie::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   if (!randomMovie) {
      parent->ioParamValue(ioFlag, name, "start_frame_index", &startFrameIndex, 0/*default value*/);
   }
}

void Movie::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "randomMovie"));
   if (!randomMovie) {
      parent->ioParamValue(ioFlag, name, "skip_frame_index", &skipFrameIndex, 0/*default value*/);
   }
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

Movie::~Movie()
{
   if (imageData != NULL) {
      delete imageData;
      imageData = NULL;
   }
   if (getParent()->icCommunicator()->commRank()==0 && filenamestream != NULL && filenamestream->isfile) {
      PV_fclose(filenamestream);
   }
   free(fileOfFileNames); fileOfFileNames = NULL;
   if (getParent()->icCommunicator()->commRank()==0 && timestampFile != NULL && timestampFile->isfile) {
       PV_fclose(timestampFile);
   }
   free(movieOutputPath);
}

int Movie::allocateDataStructures() {
   int status = Image::allocateDataStructures();

   if (!randomMovie) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "start_frame_index"));
      assert(!parent->parameters()->presentAndNotBeenRead(name, "skip_frame_index"));

      assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      //if (!autoResizeFlag){
      //   constrainOffsets();  // ensure that offsets keep loc within image bounds
      //}

      // status = readImage(filename, getOffsetX(), getOffsetY()); // readImage already called by Image::allocateDataStructures(), above
      assert(status == PV_SUCCESS);
   }
   else {
      if (randState==NULL) {
         initRandState();
      }
      status = randomFrame();
   }

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
   if(randomMovie){
      return parent->getDeltaTime();
   }
   return displayPeriod;
}

  // ensure that timeScale == 1 if new frame being loaded on NEXT time step
  
double Movie::calcTimeScale(){
    if(needUpdate(parent->simulationTime() + parent->getDeltaTime(), parent->getDeltaTime())){
      return parent->getTimeScaleMin(); 
    }
    else{
      return HyPerLayer::calcTimeScale();
    }
  }

int Movie::updateState(double time, double dt)
{
   updateImage(time, dt);
   return PV_SUCCESS;
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
   if(randomMovie){
      randomFrame();
      //Moved to updateStateWrapper
      //lastUpdateTime = time;
   } else {

      if(!flipOnTimescaleError && (parent->getTimeScale() > 0 && parent->getTimeScale() < parent->getTimeScaleMin())){
         if (parent->icCommunicator()->commRank()==0) {
            std::cout << "timeScale of " << parent->getTimeScale() << " is less than timeScaleMin of " << parent->getTimeScaleMin() << ", Movie is keeping the same frame\n";
         }
      }
      else{
         if(!readPvpFile){
            //Only do this if it's not the first update timestep
            //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
            //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).  
            if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
               //If the error is too high, keep the same frame
               if (filename != NULL) free(filename);
               filename = strdup(getNextFileName(skipFrameIndex));
            }
            assert(filename != NULL);
         }
         else{
            //Only do this if it's not the first update timestep
            //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
            //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).  
            if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
               updateFrameNum(skipFrameIndex);
            }
         }
      }
      if(writePosition && icComm->commRank()==0){
         fprintf(fp_pos->fp,"%f %s: \n",time,filename);
      }
      //nextDisplayTime removed, now using nextUpdateTime in HyPerLayer
      //while (time >= nextDisplayTime) {
      //   nextDisplayTime += displayPeriod;
      //}
      //Set frame number (member variable in Image)
      int status = readImage(filename, this->offsets[0], this->offsets[1], this->offsetAnchor);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "Movie %s: Error reading file \"%s\"\n", name, filename);
         abort();
      }
      //Write to timestamp file here when updated
      if( icComm->commRank()==0 ) {
          //Only write if the parameter is set
          if(timestampFile){
             std::ostringstream outStrStream;
             outStrStream.precision(15);
             outStrStream << frameNumber << "," << time << "," << filename << "\n";
             size_t len = outStrStream.str().length();
             int status = PV_fwrite(outStrStream.str().c_str(), sizeof(char), len, timestampFile)==len ? PV_SUCCESS : PV_FAILURE;
             if (status != PV_SUCCESS) {
                fprintf(stderr, "%s \"%s\" error: Movie::updateState failed to write to timestamp file.\n", parent->parameters()->groupKeywordFromName(name), name);
                exit(EXIT_FAILURE);
             }
             //Flush buffer
             fflush(timestampFile->fp);
          }
      }
   } // randomMovie

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
      snprintf(basicFilename, PV_PATH_MAX, "%s/%s_%.2f.%s", movieOutputPath, name, timed, writeImagesExtension);
      write(basicFilename);
   }

   int status = PV_SUCCESS;
   status = HyPerLayer::outputState(timed, last);

   return status;
}

int Movie::copyReducedImagePortion()
{
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nx0 = imageLoc.nx;
   const int ny0 = imageLoc.ny;

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   const int i0 = nx/2 - nx0/2;
   const int j0 = ny/2 - ny0/2;

   int ii = 0;
   for (int j = j0; j < j0+ny0; j++) {
      for (int i = i0; i < i0+nx0; i++) {
         imageData[ii++] = data[i+nx*j];
      }
   }

   return 0;
}

/**
 * This creates a random image patch (frame) that is used to perform a reverse correlation analysis
 * as the input signal propagates up the visual system's hierarchy.
 * NOTE: Check Image::toGrayScale() method which was the inspiration for this routine
 */
int Movie::randomFrame()
{
   assert(randomMovie); // randomMovieProb was set only if randomMovie is true
   for (int kex = 0; kex < clayer->numExtended; kex++) {
      double p = randState->uniformRandom();
      data[kex] = (p < randomMovieProb) ? 1: 0;
   }
   return 0;
}

/**
 * A function only called if readPvpFile is set
 * Will update frameNumber
 */
//This function takes care of rewinding for pvp files
int Movie::updateFrameNum(int n_skip){
   assert(readPvpFile);
   InterColComm * icComm = getParent()->icCommunicator();
   int numskip = n_skip < 1 ? 1 : n_skip;
   for(int i_skip = 0; i_skip < numskip; i_skip++){
      int status = updateFrameNum();
      if(status == PV_BREAK){
         break;
      }
   }
   return PV_SUCCESS;
}

int Movie::updateFrameNum() {
   frameNumber += 1;
   //numFrames only set if pvp file
   if(frameNumber >= numFrames){
      if(parent->columnId()==0){
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
      }
      if(resetToStartOnLoop){
         frameNumber = startFrameIndex-1;
         return PV_BREAK;
      }
      else{
         frameNumber = 0;
      }
   }
   return PV_SUCCESS;
}

// advance by n_skip lines through file of filenames, always advancing at least one line
const char * Movie::getNextFileName(int n_skip) {
   InterColComm * icComm = getParent()->icCommunicator();
   if (icComm->commRank()==0) {
      int numskip = n_skip < 1 ? 1 : n_skip;
      for (int i_skip = 0; i_skip < numskip; i_skip++){
         advanceFileName();
      }
      if (echoFramePathnameFlag){
         printf("%f: %s\n", parent->simulationTime(), inputfile);
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(inputfile, PV_PATH_MAX, MPI_CHAR, 0, icComm->communicator());
#endif // PV_USE_MPI
   return inputfile;
}

//This function takes care of rewinding for frame files
const char * Movie::advanceFileName() {
   // IMPORTANT!! This function should only be called by getNextFileName(int), and only by the root process
   assert(parent->columnId()==0);
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
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
         if (hasrewound) {
            fprintf(stderr, "Movie %s: filenamestream \"%s\" does not have any non-blank lines.\n", name, filenamestream->name);
            exit(EXIT_FAILURE);
         }
         hasrewound = true;
         frameNumber = 0;
         reset = true;
      }
      else {
         ungetc(c, filenamestream->fp);
      }

      //Always do at least once
      do{
         char * path = fgets(inputfile, maxlen, filenamestream->fp);
         if (path != NULL) {
            filenamestream->filepos += strlen(path);
            frameNumber++;
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
                  lineisblank = false;
                  break;
               }
            }
         }
      }while(reset && frameNumber < startFrameIndex+1);
      assert(inputfile && strlen(inputfile)>(size_t) 0);
      char * expandedpath = expandLeadingTilde(inputfile);
      if (strlen(expandedpath)>=PV_PATH_MAX) {
         fprintf(stderr, "Movie \"%s\": input line \"%s\" from imageListPath is too long.\n", name, expandedpath);
         exit(EXIT_FAILURE);
      }
      strncpy(inputfile, expandedpath, PV_PATH_MAX);
      free(expandedpath);
   }
   return inputfile;
}

const char * Movie::getCurrentImage(){
   return inputfile;
}

}
