/*
 * Movie.cpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#include "Movie.hpp"
#include "../io/imageio.hpp"
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
   //updateThisTimestep = false;
   // newImageFlag = false;
   return PV_SUCCESS;
}

int Movie::checkpointRead(const char * cpDir, double * timef){
   int status = Image::checkpointRead(cpDir, timef);

   //nextUpdateTime now handeled in HyPerLayer checkpoint read/write
   //if (this->useParamsImage) { //Sets nextDisplayTime = simulationtime (i.e. effectively restarting)
   //   nextDisplayTime += parent->simulationTime();
   //}

   if (writeFrameToTimestamp && timestampFile) {
      long timestampFilePos = 0L;
      parent->readScalarFromFile(cpDir, getName(), "TimestampState", &timestampFilePos, timestampFilePos);
      if (PV_fseek(timestampFile, timestampFilePos, SEEK_SET) != 0) {
         fprintf(stderr, "MovieLayer::checkpointRead error: unable to recover initial file position in timestamp file for layer %s: %s\n", name, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
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
      filenamestream = PV_fopen(fileOfFileNames, "r");
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
            pvstream = PV::PV_fopen(filename, "rb");
         }
         int numParams = NUM_PAR_BYTE_PARAMS;
         int params[numParams];
         pvp_read_header(pvstream, getParent()->icCommunicator(), params, &numParams);
         PV::PV_fclose(pvstream); pvstream = NULL;
         numFrames = params[INDEX_NBANDS];
      }
      else{
         //frameNumber Handeled here
         filename = strdup(getNextFileName(startFrameIndex));
         //if(startFrameIndex <= 1){
         //   //Movie is going to update on timestep 1, but we want it to reread the first frame here, so reset the filenamestream back to 0 in initialize
         //   if( parent->icCommunicator()->commRank()==0 ) {
         //      PV_fseek(filenamestream, 0L, SEEK_SET);
         //   }
         //}
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
                timestampFile = PV::PV_fopen(timestampFilename.c_str(), "w");
             }
             else {
                timestampFile = PV::PV_fopen(timestampFilename.c_str(), "r+");
             }
          }
          else{
             timestampFile = PV::PV_fopen(timestampFilename.c_str(), "w");
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

}

int Movie::allocateDataStructures() {
   int status = Image::allocateDataStructures();

   if (!randomMovie) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "start_frame_index"));
      assert(!parent->parameters()->presentAndNotBeenRead(name, "skip_frame_index"));

      // get size info from image so that data buffer can be allocated
      GDALColorInterp * colorbandtypes = NULL;
      status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if(status != 0) {
         fprintf(stderr, "Movie: Unable to get image info for \"%s\"\n", filename);
         abort();
      }

      assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      if (!autoResizeFlag){
         constrainOffsets();  // ensure that offsets keep loc within image bounds
      }

      status = readImage(filename, getOffsetX(), getOffsetY(), colorbandtypes);
      assert(status == PV_SUCCESS);

      free(colorbandtypes); colorbandtypes = NULL;
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

//Need update is now complety handeled in HyPerLayer based on getDeltaUpdateTime
//bool Movie::needUpdate(double time, double dt){
//   bool needNewImage = false;
//   //Always update on first timestep
//   if (time <= parent->getStartTime()){
//       needNewImage = true;
//   }
//   if(randomMovie){
//      needNewImage = true;
//   }
//   if( jitterFlag ) {
//      needNewImage = true;;
//   } // jitterFlag
//
//   //This is now handeled in HyPerLayer
//   if (time >= nextDisplayTime) {
//      needNewImage = true;
//   } // time >= nextDisplayTime
//
//
//   //if(time >= nextDisplayTime || updateThisTimestep) {
//   //} // time >= nextDisplayTime
//
//   return needNewImage;
//}

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
      if(!readPvpFile){
         //Only do this if it's not the first update timestep
         //std::cout << "time: " << time << " startTime: " << parent->getStartTime() << " dt: " << dt << "\n";
         if(fabs(time - (parent->getStartTime() + dt)) > (dt/2)){
            if (filename != NULL) free(filename);
            filename = strdup(getNextFileName(skipFrameIndex));
            assert(filename != NULL);
         }
      }
      else{
         updateFrameNum(skipFrameIndex);
      }
      if(writePosition && icComm->commRank()==0){
         fprintf(fp_pos->fp,"%f %s: \n",time,filename);
      }
      //nextDisplayTime removed, now using nextUpdateTime in HyPerLayer
      //while (time >= nextDisplayTime) {
      //   nextDisplayTime += displayPeriod;
      //}

      GDALColorInterp * colorbandtypes = NULL;
      int status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "Movie %s: Error getting image info \"%s\"\n", name, filename);
         abort();
      }
      //Set frame number (member variable in Image)
      if( status == PV_SUCCESS ) status = readImage(filename, getOffsetX(), getOffsetY(), colorbandtypes);
      free(colorbandtypes); colorbandtypes = NULL;
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
             PV_fwrite(outStrStream.str().c_str(), sizeof(char), outStrStream.str().length(), timestampFile); 
             //Flush buffer
             fflush(timestampFile->fp);
              //fprintf(timestampFile->fp, "%d,%lf, %s\n",frameNumber, lastUpdateTime, filename);
              //fflush(timestampFile->fp);
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
      snprintf(basicFilename, PV_PATH_MAX, "%s/Movie_%.2f.%s", movieOutputPath, timed, writeImagesExtension);
      write(basicFilename);
   }

   int status = PV_SUCCESS;
   if (randomMovie != 0) {
      status = writeActivitySparse(timed, false/*includeValues*/);
   }
   else {
      status = HyPerLayer::outputState(timed, last);
   }

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
void Movie::updateFrameNum(int n_skip){
   assert(readPvpFile);
   InterColComm * icComm = getParent()->icCommunicator();
   for(int i_skip = 0; i_skip < n_skip-1; i_skip++){
      updateFrameNum();
   }
   updateFrameNum();
}

void Movie::updateFrameNum() {
   frameNumber += 1;
   //numFrames only set if pvp file
   if(frameNumber >= numFrames){
      if(parent->columnId()==0){
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
      }
      frameNumber = 0;
   }
}

// advance by n_skip lines through file of filenames, always advancing at least one line
const char * Movie::getNextFileName(int n_skip) {
   InterColComm * icComm = getParent()->icCommunicator();
   if (icComm->commRank()==0) {
      for (int i_skip = 0; i_skip < n_skip-1; i_skip++){
         advanceFileName();
      }
      advanceFileName();
      if (echoFramePathnameFlag){
         printf("%s\n", inputfile);
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

   // Ignore blank lines
   bool lineisblank = true;
   while(lineisblank) {
      // if at end of file (EOF), rewind
      if ((c = fgetc(filenamestream->fp)) == EOF) {
         PV_fseek(filenamestream, 0L, SEEK_SET);
         fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
         frameNumber = 0;
      }
      else {
         ungetc(c, filenamestream->fp);
      }

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
   }
   return inputfile;
}

const char * Movie::getCurrentImage(){
   return inputfile;
}

}
