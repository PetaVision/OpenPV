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

Movie::Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames) {
   initialize_base();
   initialize(name, hc, fileOfFileNames, DISPLAY_PERIOD);
}

Movie::Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod) {
   initialize_base();
   initialize(name, hc, fileOfFileNames, defaultDisplayPeriod);
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

   if (this->useParamsImage) { //Sets nextDisplayTime = simulationtime (i.e. effectively restarting)
      nextDisplayTime += parent->simulationTime();
   }

   InterColComm * icComm = parent->icCommunicator();
   int filenamesize = strlen(cpDir)+1+strlen(name)+18;
   // The +1 is for the slash between cpDir and name; the +18 needs to be large enough to hold the suffix _PatternState.{bin,txt} plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed = snprintf(filename, filenamesize, "%s/%s_MovieState.bin", cpDir, name);
   assert(chars_needed < filenamesize);
   if( icComm->commRank() == 0 ) {
      //Only read timestamp file pos if 
      //1. There exists a timestampFile
      //2. There exists a MovieState.bin (Run being checkpointed from could have not been printing out timestamp files
      PV_Stream * pvstream = PV_fopen(filename, "r");
      if (timestampFile && pvstream){
         long timestampFilePos = 0L;
         status |= PV_fread(&timestampFilePos, sizeof(long), 1, pvstream);
         if (PV_fseek(timestampFile, timestampFilePos, SEEK_SET) != 0) {
            fprintf(stderr, "MovieLayer::checkpointRead error: unable to recover initial file position in timestamp file for layer %s\n", name);
            abort();
         }

         PV_fclose(pvstream);
      }
   }

   while (parent->simulationTime() >= nextDisplayTime) {
      nextDisplayTime += displayPeriod;
      //Follow dispPeriod for updating frame numbers and file names
      updateFrameNum(skipFrameIndex);
      if(!readPvpFile){
         if (filename != NULL) free(filename);
         filename = strdup(getNextFileName(skipFrameIndex));
         assert(filename != NULL);
      }
   }
   return status;
}

int Movie::checkpointWrite(const char * cpDir){
   int status = Image::checkpointWrite(cpDir);
   //Only do a checkpoint write if there exists a timestamp file
   if(timestampFile){
      InterColComm * icComm = parent->icCommunicator();
      int filenamesize = strlen(cpDir)+1+strlen(name)+18;
      // The +1 is for the slash between cpDir and name; the +18 needs to be large enough to hold the suffix _PatternState.{bin,txt} plus the null terminator
      char * filename = (char *) malloc( filenamesize*sizeof(char) );
      assert(filename != NULL);
      sprintf(filename, "%s/%s_MovieState.bin", cpDir, name);
      if( icComm->commRank() == 0 ) {
         //Get the file position of the timestamp file
         long timestampFilePos = getPV_StreamFilepos(timestampFile);
         PV_Stream * pvstream = PV_fopen(filename, "w");
         if(pvstream != NULL){
            status |= PV_fwrite(&timestampFilePos, sizeof(long), 1, pvstream);
            PV_fclose(pvstream);
         } 
         else{
            fprintf(stderr, "Unable to write to \"%s\"\n", filename);
            status = PV_FAILURE;
         }
         sprintf(filename, "%s/%s_MovieState.txt", cpDir, name);
         pvstream = PV_fopen(filename, "w");
         if(pvstream != NULL){
            fprintf(pvstream->fp, "timestampFilePos = %ld", timestampFilePos);
            PV_fclose(pvstream);
         }
         else{
            fprintf(stderr, "Unable to write to \"%s\"\n", filename);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

//
/*
 * Notes:
 * - writeImages, offsetX, offsetY are initialized by Image::initialize()
 */
int Movie::initialize(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod) {
   displayPeriod = defaultDisplayPeriod; // Will be replaced with params value when setParams is called.
   int status = Image::initialize(name, hc, NULL);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Image::initialize failed on Movie layer \"%s\".  Exiting.\n", name);
      exit(PV_FAILURE);
   }

   if (fileOfFileNames != NULL) {
      this->fileOfFileNames = strdup(fileOfFileNames);
      if (this->fileOfFileNames==NULL) {
         fprintf(stderr, "Movie::initialize error in layer \"%s\": unable to copy fileOfFileNames: %s\n", name, strerror(errno));
      }
   }

   PVParams * params = hc->parameters();

   assert(!params->presentAndNotBeenRead(name, "displayPeriod"));
   nextDisplayTime = hc->simulationTime() + displayPeriod + hc->getDeltaTime();

   assert(!params->presentAndNotBeenRead(name, "randomMovie")); // randomMovie should have been set in setParams
   if (randomMovie) return status; // Nothing else to be done until data buffer is allocated, in allocateDataStructures

   assert(!params->presentAndNotBeenRead(name, "readPvpFile")); // readPvpFile should have been set in setParams

   //If not pvp file, open fileOfFileNames 
   if( getParent()->icCommunicator()->commRank()==0 && !readPvpFile) {
      filenamestream = PV_fopen(fileOfFileNames, "r");
      if( filenamestream == NULL ) {
         fprintf(stderr, "Movie::initialize error opening \"%s\": %s\n", fileOfFileNames, strerror(errno));
         abort();
      }
   }

   if (!randomMovie) {
      if (startFrameIndex <= 1){
         frameNumber = 0;
      }
      else{
         frameNumber = startFrameIndex - 1;
      }
      if(readPvpFile){
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
         if(numParams != NUM_PAR_BYTE_PARAMS || params[INDEX_FILE_TYPE] != PVP_NONSPIKING_ACT_FILE_TYPE) {
            fprintf(stderr, "Movie layer \"%s\" error: file \"%s\" is not a nonspiking-activity pvp file.\n", name, filename);
            abort();
         }
         numFrames = params[INDEX_NBANDS];
      }
      else{
         // echoFramePathnameFlag = params->value(name,"echoFramePathnameFlag", false);
         filename = strdup(getNextFileName(startFrameIndex));
         assert(filename != NULL);
      }
   }

   // getImageInfo/constrainOffsets/readImage calls moved to Movie::allocateDataStructures

   // set output path for movie frames
   if(writeImages){
      // if ( params->stringPresent(name, "movieOutputPath") ) {
      //    movieOutputPath = strdup(params->stringValue(name, "movieOutputPath"));
      //    assert(movieOutputPath != NULL);
      // }
      // else {
      //    movieOutputPath = strdup( hc->getOutputPath());
      //    assert(movieOutputPath != NULL);
      //    printf("movieOutputPath is not specified in params file.\n"
      //          "movieOutputPath set to default \"%s\"\n",movieOutputPath);
      // }
      status = parent->ensureDirExists(movieOutputPath);
   }

   if(writeFrameToTimestamp){
      std::string timestampFilename = std::string(strdup(parent->getOutputPath()));
      timestampFilename += "/timestamps/";
      parent->ensureDirExists(timestampFilename.c_str());
      timestampFilename += name;
      timestampFilename += ".txt";
      if(getParent()->icCommunicator()->commRank()==0){
          //If checkpoint read is set, append, otherwise, clobber
          if(getParent()->getCheckpointReadFlag()){
             timestampFile = PV::PV_fopen(timestampFilename.c_str(), "r+");
          }
          else{
             timestampFile = PV::PV_fopen(timestampFilename.c_str(), "w");
          }
          assert(timestampFile);
      }
   }
   return PV_SUCCESS;
}

int Movie::setParams(PVParams * params) {
   int status = Image::setParams(params);
   displayPeriod = params->value(name,"displayPeriod", displayPeriod);
   randomMovie = (int) params->value(name,"randomMovie",0);
   if (randomMovie) {
      randomMovieProb   = params->value(name,"randomMovieProb", 0.05);  // 100 Hz
   }
   else {
      readPvpFile = (bool)params->value(name, "readPvpFile", 0);
      if (!readPvpFile) {
         echoFramePathnameFlag = params->value(name,"echoFramePathnameFlag", false);
      }
      startFrameIndex = params->value(name,"start_frame_index", 0);
      skipFrameIndex = params->value(name,"skip_frame_index", 0);
      autoResizeFlag = (bool)params->value(name,"autoResizeFlag",false);
   }
   assert(!params->presentAndNotBeenRead(name, "writeImages"));
   if(writeImages){
      if ( params->stringPresent(name, "movieOutputPath") ) {
         movieOutputPath = strdup(params->stringValue(name, "movieOutputPath"));
         assert(movieOutputPath != NULL);
      }
      else {
         movieOutputPath = strdup( parent->getOutputPath());
         assert(movieOutputPath != NULL);
         printf("movieOutputPath is not specified in params file.\n"
               "movieOutputPath set to default \"%s\"\n",movieOutputPath);
      }
   }
   writeFrameToTimestamp = (bool) params->value(name, "writeFrameToTimestamp", writeFrameToTimestamp);
   return status;
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
      // skip to start_frame_index if provided
      // int start_frame_index = params->value(name,"start_frame_index", 0);
      // skipFrameIndex = params->value(name,"skip_frame_index", 0);

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

bool Movie::needUpdate(double time, double dt){
   bool needNewImage = false;
   //Always update on first timestep
   if (time <= parent->getStartTime()){
       needNewImage = true;
   }
   if(randomMovie){
      needNewImage = true;
   }
   if( jitterFlag ) {
      needNewImage = true;;
   } // jitterFlag
   if (time >= nextDisplayTime) {
      needNewImage = true;
   } // time >= nextDisplayTime


   //if(time >= nextDisplayTime || updateThisTimestep) {
   //} // time >= nextDisplayTime

   return needNewImage;
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
      updateFrameNum(skipFrameIndex);
      if(!readPvpFile){
         if (filename != NULL) free(filename);
         filename = strdup(getNextFileName(skipFrameIndex));
         assert(filename != NULL);
      }
      if(writePosition && icComm->commRank()==0){
         fprintf(fp_pos->fp,"%f %s: \n",time,filename);
      }
      while (time >= nextDisplayTime) {
         nextDisplayTime += displayPeriod;
      }

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
             outStrStream << frameNumber << "," << lastUpdateTime << "," << filename << "\n";
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

//This function takes care of rewinding for pvp files
void Movie::updateFrameNum(int n_skip){
   InterColComm * icComm = getParent()->icCommunicator();
   for(int i_skip = 0; i_skip < n_skip; i_skip++){
      frameNumber += 1;
      //numFrames only set if pvp file
      if(readPvpFile){
         if(frameNumber >= numFrames){
            if(icComm->commRank()==0){
               fprintf(stderr, "Movie %s: EOF reached, rewinding file \"%s\"\n", name, fileOfFileNames);
            }
            frameNumber = 0;
         }
      }
   }
}

// skip n_skip lines before reading next frame
const char * Movie::getNextFileName(int n_skip)
{
   for (int i_skip = 0; i_skip < n_skip-1; i_skip++){
      getNextFileName();
   }
   return getNextFileName();
}

//This function takes care of rewinding for frame files
const char * Movie::getNextFileName()
{
   InterColComm * icComm = getParent()->icCommunicator();
   if( icComm->commRank()==0 ) {
      int c;
      size_t maxlen = PV_PATH_MAX;

      //TODO: add recovery procedure to handle case where access to file is temporarily unavailable
      // use stat to verify status of filepointer, keep running tally of current frame index so that file can be reopened and advanced to current frame


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
      if (echoFramePathnameFlag){
         fprintf(stderr, "%s\n", inputfile);
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(inputfile, PV_PATH_MAX, MPI_CHAR, 0, icComm->communicator());
#endif // PV_USE_MPI
   return inputfile;
}

// bool Movie::getNewImageFlag(){
//    return newImageFlag;
// }

const char * Movie::getCurrentImage(){
   return inputfile;
}

}
