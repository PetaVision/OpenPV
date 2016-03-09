/*
 * Movie.hpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#ifndef MOVIE_HPP_
#define MOVIE_HPP_

#include "Image.hpp"
#include <cMakeHeader.h>
#include <sstream>

namespace PV {

class Movie: public PV::Image {

#ifdef PV_USE_GDAL

public:
   Movie(const char * name, HyPerCol * hc);
   virtual ~Movie();

   virtual int allocateDataStructures();

   virtual pvdata_t * getImageBuffer();
   virtual PVLayerLoc getImageLoc();

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual int outputState(double time, bool last=false);
   //virtual bool needUpdate(double time, double dt);
   virtual double getDeltaUpdateTime();
   virtual double calcTimeScale(int batchIdx);
   //virtual int updateStateWrapper(double time, double dt);
   virtual int updateState(double time, double dt);
   virtual bool updateImage(double time, double dt);
   // bool        getNewImageFlag();
   //const char * getCurrentImage();
   int  randomFrame();
   //int getFrameNumber(){return frameNumber;}
   const char* getFilename(int batchIdx){return framePath[batchIdx];}
   const char* getBatchMethod(){return batchMethod;}

protected:
   Movie();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the Movie class
    * @name Movie Parameters
    * @{
    */

   /**
    * @brief displayPeriod: the amount of time each image is displayed before switching to the next image.
    * The units of displayPeriod are the same as the units of the HyPerCol's dt parameter.
    */
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief echoFramePathnameFlag: if true, print the filename to the screen when a new image file is loaded.
    */
   virtual void ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief start_frame_index: Initialize the layer with the given frame.
    * @details start_frame_index=0 means the first line of the imageListPath if a text file,
    * or the initial frame if imageListPath is a .pvp file.
    */
   virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);

   /**
    * @brief skip_frame_index: If skip_frame_index=1, go to the next frame at the end of the display period.
    * If skip_frame_index=2, skip the next frame and go to the second frame after the frame that just expired,
    * and so on.  If skip_frame_index is less than one, it behaves the same as skip_frame_index=1.
    */
   virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);

   /**
    * @brief movieOutputPath:
    */
   virtual void ioParam_movieOutputPath(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeFrameToTimestamp: if true, then every time the frame is updated, it writes the frame number, the time and the image filename
    *  to a file.  The file is placed in a directory "timestamps" in the outputPath directory, and the filename is the layer name appended with ".txt".
    */
   virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);

   /**
    * @brief flipOnTimescaleError: determines whether to change images at the end of the display period if the HyPerCol's timescale is less than the HyPerCol's timeScaleMin (defaults to true)
    */
   virtual void ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag);

   /**
    * @brief resetToStartOnLoop: If false, then when the end of file for the imageListPath file is reached, it rewinds to the beginning of the file.
    * If true, it resets to the location given by start_frame_index.
    */
   virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);

   /**
    * @brief batchMethod: Specifies how to split the file for batches.
    * byImage: Each batch skips nbatch, and starts staggered from the same part of the file 
    * byMovie: Each batch skips 1, and starts at numFrames/numBatch part of the file 
    * bySpecified: User specified start_frame_index and skip_frame_index, one for each batch
    */
   virtual void ioParam_batchMethod(enum ParamsIOFlag ioFlag);

   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /** @} */

   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readFrameNumStateFromCheckpoint(const char * cpDir);

   virtual int retrieveData(double timef, double dt);
   //bool readPvpFile;
   const char * getNextFileName(int n_skip, int batchIdx);
   int updateFrameNum(int n_skip);
   PV_Stream * timestampFile;


private:
   int initialize_base();
   //int copyReducedImagePortion();
   int updateFrameNum();
   int getNumFrames();
   const char * advanceFileName(int batchIdx);

   bool resetToStartOnLoop;

   double displayPeriod;   // length of time a frame is displayed
   //double nextDisplayTime; // time of next frame; now handled by HyPerLayer nextUpdateTime

#ifdef OBSOLETE // randomMovie was commented out of Movie.cpp on Jul 22, 2015.
   int randomMovie;       // these are used for performing a reverse correlation analysis
   float randomMovieProb;
#endif // OBSOLETE // randomMovie was commented out of Movie.cpp on Jul 22, 2015.

   bool echoFramePathnameFlag; // if true, echo the frame pathname to stdout
   // bool newImageFlag; // true when a new image was presented this timestep;

   int* startFrameIndex;
   int* skipFrameIndex;
   int* paramsStartFrameIndex;
   int* paramsSkipFrameIndex;
   int numStartFrame;
   int numSkipFrame;

   char inputfile[PV_PATH_MAX];  // current input file name
   char * movieOutputPath;  // path to output file directory for movie frames

   int numFrames; //Number of frames
   char ** framePath;
   char * batchMethod;

   PV_Stream * filenamestream;
   
   int* frameNumbers;

   bool writeFrameToTimestamp;

   bool flipOnTimescaleError;
   long * batchPos;
   bool initFlag;
   //int numTotalFrames;
   //long* frameNumber;
#else // PV_USE_GDAL
public:
   Movie(char const * name, HyPerCol * hc);
protected:
   Movie();
#endif // PV_USE_GDAL
};

}

#endif /* MOVIE_HPP_ */
