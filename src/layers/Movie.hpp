/*
 * Movie.hpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#ifndef MOVIE_HPP_
#define MOVIE_HPP_

#include "Image.hpp"
#include <sstream>

namespace PV {

class Movie: public PV::Image {
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
   virtual double calcTimeScale();
   //virtual int updateStateWrapper(double time, double dt);
   virtual int updateState(double time, double dt);
   virtual bool updateImage(double time, double dt);
   // bool        getNewImageFlag();
   const char * getCurrentImage();

   int  randomFrame();

protected:
   Movie();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imagePath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_frameNumber(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imageListPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_randomMovie(enum ParamsIOFlag ioFlag);
   virtual void ioParam_randomMovieProb(enum ParamsIOFlag ioFlag);
   virtual void ioParam_readPvpFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);
   virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);
   virtual void ioParam_movieOutputPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag);
   virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readFrameNumStateFromCheckpoint(const char * cpDir);

   bool readPvpFile;
   const char * getNextFileName(int n_skip);
   int updateFrameNum(int n_skip);
   int skipFrameIndex; // skip this number of frames between each load
   PV_Stream * timestampFile;


private:
   int initialize_base();
   int copyReducedImagePortion();
   int updateFrameNum();
   const char * advanceFileName();

   bool resetToStartOnLoop;

   double displayPeriod;   // length of time a frame is displayed
   //double nextDisplayTime; // time of next frame; now handled by HyPerLayer nextUpdateTime

   int randomMovie;       // these are used for performing a reverse correlation analysis
   float randomMovieProb;

   bool echoFramePathnameFlag; // if true, echo the frame pathname to stdout
   // bool newImageFlag; // true when a new image was presented this timestep;

   int startFrameIndex;

   char inputfile[PV_PATH_MAX];  // current input file name
   char * movieOutputPath;  // path to output file directory for movie frames

   int numFrames; //Number of frames
   char * fileOfFileNames;

   PV_Stream * filenamestream;

   bool writeFrameToTimestamp;

   bool flipOnTimescaleError;
};

}

#endif /* MOVIE_HPP_ */
