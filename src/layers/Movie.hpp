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
   /**
    * List of parameters needed from the Movie class
    * @name Movie Parameters
    * @{
    */

   /**
    * @brief imagePath: Movie does not use the Image parameter imagePath.  Instead, it uses imageListPath to define the list of images.
    */
   virtual void ioParam_imagePath(enum ParamsIOFlag ioFlag);

   /**
    * @brief frameNumber: Movie does not use the Image parameter frameNumber.  Instead, it uses start_frame_index to load the first frame.
    */
   virtual void ioParam_frameNumber(enum ParamsIOFlag ioFlag);

   /**
    * @brief imageListPath: The file containing the list of images.
    * @details Relative paths are with respect to the working directory.
    * imageListPath can point to either a text file containing a list of image files or URLs, or a .pvp file.
    * If a .pvp file, each frame of the file consititutes an image.
    */
   virtual void ioParam_imageListPath(enum ParamsIOFlag ioFlag);

   /**
    * @brief displayPeriod: the amount of time each image is displayed before switching to the next image.
    * The units of displayPeriod are the same as the units of the HyPerCol's dt parameter.
    */
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief randomMovie: if true, image pixels are randomly set to one or zero, instead of being loaded from the imageListPath
    */
   virtual void ioParam_randomMovie(enum ParamsIOFlag ioFlag);

   /**
    * @brief randomMovieProb: if randomMovie is true, the probability that that a pixel is set to one.
    */
   virtual void ioParam_randomMovieProb(enum ParamsIOFlag ioFlag);

   /**
    * @brief readPvpFile: if true, the file in imageListPath is treated as a pvp file.
    */
   virtual void ioParam_readPvpFile(enum ParamsIOFlag ioFlag);

   /**
    * @brief echoFramePathnameFlag: if true, print the filename to the screen when a new image file is loaded.
    */
   virtual void ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief start_frame_index: Initialize the layer with the given frame.
    * @details start_frame_index=1 means the first line of the imageListPath if a text file,
    * or the initial frame if imageListPath is a .pvp file.  A start_frame_index of zero (or less)
    * gets changed internally to start_frame_index=1.
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
   /** @} */

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
