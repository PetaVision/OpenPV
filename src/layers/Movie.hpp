/*
 * Movie.hpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#ifndef MOVIE_HPP_
#define MOVIE_HPP_

#include "Image.hpp"

namespace PV {

class Movie: public PV::Image {
public:
   Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames);
   Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod);
   virtual ~Movie();

   virtual pvdata_t * getImageBuffer();
   virtual PVLayerLoc getImageLoc();

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int outputState(double time, bool last=false);
   virtual int updateState(double time, double dt);
   bool        updateImage(double time, double dt);

   int  randomFrame();
   void calcBias(int step, int sizeLength);
   void calcBiasedOffset(int step, int sizeLength);
   int  calcPosition(int pos, int step, int sizeLength);

   int resetPositionInBounds();

protected:
   Movie();
   int initialize(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod);

private:
   int initialize_base();
   int initializeMovie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod);
   int copyReducedImagePortion();
   const char * getNextFileName();
   const char * getNextFileName(int n_skip);

   float displayPeriod;   // length of time a frame is displayed
   float nextDisplayTime; // time of next frame

   int stepSize;

   // int offsetX; // moved to Image
   // int offsetY;

   int jitterFlag;        // If true, use jitter

   int biasX;             // offsetX/Y jitter around biasX/Y location
   int biasY;

   float recurrenceProb;  // If using jitter, probability that offset returns to bias position
   float persistenceProb; // If using jitter, probability that offset stays the same

   int writePosition;     // If using jitter, write positions to input/image-pos.txt
   int biasChangeTime;    // If using jitter, time period for recalculating bias position

   int randomMovie;       // these are used for performing a reverse correlation analysis
   float randomMovieProb;

   bool echoFramePathnameFlag; // if true, echo the frame pathname to stdout

   int skipFrameIndex; // skip this number of frames between each load

   char inputfile[PV_PATH_MAX];  // current input file name
   char * movieOutputPath;  // path to output file directory for movie frames

   FILE * fp;
   FILE * fp_pos;
};

}

#endif /* MOVIE_HPP_ */
