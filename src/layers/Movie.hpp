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
   Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod);
   virtual ~Movie();

   virtual pvdata_t * getImageBuffer();
   virtual PVLayerLoc getImageLoc();

   virtual int outputState(float time, bool last=false);
   virtual int updateState(float time, float dt);
   bool        updateImage(float time, float dt);

   int calcPosition(int pos, int step, int sizeLength);
   int resetPositionInBounds();

private:

   int copyReducedImagePortion();
   const char * getNextFileName();

   float displayPeriod;     // length of time a frame is displayed
   float nextDisplayTime;   // time of next frame

   int offsetX;             // offset of layer section within full movie frame
   int offsetY;

   char inputfile[PV_PATH_MAX];  // current input file name

   FILE * fp;
};

}

#endif /* MOVIE_HPP_ */
