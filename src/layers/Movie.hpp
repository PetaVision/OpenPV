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
   Movie(const char * fileOfFileNames, HyPerCol * hc, float displayPeriod);
   virtual ~Movie();

   virtual pvdata_t * getImageBuffer();
   virtual LayerLoc   getImageLoc();

   bool updateImage(float time, float dt);

   int getReducedImage(const char * filename);
   const char * getNextFileName();

   LayerLoc   imageLoc;
   pvdata_t * imageData;

   float displayPeriod;     // length of time a frame is displayed
   float nextDisplayTime;   // time of next frame

   char inputfile[PV_PATH_MAX];  // current input file name

   FILE * fp;
};

}

#endif /* MOVIE_HPP_ */
