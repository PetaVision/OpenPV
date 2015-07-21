/*
 * ShadowRandomPatchMovie.cpp
 *
 *      Author: Pete Schultz
 * Like RandomPatchMovie, but whenever instead of generating its own random numbers,
 * it copies off the worksheet of another RandomPatchMovie
 *
 * This way, you can be sure to take the same patch from the same image if
 * retina-on and retina-off images are loaded using two different catalogs of images.
 */

#include "ShadowRandomPatchMovie.hpp"

namespace PV {

ShadowRandomPatchMovie::ShadowRandomPatchMovie() {
   initialize_base();
}

ShadowRandomPatchMovie::ShadowRandomPatchMovie(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ShadowRandomPatchMovie::~ShadowRandomPatchMovie() {
   free(shadowedRandomPatchMovieName);
}

int ShadowRandomPatchMovie::initialize_base() {
   shadowedRandomPatchMovieName = NULL;
   shadowedRandomPatchMovie = NULL;
   return PV_SUCCESS;
}

int ShadowRandomPatchMovie::initialize(const char * name, HyPerCol * hc) {
   assert( this->shadowedRandomPatchMovie == NULL );
   int status = RandomPatchMovie::initialize(name, hc);
   return status;
}

int ShadowRandomPatchMovie::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = RandomPatchMovie::ioParamsFillGroup(ioFlag);
   ioParam_shadowedRandomPatchMovie(ioFlag);
   return status;
}

void ShadowRandomPatchMovie::ioParam_shadowedRandomPatchMovie(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "shadowedRandomPatchMovie", &shadowedRandomPatchMovieName);

}

int ShadowRandomPatchMovie::communicateInitInfo() {
   int status = RandomPatchMovie::communicateInitInfo();
   HyPerLayer * shadowedLayer = parent->getLayerFromName(shadowedRandomPatchMovieName);
   shadowedRandomPatchMovie = dynamic_cast<RandomPatchMovie *>(shadowedLayer);
   if( this->shadowedRandomPatchMovie == NULL ) {
      fprintf(stderr, "ShadowRandomPatchMovie \"%s\": shadowed layer \"%s\" must be a RandomPatchMovie\n", name, shadowedRandomPatchMovieName);
      exit(EXIT_FAILURE);
   }
   return status;
}

int ShadowRandomPatchMovie::getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr) {
   *offsetXptr = shadowedRandomPatchMovie->getOffsetX();
   *offsetYptr = shadowedRandomPatchMovie->getOffsetY();
   return PV_SUCCESS;
}

int ShadowRandomPatchMovie::getRandomFileIndex() {
   return shadowedRandomPatchMovie->getFileIndex();
}

}  // end namespace PV
