/*
 * ShadowRandomPatchMovie.hpp
 *
 *  Created on: Sep 23, 2011
 *      Author: pschultz
 */

#ifndef SHADOWRANDOMPATCHMOVIE_HPP_
#define SHADOWRANDOMPATCHMOVIE_HPP_

#include "RandomPatchMovie.hpp"
#include <columns/buildandrun.hpp>

namespace PV {

class ShadowRandomPatchMovie : public RandomPatchMovie {
public:
   ShadowRandomPatchMovie(const char * name, HyPerCol * hc);
   virtual ~ShadowRandomPatchMovie();
   virtual int getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr);
   virtual int getRandomFileIndex();

protected:
   ShadowRandomPatchMovie();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shadowedRandomPatchMovie(enum ParamsIOFlag ioFlag);
   virtual int communicateInitInfo();

private:
   int initialize_base();

protected:
   char * shadowedRandomPatchMovieName;
   RandomPatchMovie * shadowedRandomPatchMovie;
};

}  // end namespace PV


#endif /* SHADOWRANDOMPATCHMOVIE_HPP_ */
