/*
 * ShadowRandomPatchMovie.hpp
 *
 *  Created on: Sep 23, 2011
 *      Author: pschultz
 */

#ifndef SHADOWRANDOMPATCHMOVIE_HPP_
#define SHADOWRANDOMPATCHMOVIE_HPP_

#include "RandomPatchMovie.hpp"
#include <src/columns/buildandrun.hpp>

namespace PV {

class ShadowRandomPatchMovie : public RandomPatchMovie {
public:
   ShadowRandomPatchMovie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod=DISPLAY_PERIOD);
   virtual ~ShadowRandomPatchMovie();
   virtual int getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr);
   virtual int getRandomFileIndex();

protected:
   ShadowRandomPatchMovie();
   int initialize(const char * name, HyPerCol * hc, const char * fileOfFileNames, float defaultDisplayPeriod);
   RandomPatchMovie * shadowedRandomPatchMovie;

private:
   int initialize_base();
};

}  // end namespace PV


#endif /* SHADOWRANDOMPATCHMOVIE_HPP_ */
