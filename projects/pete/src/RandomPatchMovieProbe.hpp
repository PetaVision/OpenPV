/*
 * RandomPatchMovieProbe.hpp
 *
 *  Created on: Sep 20, 2011
 *      Author: pschultz
 */

#ifndef RANDOMPATCHMOVIEPROBE_HPP_
#define RANDOMPATCHMOVIEPROBE_HPP_

#include <io/LayerProbe.hpp>
#include "RandomPatchMovie.hpp"

namespace PV {

class RandomPatchMovieProbe : public LayerProbe {
public:
   RandomPatchMovieProbe(const char * probename, HyPerCol * hc);
   virtual ~RandomPatchMovieProbe();
   int initRandomPatchMovieProbe(const char * probename, HyPerCol * hc);
   virtual int outputState(double timed);

protected:
   char * name;
   float displayPeriod;
   float nextDisplayTime;
}; // end class RandomPatchMovieProbe

}  // end namespace PV

#endif /* RANDOMPATCHMOVIEPROBE_HPP_ */
