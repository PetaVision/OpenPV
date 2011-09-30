/*
 * RandomPatchMovieProbe.hpp
 *
 *  Created on: Sep 20, 2011
 *      Author: pschultz
 */

#ifndef RANDOMPATCHMOVIEPROBE_HPP_
#define RANDOMPATCHMOVIEPROBE_HPP_

#include <src/io/LayerProbe.hpp>
#include "RandomPatchMovie.hpp"

namespace PV {

class RandomPatchMovieProbe : public LayerProbe {
public:
   RandomPatchMovieProbe(const char * filename, HyPerCol * hc, const char * probename = NULL);
   virtual ~RandomPatchMovieProbe();
   int initialize(const char * probename);
   virtual int outputState(float time, HyPerLayer * l);

protected:
   char * name;
   float displayPeriod;
   float nextDisplayTime;
}; // end class RandomPatchMovieProbe

}  // end namespace PV

#endif /* RANDOMPATCHMOVIEPROBE_HPP_ */
