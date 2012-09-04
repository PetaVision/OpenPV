/*
 * LIFTestProbe.hpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#ifndef LIFTESTPROBE_HPP_
#define LIFTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

class LIFTestProbe : public StatsProbe {
public:
   LIFTestProbe(const char * filename, HyPerLayer * layer, const char * msg, const char * probename);
   LIFTestProbe(HyPerLayer * layer, const char * msg, const char * probename);
   LIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg, const char * probename);
   LIFTestProbe(HyPerLayer * layer, PVBufType type, const char * msg, const char * probename);
   virtual ~LIFTestProbe();

   virtual int outputState(float timef);

protected:
   LIFTestProbe();
   int initLIFTestProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg, const char * probename);

private:
   int initialize_base();

private:
   double * radii;
   double * rates;
   double * targetrates;
   double * stddevs;
   int * counts;

   float endingTime; // The time, in the same units dt is in, at which to stop the test.
   float tolerance; // Number of standard deviations that the observed rates can differ from the expected rates.
};

} /* namespace PV */
#endif /* LIFTESTPROBE_HPP_ */
