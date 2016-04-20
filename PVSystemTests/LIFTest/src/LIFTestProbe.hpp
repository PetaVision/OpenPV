/*
 * LIFTestProbe.hpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#ifndef LIFTESTPROBE_HPP_
#define LIFTESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>

namespace PV {

class LIFTestProbe : public StatsProbe {
public:
   LIFTestProbe(const char * probeName, HyPerCol * hc);
   virtual ~LIFTestProbe();

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   virtual int outputState(double timed);

protected:
   LIFTestProbe();
   int initLIFTestProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_endingTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tolerance(enum ParamsIOFlag ioFlag);

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

BaseObject * createLIFTestProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* LIFTESTPROBE_HPP_ */
