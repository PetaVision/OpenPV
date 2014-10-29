/*
 * ConnStatsProbe.hpp
 *
 *  Created on: Oct 27, 2014
 *      Author: pschultz
 */

#ifndef CONNSTATSPROBE_HPP_
#define CONNSTATSPROBE_HPP_

#include "BaseHyPerConnProbe.hpp"

namespace PV {

class ConnStatsProbe: public BaseHyPerConnProbe {
public:
   /**
    * The public constructor for creating a ConnStatsProbe class
    * probeName is a name uniquely identifying the ConnStatsProbe
    * hc is the HyPerCol that owns the connection the probe is
    * associated with.
    */
   ConnStatsProbe(const char * probeName, HyPerCol * hc);
   virtual ~ConnStatsProbe();

   /**
    * Allocates buffers to hold the stats for each patch.
    * Each of the member variables sums, sumabs, sumsquares, maxes, mins, maxabs
    * are allocated as a buffer of length (numArbors+1)*numDataPatches
    */
   virtual int allocateDataStructures();

   /**
    * For each patch in targetConn, computes the sum, L1-norm, L2-norm,
    * min, max, and max-absolute values of the weights in that patch.
    * If targetConn has more than one arbor, it reports the stats for
    * each arbor separately, as well as the stats for the arbors in
    * aggregate.
    */
   virtual int outputState(double simtime);

protected:
   /**
    * The default constructor called by classes derived from ConnStatsProbe.
    * It calls initialize_base() but does not call initialize(); the
    * derived class's initialization should make sure to call
    * ConnStatsProbe::initialize()
    */
   ConnStatsProbe();

   /**
    * The initialization that sets parameters to their desired values.
    * (Currently ConnStatsProbe has no parameters)
    */
   int initialize(const char * probeName, HyPerCol * hc);

private:
   /**
    * The routine called by all ConnStatsProbe constructors,
    * that sets member variables to safe values and defaults.
    */
   int initialize_base();

//Member variables
protected:
   char * statsptr; // All stats below are allocated in one buffer
   double * sums;
   double * sumabs;
   double * sumsquares;
   float * maxes;
   float * mins;
   float * maxabs;
};

} /* namespace PV */

#endif /* CONNSTATSPROBE_HPP_ */
