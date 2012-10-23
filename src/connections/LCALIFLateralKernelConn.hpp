/*
 * LCALIFLateralKernelConn.hpp
 *
 *  Created on: Oct 17, 2012
 *      Author: pschultz
 */

#ifndef LCALIFLATERALKERNELCONN_HPP_
#define LCALIFLATERALKERNELCONN_HPP_

#include "KernelConn.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"

namespace PV {

class LCALIFLateralKernelConn: public KernelConn {

   // Methods
   public:
      LCALIFLateralKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, const char * filename, InitWeights * weightInit);
      virtual ~LCALIFLateralKernelConn();
      virtual int updateWeights(int axonId = 0);

      float getIntegratedSpikeCount(int kex) {return integratedSpikeCount[kex];}
      float getIntegrationTimeConstant() {return integrationTimeConstant;}
      float getInhibitionTimeConstant() {return inhibitionTimeConstant;}
      float getTargetRateKHz() {return targetRateKHz;}

      virtual int setParams(PVParams * params); // Really should be protected

      virtual int checkpointWrite(const char * cpDir);
      virtual int checkpointRead(const char * cpDir, double * timef);

   protected:
      LCALIFLateralKernelConn();
      int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, const char * filename, InitWeights * weightInit);
      virtual int update_dW(int axonId = 0);

      virtual float readIntegrationTimeConstant() {return getParent()->parameters()->value(name, "integrationTimeConstant", 1.0);}
      virtual float readInhibitionTimeConstant() {return getParent()->parameters()->value(name, "inhibitionTimeConstant", 1.0);}
      virtual float readTargetRate() {return getParent()->parameters()->value(name, "targetRate", 1.0);}

      virtual int updateIntegratedSpikeCount();

   private:
      int initialize_base();

   // Member variables
   protected:
      float * integratedSpikeCount; // The leaky count of spikes (the weight is a decaying exponential of time since that spike)
      float integrationTimeConstant; // Time constant for the integrated spike counts, often the same as the the LCALIFLayer's tau_LCA
      float inhibitionTimeConstant; // Time constant tau_{inh}, the timescale for updating he weights in this connection
      float targetRateKHz;          // Target rate in kilohertz; note that params file is understood to give value in hertz
};

} /* namespace PV */
#endif /* LCALIFLATERALKERNELCONN_HPP_ */
