/*
 * LCALIFLateralKernelConn.hpp
 *
 *  Created on: Oct 17, 2012
 *      Author: pschultz
 */

#ifndef LCALIFLATERALKERNELCONN_HPP_
#define LCALIFLATERALKERNELCONN_HPP_

#include "HyPerConn.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"

namespace PV {

class LCALIFLateralKernelConn: public HyPerConn {

   // Methods
public:
   LCALIFLateralKernelConn(const char * name, HyPerCol * hc);
   virtual ~LCALIFLateralKernelConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonId = 0);

   float getIntegratedSpikeCount(int kex) {return integratedSpikeCount[kex];}
   float getIntegrationTimeConstant() {return integrationTimeConstant;}
   float getInhibitionTimeConstant() {return inhibitionTimeConstant;}
   float getTargetRateKHz() {return targetRateKHz;}

   virtual int checkpointWrite(const char * cpDir);
   virtual int checkpointRead(const char * cpDir, double * timef);

protected:
   LCALIFLateralKernelConn();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_integrationTimeConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inhibitionTimeConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWUpdatePeriod(enum ParamsIOFlag ioFlag);

   virtual int update_dW(int axonId = 0);

   virtual int updateIntegratedSpikeCount();

private:
   int initialize_base();

   // Member variables
protected:
   PVLayerCube * integratedSpikeCountCube;
   float * integratedSpikeCount; // The leaky count of spikes (the weight is a decaying exponential of time since that spike)
   float integrationTimeConstant; // Time constant for the integrated spike counts, often the same as the the LCALIFLayer's tau_LCA
   float inhibitionTimeConstant; // Time constant tau_{inh}, the timescale for updating he weights in this connection
   float targetRateHz;          // Target rate in hertz; note that params file is understood to give value in hertz
   float targetRateKHz;          // Target rate in kilohertz; note that params file is understood to give value in hertz
   double dWUpdatePeriod;         // Only update dW this often.  Param value is in the same units as dt.
   double dWUpdateTime;           // The next time that dW will update

   MPI_Datatype * mpi_datatype;  // Used to mirror the integrated spike count
   float ** interiorCounts;         // We should average only over the patches where the presynaptic neuron is in the restricted patch, to eliminate correlations caused by mirroring.  This buffer maintains the count to divide by in obtaining the average.
};

} /* namespace PV */
#endif /* LCALIFLATERALKERNELCONN_HPP_ */
