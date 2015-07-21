/*
 * LCALIFLateralConn.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: pschultz
 */

#ifndef LCALIFLATERALCONN_HPP_
#define LCALIFLATERALCONN_HPP_

#include "HyPerConn.hpp"
#include "../layers/LCALIFLayer.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"

namespace PV {

class LCALIFLateralConn: public PV::HyPerConn {

// Methods
public:
   LCALIFLateralConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~LCALIFLateralConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonId = 0);

   float getIntegratedSpikeCount(int kex) {return integratedSpikeCount[kex];}
   float getIntegrationTimeConstant() {return integrationTimeConstant;}
   float getInhibitionTimeConstant() {return inhibitionTimeConstant;}
   float getTargetRateKHz() {return targetRateKHz;}

   virtual int outputState(double time, bool last=false);

   virtual int checkpointWrite(const char * cpDir);
   virtual int checkpointRead(const char * cpDir, double * timef);

protected:
   LCALIFLateralConn();
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_integrationTimeConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inhibitionTimeConstant(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_coorThresh(enum ParamsIOFlag ioFlag);
   virtual int calc_dW(int axonId = 0);

   virtual int updateIntegratedSpikeCount();

private:
   int initialize_base();

// Member variables
protected:
   PVLayerCube * integratedSpikeCountCube;
   float * integratedSpikeCount; // The leaky count of spikes (the weight is a decaying exponential of time since that spike)
   MPI_Datatype * mpi_datatype;  // Used to mirror the presynaptic traces
   float integrationTimeConstant; // Time constant for the integrated spike counts, often the same as the the LCALIFLayer's tau_LCA
   float inhibitionTimeConstant; // Time constant tau_{inh}, the timescale for updating he weights in this connection
   float targetRateKHz;          // Target rate in kilohertz; note that params file is understood to give value in hertz
   float corrThresh;             // Correlation threshold
};

} /* namespace PV */
#endif /* LCALIFLATERALCONN_HPP_ */
