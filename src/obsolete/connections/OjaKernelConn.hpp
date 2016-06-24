/*
 * OjaKernelConn.hpp
 *
 *  Created on: Oct 10, 2012
 *      Author: pschultz
 */

#ifndef OJAKERNELCONN_HPP_
#define OJAKERNELCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class OjaKernelConn: public PV::HyPerConn {

// Methods
public:
   // Public constructor, called when creating a new OjaKernelConn object
   OjaKernelConn(const char * name, HyPerCol * hc);

   virtual ~OjaKernelConn();

   virtual int allocateDataStructures();
   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);

   virtual int updateState(double timef, double dt);

   // Public get-methods
   float getLearningTime() {return learningTime;}
   float getInputTargetRate() {return inputTargetRate;}
   float getOutputTargetRate() {return outputTargetRate;}
   float getIntegrationTime() {return integrationTime;}

   const pvdata_t * getInputFiringRate(int axonID) {return inputFiringRate[axonID];}
   const pvdata_t * getOutputFiringRate() {return outputFiringRate;}

protected:
   OjaKernelConn(); // Called by derived classes' constructors
   int initialize(const char * name, HyPerCol * hc);
   virtual int calc_dW(int axonId);
   virtual int update_dW(int axonId);
   virtual int updateWeights(int axonId);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_learningTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputTargetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputTargetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_alphaMultiplier(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWUpdatePeriod(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
protected:
   float learningTime;
   float inputTargetRate;
   float outputTargetRate;
   float integrationTime; // \tau_{\hbox{\it Oja}}
   PVLayerCube ** inputFiringRateCubes; // inputFiringRate[arbor]
   pvdata_t ** inputFiringRate; // inputFiringRate[arbor][patchIndex]
   pvdata_t * outputFiringRate; // outputFiringRate[output neuron (in restricted space)]
   MPI_Datatype * mpi_datatype;   // Used to mirror the inputFiringRateCubes
   float alphaMultiplier;
   double dWUpdatePeriod;
   double dWUpdateTime;
};

} /* namespace PV */
#endif /* OJAKERNELCONN_HPP_ */
