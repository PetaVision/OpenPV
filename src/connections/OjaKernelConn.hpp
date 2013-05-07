/*
 * OjaKernelConn.hpp
 *
 *  Created on: Oct 10, 2012
 *      Author: pschultz
 */

#ifndef OJAKERNELCONN_HPP_
#define OJAKERNELCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class OjaKernelConn: public PV::KernelConn {

// Methods
public:
   // Public constructor, called when creating a new OjaKernelConn object
   OjaKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                 const char * filename = NULL, InitWeights *weightInit = NULL);

   virtual ~OjaKernelConn();

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
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                  const char * filename, InitWeights *weightInit);
   virtual int calc_dW(int axonId);
   virtual int update_dW(int axonId);
   virtual int updateWeights(int axonId);

   virtual int setParams(PVParams * params);
   // Load member variables from params.  Virtual so that derived classes can deactivate a param if it isn't needed.
   virtual void readLearningTime(PVParams * params) {learningTime = params->value(name, "learningTime", 1.0);}

   // params file specifies target rates in hertz; convert to KHz since times are in ms
   virtual void readInputTargetRate(PVParams * params) {inputTargetRate = 0.001*params->value(name, "inputTargetRate", 1.0);}
   virtual void readOutputTargetRate(PVParams * params) {outputTargetRate = 0.001*params->value(name, "outputTargetRate", 1.0);}

   virtual void readIntegrationTime(PVParams * params) {integrationTime = params->value(name, "integrationTime", 1.0);}
   virtual void readAlphaMultiplier(PVParams * params) {alphaMultiplier = params->value(name, "alphaMultiplier", 1.0);}

   virtual void read_dWUpdatePeriod(PVParams * params) {dWUpdatePeriod = params->value(name, "dWUpdatePeriod", 1.0);}
   virtual void readInitialWeightUpdateTime(PVParams * params);

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
   float dWUpdatePeriod;
   float dWUpdateTime;
};

} /* namespace PV */
#endif /* OJAKERNELCONN_HPP_ */
