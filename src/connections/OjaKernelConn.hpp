/*
 * OjaKernelConn.h
 *
 *  Created on: Oct 10, 2012
 *      Author: pschultz
 */

#ifndef OJAKERNELCONN_H_
#define OJAKERNELCONN_H_

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
   virtual int update_dW(int axonId);
   virtual int updateWeights(int axonId);

   // Load member variables from params.  Virtual so that derived classes can deactivate a param if it isn't needed.
   virtual float readLearningTime() {return getParent()->parameters()->value(name, "learningTime", 1.0);}
   virtual float readInputTargetRate() {return getParent()->parameters()->value(name, "inputTargetRate", 1.0);}
   virtual float readOutputTargetRate() {return getParent()->parameters()->value(name, "outputTargetRate", 1.0);}
   virtual float readIntegrationTime() {return getParent()->parameters()->value(name, "integrationTime", 1.0);}
   virtual float readAlphaMultiplier() {return getParent()->parameters()->value(name, "alphaMultiplier", 1.0);}

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

};

} /* namespace PV */
#endif /* OJAKERNELCONN_H_ */
