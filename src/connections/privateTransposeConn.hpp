/*
 * privateTransposeConn.hpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#ifndef PRIVATETRANSPOSECONN_HPP_
#define PRIVATETRANSPOSECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class privateTransposeConn : public HyPerConn {
  public:
   privateTransposeConn(
         const char *name,
         HyPerCol *hc,
         HyPerConn *parentConn,
         bool needWeights = true);
   virtual ~privateTransposeConn();
   virtual int communicateInitInfo(CommunicateInitInfoMessage const *message);
   virtual int allocateDataStructures();
   inline HyPerConn *getOriginalConn() { return postConn; }

   virtual bool needUpdate(double timed, double dt);
   virtual int updateState(double time, double dt);
   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
   virtual int finalizeUpdate(double time, double dt);

   virtual int deliver();

  protected:
   int initialize(const char *name, HyPerCol *hc, HyPerConn *parentConn, bool needWeights);
   virtual int setDescription();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int setPatchSize();
   virtual int setNeededRNGSeeds() { return 0; }
   virtual int setInitialValues();
   virtual PVPatch ***initializeWeights(PVPatch ***arbors, float **dataStart);
   int transpose(int arborId);
   virtual int reduceKernels(int arborID);
   virtual int initializeDelays(const float *fDelayArray, int size);
   virtual int constructWeights();

  private:
   int transposeSharedWeights(int arborId);
   int transposeNonsharedWeights(int arborId);
   bool needAllocWeights;

   /**
    * Calculates the parameters of the the region that needs to be sent to adjoining processes using
    * MPI.
    * Used only in the sharedWeights=false case, because in that case an individual weight's pre and
    * post neurons can live in different processes.
    */
   int mpiexchangesize(
         int neighbor,
         int *size,
         int *startx,
         int *stopx,
         int *starty,
         int *stopy,
         int *blocksize,
         size_t *buffersize);
};

} // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
