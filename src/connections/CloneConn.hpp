/*
 * CloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONECONN_HPP_
#define CLONECONN_HPP_

#include "HyPerConn.hpp"
#include "../weightinit/InitCloneKernelWeights.hpp"

namespace PV {

class CloneConn : public HyPerConn {

public:
   CloneConn(const char * name, HyPerCol * hc);
   virtual ~CloneConn();

   virtual int communicateInitInfo();

   virtual int updateState(double time, double dt);

   virtual int writeWeights(double time, bool last=false){return PV_SUCCESS;};
   virtual int writeWeights(const char * filename){return PV_SUCCESS;};
   virtual int checkpointWrite(const char * cpDir){return PV_SUCCESS;};
   virtual int checkpointRead(const char * cpDir, double *timef){return PV_SUCCESS;};

   HyPerConn * getOriginalConn(){return originalConn;}

protected:
   CloneConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   virtual int setWeightInitializer();
   virtual PVPatch *** initializeWeights(PVPatch *** patches, pvdata_t ** dataStart);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr) { return PV_SUCCESS; }
   virtual int constructWeights();
   void constructWeightsOutOfMemory();
   virtual int createAxonalArbors(int arborId);

   virtual int  setPatchSize(); // virtual int setPatchSize(const char * filename); // filename is now a member variable.

   char * originalConnName;
   HyPerConn * originalConn;

private:
   int initialize_base();
   int deleteWeights();

}; // end class CloneConn

}  // end namespace PV

#endif /* CLONECONN_HPP_ */
