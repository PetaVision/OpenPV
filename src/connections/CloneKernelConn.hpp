/*
 * CloneKernelConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONN_HPP_
#define CLONEKERNELCONN_HPP_

#include "KernelConn.hpp"
#include "../weightinit/InitCloneKernelWeights.hpp"

namespace PV {

class CloneKernelConn : public KernelConn {

public:
   CloneKernelConn(const char * name, HyPerCol * hc);
   virtual ~CloneKernelConn();

   virtual int communicateInitInfo();

   virtual int updateState(double time, double dt);

   virtual int writeWeights(double time, bool last=false){return PV_SUCCESS;};
   virtual int writeWeights(const char * filename){return PV_SUCCESS;};
   virtual int checkpointWrite(const char * cpDir){return PV_SUCCESS;};
   virtual int checkpointRead(const char * cpDir, double *timef){return PV_SUCCESS;};

protected:
   CloneKernelConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
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
   virtual int setWeightInitializer();
   virtual PVPatch *** initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches,
            const char * filename);
   virtual int constructWeights();
   void constructWeightsOutOfMemory();
   virtual int createAxonalArbors(int arborId);

   virtual int  setPatchSize(); // virtual int setPatchSize(const char * filename); // filename is now a member variable.

   char * originalConnName;
   KernelConn * originalConn;

private:
   int initialize_base();
   int deleteWeights();

}; // end class CloneKernelConn

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
