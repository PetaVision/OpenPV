/*
 * PoolingConn.hpp
 *
 *  Created on: March 19, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "HyPerConn.hpp"
namespace PV {

class PoolingConn: public HyPerConn {

public:
   PoolingConn();
   PoolingConn(const char * name, HyPerCol * hc);
   virtual ~PoolingConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);
   virtual int finalizeUpdate(double time, double dt);
protected:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   virtual int setInitialValues();
   virtual int constructWeights();

   virtual void deliverOnePreNeuronActivity(int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);
   virtual void deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, uint4 * rngPtr);

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   int deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID){
      std::cout << "Pooling Conn does not allow GPU deliver\n";
      exit(-1);
   }
   int deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID){
      std::cout << "Pooling Conn does not allow GPU deliver\n";
      exit(-1);
   }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)


private:
   int initialize_base();


}; // end class 

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
