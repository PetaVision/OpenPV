/*
 * PoolingConn.hpp
 *
 *  Created on: March 19, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "HyPerConn.hpp"
#include "../layers/PoolingIndexLayer.hpp"
namespace PV {

class PoolingConn: public HyPerConn {

public:
   enum AccumulateType {UNDEFINED, MAX, SUM, AVG};
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
   PoolingIndexLayer* getPostIndexLayer(){return postIndexLayer;}
   bool needPostIndex(){return needPostIndexLayer;}
   inline AccumulateType getPoolingType() const { return poolingType; }

protected:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);
   void ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag);
   void ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual int setInitialValues();
   virtual int constructWeights();

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID);
   //virtual void deliverOnePreNeuronActivity(int kPreExt, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);
   //virtual void deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t*
   //gSynPatchPos, float dt_factor, taus_uint4 * rngPtr);
   void clearGateIdxBuffer();

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   int deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID){
      pvError() << "Pooling Conn does not allow GPU deliver\n";
   }
   int deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID){
      pvError() << "Pooling Conn does not allow GPU deliver\n";
   }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)


private:
   int initialize_base();
   void unsetAccumulateType();
   int ** thread_gateIdxBuffer;
   bool needPostIndexLayer;
   char* postIndexLayerName;
   PoolingIndexLayer* postIndexLayer;
   AccumulateType poolingType;
}; // end class PoolingConn

BaseObject * createPoolingConn(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
