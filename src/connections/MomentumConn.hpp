/*
 * MomentumConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMCONN_HPP_
#define MOMENTUMCONN_HPP_

#include "HyPerConn.hpp"
namespace PV {

class MomentumConn: public HyPerConn {

public:
   MomentumConn();
   MomentumConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~MomentumConn();
   virtual int allocateDataStructures();

   virtual int updateState(double time, double dt);
   virtual int applyMomentum(int arbor_ID);
   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int reduceKernels(const int arborID);
   virtual int checkpointWrite(const char * cpDir);
protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchPeriod(enum ParamsIOFlag ioFlag);
   inline pvwdata_t* get_prev_dwDataHead(int arborId, int dataIndex) {
      return &prev_dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }
   virtual int defaultUpdate_dW(int arbor_ID);
   virtual int sumKernels(const int arborID);


private:
   int initialize_base();
   pvwdata_t** prev_dwDataStart;
   float momentumTau;
   float momentumDecay;
   char* momentumMethod;
   int batchIdx;
   int batchPeriod;


}; // end class 

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
