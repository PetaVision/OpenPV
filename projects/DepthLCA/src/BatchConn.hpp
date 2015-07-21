/*
 * BatchConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef BATCHCONN_HPP_
#define BATCHCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class BatchConn: public HyPerConn{
public:
   BatchConn(const char * name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchPeriod(enum ParamsIOFlag ioFlag);
   virtual int reduceKernels(const int arborID);
   virtual int allocateDataStructures();
   void sumKernelActivations();
   int sumKernels(const int arborID);
   virtual int updateState(double time, double dt);
protected:
   BatchConn();
   int initialize_base();
   virtual int defaultUpdate_dW(int arbor_ID);
   virtual int normalize_dW(int arbor_ID);

private:
   int batchIdx;
   int batchPeriod;
};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
