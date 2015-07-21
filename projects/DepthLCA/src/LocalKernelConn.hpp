/*
 * LocalKernelConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef LOCALKERNELCONN_HPP_
#define LOCALKERNELCONN_HPP_

#include <connections/KernelConn.hpp>
#include <assert.h>
#include <string.h>

namespace PV {

class LocalKernelConn: public KernelConn {
public:
   LocalKernelConn(const char * name, HyPerCol *hc);
   //int defaultUpdateInd_dW(int arbor_ID, int kExt);
   int updateWeights(int arbor_ID);

protected:
   LocalKernelConn();
   int initialize_base();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_decay(enum ParamsIOFlag ioFlag);
   float decay;

};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
