/*
 * VaryingKernelConn.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef VARYINGKERNELCONN_HPP_
#define VARYINGKERNELCONN_HPP_

#include <connections/KernelConn.hpp>

namespace PV {

class VaryingKernelConn : public KernelConn {

public:
   VaryingKernelConn(const char * name, HyPerCol * hc);
   virtual ~VaryingKernelConn();
   virtual int allocateDataStructures();

protected:
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void readPlasticityFlag(PVParams * params);
   virtual void readShmget_flag(PVParams * params);
   virtual int calc_dW(int axonId);

}; // end class VaryingKernelConn

}  // end namespace PV block


#endif /* VARYINGKERNELCONN_HPP_ */
