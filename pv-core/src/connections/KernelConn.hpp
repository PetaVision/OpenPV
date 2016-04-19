/*
 * KernelConn.hpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#ifndef KERNELCONN_HPP_
#define KERNELCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class KernelConn: public HyPerConn {

public:
   KernelConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~KernelConn();

protected:
   KernelConn();
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
}; // class KernelConn

BaseObject * createKernelConn(char const * name, HyPerCol * hc);

} // namespace PV

#endif /* KERNELCONN_HPP_ */
