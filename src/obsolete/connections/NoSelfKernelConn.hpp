/*
 * NoSelfKernelConn.hpp
 *
 *  Created on: Sep 20, 2011
 *      Author: gkenyon
 */

#ifndef NOSELFKERNELCONN_HPP_
#define NOSELFKERNELCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class NoSelfKernelConn: public PV::HyPerConn {
public:
   NoSelfKernelConn();

   NoSelfKernelConn(const char * name, HyPerCol * hc);
   int zeroSelfWeights(int numPatches, int arborId);
   virtual int normalizeWeights();
   // virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);

protected:
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
};

} /* namespace PV */
#endif /* NOSELFKERNELCONN_HPP_ */
