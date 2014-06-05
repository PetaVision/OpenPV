/*
 * KernelConn.cpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#include "KernelConn.hpp"

namespace PV {

KernelConn::KernelConn()
{
}

KernelConn::KernelConn(const char * name, HyPerCol * hc) : HyPerConn() {
   if (hc->columnId()==0) {
      fprintf(stderr, "KernelConn \"%s\" warning: class KernelConn is deprecated.  Instead use HyPerConn with parameter sharedWeights set to true.\n", name);
   }
   HyPerConn::initialize(name, hc);
}

KernelConn::~KernelConn() {
}

void KernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

} // namespace PV

