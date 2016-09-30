/*
 * KernelConn.cpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#include "KernelConn.hpp"

namespace PV {

KernelConn::KernelConn(
      const char *name,
      HyPerCol *hc,
      InitWeights *weightInitializer,
      NormalizeBase *weightNormalizer)
      : HyPerConn() {
   if (hc->columnId() == 0) {
      pvError().printf(
            "KernelConn \"%s\": class KernelConn is obsolete.  Instead use HyPerConn with "
            "parameter sharedWeights set to true.\n",
            name);
   }
   MPI_Barrier(hc->getCommunicator()->communicator());
   exit(EXIT_FAILURE);
}

} // namespace PV
