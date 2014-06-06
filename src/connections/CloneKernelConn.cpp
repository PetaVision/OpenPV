/* CloneKernelConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneKernelConn.hpp"

namespace PV {

CloneKernelConn::CloneKernelConn() : CloneConn() {
}

CloneKernelConn::CloneKernelConn(const char * name, HyPerCol * hc) {
   initialize(name, hc);
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc) {
   int status = CloneConn::initialize(name, hc);
   if (hc->columnId()==0) {
      fprintf(stderr, "%s \"%s\": CloneKernelConn has been deprecated.  Use CloneConn.\n", hc->parameters()->groupKeywordFromName(name), name);
   }
   return status;
}

int CloneKernelConn::communicateInitInfo() {
   int status = CloneConn::communicateInitInfo();
   if (originalConn->usingSharedWeights()==false) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": originalConn \"%s\" does not use shared weights but CloneKernelConn assumes shared weights.\n",
               parent->parameters()->groupKeywordFromName(name), name, originalConn->getName());
         fprintf(stderr, "Use CloneConn instead of CloneKernelConn.\n");
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

CloneKernelConn::~CloneKernelConn() {
}

} // end namespace PV
