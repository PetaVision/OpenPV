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
      fprintf(stderr, "%s \"%s\": CloneKernelConn has been deprecated.  Use CloneConn.\n", this->getKeyword(), name);
   }
   return status;
}

int CloneKernelConn::communicateInitInfo() {
   int status = CloneConn::communicateInitInfo();
   if (originalConn->usingSharedWeights()==false) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": originalConn \"%s\" does not use shared weights but CloneKernelConn assumes shared weights.\n",
               this->getKeyword(), name, originalConn->getName());
         fprintf(stderr, "Use CloneConn instead of CloneKernelConn.\n");
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

CloneKernelConn::~CloneKernelConn() {
}

BaseObject * createCloneKernelConn(char const * name, HyPerCol * hc) {
   return hc ? new CloneKernelConn(name, hc) : NULL;
}

} // end namespace PV
