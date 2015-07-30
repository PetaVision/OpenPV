/*
 * LayerFunction.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunction.hpp"

namespace PV {

LayerFunction::LayerFunction(const char * name) {
   this->name = NULL;
   setName(name);
}

LayerFunction::~LayerFunction() {
   free(name);
}

void LayerFunction::setName(const char * name) {
   size_t len = strlen(name);
   if( this->name ) {
      free( this->name );
   }
   this->name = (char *) malloc( (len+1)*sizeof(char) );
   if( this->name) {
      strcpy(this->name, name);
   }
}

pvdata_t LayerFunction::evaluate(float time, HyPerLayer * l, int batchIdx) {
   pvdata_t value = evaluateLocal(time, l, batchIdx);
#ifdef PV_USE_MPI
   value = functionReduce(value, l);
#endif // PV_USE_MPI
   return value;
}

#ifdef PV_USE_MPI
pvdata_t LayerFunction::functionReduce(pvdata_t localValue, HyPerLayer * l) {
   InterColComm * icComm = l->getParent()->icCommunicator();
   MPI_Comm comm = icComm->communicator();
   int rank = icComm->commRank();
   double value = (double) localValue;
   double reduced;
   int ierr;
   ierr = MPI_Reduce(&value, &reduced, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
   return rank == 0 ? (pvdata_t) reduced : 0.0;
}
#endif // PV_USE_MPI

}  // end namespace PV
