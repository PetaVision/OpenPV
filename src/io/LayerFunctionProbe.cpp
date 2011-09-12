/*
 * LayerFunctionProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunctionProbe.hpp"

namespace PV {

LayerFunctionProbe::LayerFunctionProbe(const char * msg) :
   StatsProbe(BufV, msg) {
   initialize(NULL);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, HyPerCol * hc, const char * msg) :
   StatsProbe(filename, hc, BufV, msg) {
   initialize(NULL);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, const char *)

LayerFunctionProbe::LayerFunctionProbe(const char * msg, LayerFunction * F) :
   StatsProbe(BufV, msg) {
   initialize(F);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, LayerFunction *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, HyPerCol * hc, const char * msg, LayerFunction * F) :
   StatsProbe(filename, hc, BufV, msg) {
   initialize(F);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, const char *, LayerFunction *)

int LayerFunctionProbe::initialize(LayerFunction * F) {
   int status = F==NULL ? PV_SUCCESS : setFunction(F);
   return status;
}

int LayerFunctionProbe::setFunction(LayerFunction * F) {
   LayerFunction * Fcheck = dynamic_cast<LayerFunction *>(F);
   function = Fcheck;
   if( Fcheck != NULL) {
      return PV_SUCCESS;
   }
   else {
      fprintf(stderr,"LayerFunctionProbe \"%s\" specified LayerFunction is not valid.\n", msg );
      return PV_FAILURE;
   }
}

int LayerFunctionProbe::outputState(float time, HyPerLayer * l) {
   pvdata_t val = function->evaluate(time, l);
#ifdef PV_USE_MPI
   if( l->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   if( function ) {
      return writeState(time, l, val);
   }
   else {
      fprintf(stderr, "LayerFunctionProbe \"%s\" for layer %s: function has not been set\n", msg, l->getName());
      return PV_FAILURE;
   }
}  // end LayerFunctionProbe::outputState(float, HyPerLayer *)

int LayerFunctionProbe::writeState(float time, HyPerLayer * l, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int printstatus = fprintf(fp, "%st = %6.3f numNeurons = %8d Value            = %f\n", msg, time, l->getNumGlobalNeurons(), value);
   return printstatus > 0 ? PV_SUCCESS : PV_FAILURE;
}

}  // end namespace PV
