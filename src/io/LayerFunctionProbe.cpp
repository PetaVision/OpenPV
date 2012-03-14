/*
 * LayerFunctionProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunctionProbe.hpp"

namespace PV {

LayerFunctionProbe::LayerFunctionProbe(HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initLayerFunctionProbe(NULL, layer, msg, NULL);
}  // end LayerFunctionProbe::LayerFunctionProbe(HyPerLayer *, const char *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : StatsProbe()
{
   initLayerFunctionProbe(filename, layer, msg, NULL);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, HyPerLayer *, const char *)

LayerFunctionProbe::LayerFunctionProbe(HyPerLayer * layer, const char * msg, LayerFunction * F)
   : StatsProbe()
{
   initLayerFunctionProbe(NULL, layer, msg, F);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, LayerFunction *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg, LayerFunction * F)
   : StatsProbe()
{
   initLayerFunctionProbe(filename, layer, msg, F);
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, const char *, LayerFunction *)

LayerFunctionProbe::LayerFunctionProbe()
   : StatsProbe()
{
   // Derived classes should call LayerFunctionProbe::initLayerFunctionProbe
}

LayerFunctionProbe::~LayerFunctionProbe() {
   delete function;
}

int LayerFunctionProbe::initLayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg, LayerFunction * F) {
   initStatsProbe(filename, layer, BufV, msg);
   F == NULL ? PV_SUCCESS : setFunction(F);
   return PV_SUCCESS;
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

int LayerFunctionProbe::outputState(float timef) {
   pvdata_t val = function->evaluate(timef, getTargetLayer());
#ifdef PV_USE_MPI
   if( getTargetLayer()->getParent()->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   if( function ) {
      return writeState(timef, getTargetLayer(), val);
   }
   else {
      fprintf(stderr, "LayerFunctionProbe \"%s\" for layer %s: function has not been set\n", msg, getTargetLayer()->getName());
      return PV_FAILURE;
   }
}  // end LayerFunctionProbe::outputState(float, HyPerLayer *)

int LayerFunctionProbe::writeState(float timef, HyPerLayer * l, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(l->getParent()->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int printstatus = fprintf(fp, "%st = %6.3f numNeurons = %8d Value            = %f\n", msg, timef, l->getNumGlobalNeurons(), value);
   return printstatus > 0 ? PV_SUCCESS : PV_FAILURE;
}

}  // end namespace PV
