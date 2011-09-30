/*
 * RandomPatchMovieProbe.cpp
 *
 *  Author: Pete Schultz
 */

#include "RandomPatchMovieProbe.hpp"

namespace PV {

RandomPatchMovieProbe::RandomPatchMovieProbe(const char * filename, HyPerCol * hc, const char * probename /*default NULL*/)
      : LayerProbe(filename, hc)
{
   initialize(probename);
}  // end RandomPatchMovieProbe(const char *, HyPerCol *)

RandomPatchMovieProbe::~RandomPatchMovieProbe()
{
   free(name);
}  // end ~RandomPatchMovieProbe()

int RandomPatchMovieProbe::initialize(const char * probename) {
   if( probename == NULL )
      name = strdup("");
   else
      name = strdup(probename);
   return name == NULL ? PV_FAILURE : PV_SUCCESS;
}

int RandomPatchMovieProbe::outputState(float time, HyPerLayer * l) {
#ifdef PV_USE_MPI
   int rank = l->getParent()->icCommunicator()->commRank();
   if( rank != 0) return PV_SUCCESS;
#endif // PV_USE_MPI
   RandomPatchMovie * rpm = dynamic_cast<RandomPatchMovie *>(l);
   if( rpm == NULL ) {
      fprintf(stderr, "RandomPatchMovieProbe: Layer \"%s\" is not a RandomPatchMovie.", l->getName());
      return PV_FAILURE;
   }
   if( time == 0 ) {
      displayPeriod = rpm->getDisplayPeriod();
      nextDisplayTime = 0;
   }
   if( time >= nextDisplayTime ) {
      nextDisplayTime += displayPeriod;
      const PVLayerLoc * loc = l->getLayerLoc();
      fprintf(fp, "RandomPatchMovie \"%s\": Time %f, Offset (%d,%d), Patch size (%d,%d), File \"%s\"\n", name, time, rpm->getOffsetX(), rpm->getOffsetY(), loc->nxGlobal, loc->nyGlobal, rpm->getFilename());
   }
   return PV_SUCCESS;
}  // end outputState(float, HyPerLayer *)

}  // end namespace PV


