/*
 * RandomPatchMovieProbe.cpp
 *
 *  Author: Pete Schultz
 */

#include "RandomPatchMovieProbe.hpp"

namespace PV {

RandomPatchMovieProbe::RandomPatchMovieProbe(const char * filename, HyPerLayer * layer, const char * probename /*default NULL*/)
      : LayerProbe()
{
   initRandomPatchMovieProbe(filename, layer, probename);
}  // end RandomPatchMovieProbe(const char *, HyPerCol *)

RandomPatchMovieProbe::~RandomPatchMovieProbe()
{
   free(name);
}  // end ~RandomPatchMovieProbe()

int RandomPatchMovieProbe::initRandomPatchMovieProbe(const char * filename, HyPerLayer * layer, const char * probename) {
   int status = initLayerProbe(filename, layer);
   if( probename == NULL )
      name = strdup("");
   else
      name = strdup(probename);
   if( name == NULL ) status = PV_FAILURE;
   return status;
}

int RandomPatchMovieProbe::outputState(float timef) {
#ifdef PV_USE_MPI
   int rank = getTargetLayer()->getParent()->icCommunicator()->commRank();
   if( rank != 0) return PV_SUCCESS;
#endif // PV_USE_MPI
   RandomPatchMovie * rpm = dynamic_cast<RandomPatchMovie *>(getTargetLayer());
   if( rpm == NULL ) {
      fprintf(stderr, "RandomPatchMovieProbe: Layer \"%s\" is not a RandomPatchMovie.", getTargetLayer()->getName());
      return PV_FAILURE;
   }
   if( timef == 0 ) {
      displayPeriod = rpm->getDisplayPeriod();
      nextDisplayTime = 0;
   }
   if( timef >= nextDisplayTime ) {
      nextDisplayTime += displayPeriod;
      const PVLayerLoc * loc = rpm->getLayerLoc();
      fprintf(fp, "RandomPatchMovie \"%s\": Time %f, Offset (%d,%d), Patch size (%d,%d), File \"%s\"\n", name, timef, rpm->getOffsetX(), rpm->getOffsetY(), loc->nxGlobal, loc->nyGlobal, rpm->getFilename());
   }
   return PV_SUCCESS;
}  // end outputState(float, HyPerLayer *)

}  // end namespace PV


