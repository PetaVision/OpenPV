/*
 * RandomPatchMovieProbe.cpp
 *
 *  Author: Pete Schultz
 */

#include "RandomPatchMovieProbe.hpp"

namespace PV {

RandomPatchMovieProbe::RandomPatchMovieProbe(const char * probename, HyPerCol * hc)
      : LayerProbe()
{
   initRandomPatchMovieProbe(probename, hc);
}  // end RandomPatchMovieProbe(const char *, HyPerCol *)

RandomPatchMovieProbe::~RandomPatchMovieProbe()
{
   free(name);
}  // end ~RandomPatchMovieProbe()

int RandomPatchMovieProbe::initRandomPatchMovieProbe(const char * probename, HyPerCol * hc) {
   int status = initLayerProbe(probename, hc);
   return status;
}

int RandomPatchMovieProbe::outputState(double timed) {
#ifdef PV_USE_MPI
   int rank = getTargetLayer()->getParent()->icCommunicator()->commRank();
   if( rank != 0) return PV_SUCCESS;
#endif // PV_USE_MPI
   RandomPatchMovie * rpm = dynamic_cast<RandomPatchMovie *>(getTargetLayer());
   if( rpm == NULL ) {
      fprintf(stderr, "RandomPatchMovieProbe: Layer \"%s\" is not a RandomPatchMovie.", getTargetLayer()->getName());
      return PV_FAILURE;
   }
   if( timed == 0 ) {
      displayPeriod = rpm->getDisplayPeriod();
      nextDisplayTime = 0;
   }
   if( timed >= nextDisplayTime ) {
      nextDisplayTime += displayPeriod;
      const PVLayerLoc * loc = rpm->getLayerLoc();
      fprintf(outputstream->fp, "RandomPatchMovie \"%s\": Time %f, Offset (%d,%d), Patch size (%d,%d), File \"%s\"\n", name, timed, rpm->getOffsetX(), rpm->getOffsetY(), loc->nxGlobal, loc->nyGlobal, rpm->getFilename());
   }
   return PV_SUCCESS;
}  // end outputState(float, HyPerLayer *)

}  // end namespace PV


