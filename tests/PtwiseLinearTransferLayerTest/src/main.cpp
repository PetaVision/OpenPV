/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * outputLayer = hc->getLayer(1);
   assert(!strcmp(outputLayer->getName(), "Output"));
   float const * V = outputLayer->getV();
   float const * A = outputLayer->getLayerData();
   PVLayerLoc const * loc = outputLayer->getLayerLoc();
   PVHalo const * halo = &loc->halo;
   int N = outputLayer->getNumNeurons();
   for (int k=0; k<N; k++) {
      int kExt = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
      float v = V[k];
      float a = A[kExt];
      // Based on params file having verticesV = [0.5, 0.5]; verticesA = [0.0, 1.0]; slopeNegInf = 0.0; slopePosInf = 0.0;
      // i.e. indicator function of V>=0.5.
      // TODO: Currently, PtwiseLinearTransferLayer does continuity from right at jumps.  Need to generalize.
      if (v < 0.5) {
         assert(a==0.0);
      }
      else {
         assert(a==1.0);
      }
   }
   return PV_SUCCESS;
}
