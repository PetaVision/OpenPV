/*
 * pv.cpp
 *
 */

#include <columns/ComponentBasedObject.hpp>
#include <columns/buildandrun.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/InternalStateBuffer.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   auto *outputLayer = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("Output"));
   FatalIf(outputLayer == nullptr, "Test failed.\n");
   auto *activityComponent  = outputLayer->getComponentByType<ActivityComponent>();
   auto *internalState      = activityComponent->getComponentByType<InternalStateBuffer>();
   float const *V           = internalState->getBufferData();
   auto *publisherComponent = outputLayer->getComponentByType<BasePublisherComponent>();
   float const *A           = publisherComponent->getLayerData();
   PVLayerLoc const *loc    = publisherComponent->getLayerLoc();
   PVHalo const &halo       = loc->halo;
   int N                    = loc->nx * loc->ny * loc->nf;
   for (int k = 0; k < N; k++) {
      int kExt = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo.lt, halo.rt, halo.dn, halo.up);
      float v  = V[k];
      float a  = A[kExt];
      // Based on params file having verticesV = [0.5, 0.5]; verticesA = [0.0, 1.0];
      // slopeNegInf = 0.0; slopePosInf = 0.0;
      // i.e. indicator function of V>=0.5.
      // TODO: Currently, jumps in verticesV/verticesA are continuous from the right.  Need to
      // generalize.
      if (v < 0.5f) {
         FatalIf(a != 0.0f, "Test failed.\n");
      }
      else {
         FatalIf(a != 1.0f, "Test failed.\n");
      }
   }
   return PV_SUCCESS;
}
