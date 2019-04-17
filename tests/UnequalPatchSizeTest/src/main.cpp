/*
 * pv.cpp
 *
 */

#include "columns/ComponentBasedObject.hpp"
#include "columns/buildandrun.hpp"
#include "components/PatchSize.hpp"
#include "layers/HyPerLayer.hpp"
#include "probes/RequireAllZeroActivityProbe.hpp"

int customexit(HyPerCol *hc, int argc, char *argv[]);
int correctHaloSize(int patchsize, int nPre, int nPost);

int main(int argc, char *argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   // Make sure comparison layer is all zeros
   auto *layer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("compare"));
   auto *probe = dynamic_cast<RequireAllZeroActivityProbe *>(hc->getObjectFromName("check_output"));
   FatalIf(probe->getNonzeroFound(), "%s contains a nonzero value.\n", layer->getName());

   // Check halo of input layer
   auto *inlayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("input"));
   FatalIf(!inlayer, "Unable to find layer \"input\".\n");
   auto *outlayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("output"));
   FatalIf(!outlayer, "Unable to find layer \"output\".\n");
   auto *conn = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("input_to_output"));

   FatalIf(conn == nullptr, "Unable to find connection \"input_to_output\".\n");

   int nxp       = conn->getComponentByType<PatchSize>()->getPatchSizeX();
   int nxPre     = inlayer->getLayerLoc()->nx;
   int nxPost    = outlayer->getLayerLoc()->nx;
   int xHaloSize = correctHaloSize(nxp, nxPre, nxPost);

   int nyp       = conn->getComponentByType<PatchSize>()->getPatchSizeY();
   int nyPre     = inlayer->getLayerLoc()->ny;
   int nyPost    = outlayer->getLayerLoc()->ny;
   int yHaloSize = correctHaloSize(nyp, nyPre, nyPost);

   FatalIf(inlayer->getLayerLoc()->halo.lt != xHaloSize, "Halo incorrect size on left side.\n");
   FatalIf(inlayer->getLayerLoc()->halo.rt != xHaloSize, "Halo incorrect size on right side.\n");
   FatalIf(inlayer->getLayerLoc()->halo.dn != yHaloSize, "Halo incorrect size on bottom side.\n");
   FatalIf(inlayer->getLayerLoc()->halo.up != yHaloSize, "Halo incorrect size on top side.\n");

   if (hc->columnId() == 0) {
      InfoLog().printf("Success.\n");
   }
   return 0;
}

int correctHaloSize(int patchsize, int nPre, int nPost) {
   int haloSize;
   if (nPost > nPre) { // one-to-many connection
      int many = nPost / nPre;
      FatalIf(many * nPre != nPost, "Test failed.\n");
      FatalIf(patchsize % many != 0, "Test failed.\n");
      int numcells = patchsize / many;
      FatalIf(numcells % 2 != 1, "Test failed.\n");
      haloSize = (numcells - 1) / 2;
   }
   else if (nPost < nPre) { // many-to-one connection
      int many = nPre / nPost;
      FatalIf(many * nPost != nPre, "Test failed.\n");
      FatalIf(patchsize % 2 != 1, "Test failed.\n");
      haloSize = many * (patchsize - 1) / 2;
   }
   else {
      FatalIf(nPost != nPre, "Test failed.\n");
      FatalIf(patchsize % 2 != 1, "Test failed.\n");
      haloSize = (patchsize - 1) / 2;
   }
   return haloSize;
}
