/*
 * pv.cpp
 *
 */

#include "columns/buildandrun.hpp"
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
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("compare"));
   FatalIf(!(layer->getNumProbes() == 1), "Test failed.\n");
   LayerProbe *probe                   = layer->getProbe(0);
   RequireAllZeroActivityProbe *rProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   FatalIf(!(!rProbe->getNonzeroFound()), "Test failed.\n");

   // Check halo of input layer
   HyPerLayer *inlayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("input"));
   FatalIf(!(inlayer), "Test failed.\n");
   HyPerLayer *outlayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("output"));
   FatalIf(!(outlayer), "Test failed.\n");
   HyPerConn *conn = dynamic_cast<HyPerConn *>(hc->getObjectFromName("input_to_output"));

   FatalIf(!(conn), "Test failed.\n");

   int nxp       = conn->xPatchSize();
   int nxPre     = inlayer->getLayerLoc()->nx;
   int nxPost    = outlayer->getLayerLoc()->nx;
   int xHaloSize = correctHaloSize(nxp, nxPre, nxPost);

   int nyp       = conn->yPatchSize();
   int nyPre     = inlayer->getLayerLoc()->ny;
   int nyPost    = outlayer->getLayerLoc()->ny;
   int yHaloSize = correctHaloSize(nyp, nyPre, nyPost);

   FatalIf(!(inlayer->getLayerLoc()->halo.lt == xHaloSize), "Test failed.\n");
   FatalIf(!(inlayer->getLayerLoc()->halo.rt == xHaloSize), "Test failed.\n");
   FatalIf(!(inlayer->getLayerLoc()->halo.dn == yHaloSize), "Test failed.\n");
   FatalIf(!(inlayer->getLayerLoc()->halo.up == yHaloSize), "Test failed.\n");

   if (hc->columnId() == 0) {
      InfoLog().printf("Success.\n");
   }
   return 0;
}

int correctHaloSize(int patchsize, int nPre, int nPost) {
   int haloSize;
   if (nPost > nPre) { // one-to-many connection
      int many = nPost / nPre;
      FatalIf(!(many * nPre == nPost), "Test failed.\n");
      FatalIf(!(patchsize % many == 0), "Test failed.\n");
      int numcells = patchsize / many;
      FatalIf(!(numcells % 2 == 1), "Test failed.\n");
      haloSize = (numcells - 1) / 2;
   }
   else if (nPost < nPre) { // many-to-one connection
      int many = nPre / nPost;
      FatalIf(!(many * nPost == nPre), "Test failed.\n");
      FatalIf(!(patchsize % 2 == 1), "Test failed.\n");
      haloSize = many * (patchsize - 1) / 2;
   }
   else {
      FatalIf(!(nPost == nPre), "Test failed.\n");
      FatalIf(!(patchsize % 2 == 1), "Test failed.\n");
      haloSize = (patchsize - 1) / 2;
   }
   return haloSize;
}
