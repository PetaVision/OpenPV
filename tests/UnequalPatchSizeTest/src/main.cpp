/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);
int correctHaloSize(int patchsize, int nPre, int nPost);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   // Make sure comparison layer is all zeros
   HyPerLayer * layer = hc->getLayerFromName("compare");
   pvErrorIf(!(layer->getNumProbes()==1), "Test failed.\n");
   LayerProbe * probe = layer->getProbe(0);
   RequireAllZeroActivityProbe * rProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   pvErrorIf(!(!rProbe->getNonzeroFound()), "Test failed.\n");
   
   // Check halo of input layer
   HyPerLayer * inlayer = hc->getLayerFromName("input");
   pvErrorIf(!(inlayer), "Test failed.\n");
   HyPerLayer * outlayer = hc->getLayerFromName("output");
   pvErrorIf(!(outlayer), "Test failed.\n");
   BaseConnection * baseConn = hc->getConnFromName("input_to_output");
   HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);

   pvErrorIf(!(conn), "Test failed.\n");
   
   int nxp = conn->xPatchSize();
   int nxPre = inlayer->getLayerLoc()->nx;
   int nxPost = outlayer->getLayerLoc()->nx;
   int xHaloSize = correctHaloSize(nxp, nxPre, nxPost);

   int nyp = conn->yPatchSize();
   int nyPre = inlayer->getLayerLoc()->ny;
   int nyPost = outlayer->getLayerLoc()->ny;
   int yHaloSize = correctHaloSize(nyp, nyPre, nyPost);
   
   pvErrorIf(!(inlayer->getLayerLoc()->halo.lt == xHaloSize), "Test failed.\n");
   pvErrorIf(!(inlayer->getLayerLoc()->halo.rt == xHaloSize), "Test failed.\n");
   pvErrorIf(!(inlayer->getLayerLoc()->halo.dn == yHaloSize), "Test failed.\n");
   pvErrorIf(!(inlayer->getLayerLoc()->halo.up == yHaloSize), "Test failed.\n");
   
   if (hc->columnId()==0) { pvInfo().printf("Success.\n"); }
   return 0;
}

int correctHaloSize(int patchsize, int nPre, int nPost) {
   int haloSize;
   if (nPost > nPre) { // one-to-many connection
      int many = nPost/nPre;
      pvErrorIf(!(many * nPre == nPost), "Test failed.\n");
      pvErrorIf(!(patchsize % many == 0), "Test failed.\n");
      int numcells = patchsize/many;
      pvErrorIf(!(numcells % 2 == 1), "Test failed.\n");
      haloSize = (numcells-1)/2;
   }
   else if (nPost < nPre) { // many-to-one connection
      int many = nPre/nPost;
      pvErrorIf(!(many * nPost == nPre), "Test failed.\n");
      pvErrorIf(!(patchsize % 2 == 1), "Test failed.\n");
      haloSize = many * (patchsize-1)/2;
   }
   else {
      pvErrorIf(!(nPost==nPre), "Test failed.\n");
      pvErrorIf(!(patchsize % 2 == 1), "Test failed.\n");
      haloSize = (patchsize-1)/2;
   }
   return haloSize;
}
