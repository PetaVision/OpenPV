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
   assert(layer->getNumProbes()==1);
   LayerProbe * probe = layer->getProbe(0);
   RequireAllZeroActivityProbe * rProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   assert(!rProbe->getNonzeroFound());
   
   // Check halo of input layer
   HyPerLayer * inlayer = hc->getLayerFromName("input");
   assert(inlayer);
   HyPerLayer * outlayer = hc->getLayerFromName("output");
   assert(outlayer);
   BaseConnection * baseConn = hc->getConnFromName("input_to_output");
   HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);

   assert(conn);
   
   int nxp = conn->xPatchSize();
   int nxPre = inlayer->getLayerLoc()->nx;
   int nxPost = outlayer->getLayerLoc()->nx;
   int xHaloSize = correctHaloSize(nxp, nxPre, nxPost);

   int nyp = conn->yPatchSize();
   int nyPre = inlayer->getLayerLoc()->ny;
   int nyPost = outlayer->getLayerLoc()->ny;
   int yHaloSize = correctHaloSize(nyp, nyPre, nyPost);
   
   assert(inlayer->getLayerLoc()->halo.lt == xHaloSize);
   assert(inlayer->getLayerLoc()->halo.rt == xHaloSize);
   assert(inlayer->getLayerLoc()->halo.dn == yHaloSize);
   assert(inlayer->getLayerLoc()->halo.up == yHaloSize);
   
   if (hc->columnId()==0) { printf("Success.\n"); }
   return 0;
}

int correctHaloSize(int patchsize, int nPre, int nPost) {
   int haloSize;
   if (nPost > nPre) { // one-to-many connection
      int many = nPost/nPre;
      assert(many * nPre == nPost);
      assert(patchsize % many == 0);
      int numcells = patchsize/many;
      assert(numcells % 2 == 1);
      haloSize = (numcells-1)/2;
   }
   else if (nPost < nPre) { // many-to-one connection
      int many = nPre/nPost;
      assert(many * nPost == nPre);
      assert(patchsize % 2 == 1);
      haloSize = many * (patchsize-1)/2;
   }
   else {
      assert(nPost==nPre);
      assert(patchsize % 2 == 1);
      haloSize = (patchsize-1)/2;
   }
   return haloSize;
}
