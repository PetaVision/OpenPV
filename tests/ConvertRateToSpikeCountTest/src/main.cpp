/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, customexit);
   if (status == PV_SUCCESS) {
      printf("%s succeeded.\n", argv[0]);
   }
   else {
      fprintf(stderr, "%s failed with return code %d.\n", argv[0], status);
   }
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   HyPerLayer * layer = hc->getLayerFromName("comparison");
   assert(layer!=NULL);
   assert(layer->getNumProbes()>0);
   RequireAllZeroActivityProbe * probe = NULL;
   for (int k=0; k<layer->getNumProbes(); k++) {
      LayerProbe * p = layer->getProbe(k);
      probe = dynamic_cast<RequireAllZeroActivityProbe *>(p);
      if (probe != NULL) { break; }
   }
   assert(probe!=NULL);
   int status = PV_SUCCESS;
   if (probe->getNonzeroFound()) {
      if (hc->columnId()==0) {
         fprintf(stderr, "comparison layer had a nonzero activity at time %f\n", probe->getNonzeroTime());
         status = PV_FAILURE;
      }
   }
   return status;
}
