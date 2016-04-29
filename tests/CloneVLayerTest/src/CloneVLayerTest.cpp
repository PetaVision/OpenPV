/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   PV_Arguments * arguments = initObj.getArguments();
   if (arguments->getParamsFile() == NULL) {
      arguments->setParamsFile("input/CloneVLayerTest.params");
   }

   int status;
   status = rebuildandrun(&initObj, NULL, &customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int check_clone_id = -1;
   int check_sigmoid_id = -1;
   for (int k=0; k<hc->numberOfLayers(); k++) {
      if (!strcmp(hc->getLayer(k)->getName(), "CheckClone")) {
         assert(check_clone_id<0);
         check_clone_id = k;
      }
      if (!strcmp(hc->getLayer(k)->getName(), "CheckSigmoid")) {
         assert(check_sigmoid_id<0);
         check_sigmoid_id = k;
      }
   }

   int N;

   HyPerLayer * check_clone_layer = hc->getLayer(check_clone_id);
   assert(check_clone_layer!=NULL);
   const pvdata_t * check_clone_layer_data = check_clone_layer->getLayerData();
   N = check_clone_layer->getNumExtended();

   for (int k=0; k<N; k++) {
      assert(fabsf(check_clone_layer_data[k])<1e-6);
   }

   HyPerLayer * check_sigmoid_layer = hc->getLayer(check_sigmoid_id);
   assert(check_sigmoid_layer!=NULL);
   const pvdata_t * check_sigmoid_layer_data = check_sigmoid_layer->getLayerData();
   N = check_sigmoid_layer->getNumExtended();

   for (int k=0; k<N; k++) {
      assert(fabsf(check_sigmoid_layer_data[k])<1e-6);
   }

   if (hc->columnId()==0) {
      printf("%s passed.\n", argv[0]);
   }
   return PV_SUCCESS;
}
