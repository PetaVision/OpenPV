/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   // If command line args specify the params file, use that file; otherwise use input/CloneVLayerTest.params
   int paramfilestatus = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
   int cl_argc = argc + (paramfilestatus!=0 ? 2 : 0);
   char ** cl_argv = (char **) malloc((size_t) (cl_argc+1) * sizeof(char *));
   assert(cl_argv!=NULL);
   for (int a=0; a<argc; a++) {
      cl_argv[a] = strdup(argv[a]);
      assert(cl_argv[a]);
   }
   if (paramfilestatus!=0) {
      cl_argv[argc] = strdup("-p");
      assert(cl_argv[argc]);
      cl_argv[argc+1] = strdup("input/CloneVLayerTest.params");
      assert(cl_argv[argc+1]);
   }
   cl_argv[cl_argc] = NULL;

   int status;
   status = buildandrun(cl_argc, cl_argv, NULL, &customexit, NULL);
   for (int a=0; a<cl_argc; a++) {
      free(cl_argv[a]);
   }
   free(cl_argv); cl_argv = NULL;
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
