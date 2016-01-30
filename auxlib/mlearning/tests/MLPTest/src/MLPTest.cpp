#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include <MLearningGroupHandler.hpp>
#include "CustomGroupHandler.hpp"

//int main(int argc, char * argv[]) {
//   int paramfilestatus = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
//   int cl_argc = argc + (paramfilestatus!=0 ? 2 : 0);
//   char ** cl_argv = (char **) malloc((size_t) (cl_argc+1) * sizeof(char *));
//   assert(cl_argv!=NULL);
//   for (int a=0; a<argc; a++) {
//      cl_argv[a] = strdup(argv[a]);
//      assert(cl_argv[a]);
//   }
//   if (paramfilestatus!=0) {
//      cl_argv[argc] = strdup("-p");
//      assert(cl_argv[argc]);
//      cl_argv[argc+1] = strdup("input/MatchingPursuitTest.params");
//      assert(cl_argv[argc+1]);
//   }
//   cl_argv[cl_argc] = NULL;
//
//   PV::ParamGroupHandler * customGroupHandlers[2];
//   customGroupHandlers[0] = new PVMatchingPursuit::MatchingPursuitGroupHandler;
//   customGroupHandlers[1] = new CustomGroupHandler;
//   int status = buildandrun(cl_argc, cl_argv, NULL, NULL, customGroupHandlers, 2);
//   for (int a=0; a<cl_argc; a++) {
//      free(cl_argv[a]); cl_argv[a]=NULL;
//   }
//   free(cl_argv); cl_argv = NULL;
//   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
//}

/*
 *  * main.cpp for MLPTest
 *   *
 *    */


#include <columns/buildandrun.hpp>
#include "ComparisonLayer.hpp"
#include "InputLayer.hpp"
#include "GTLayer.hpp"

int main(int argc, char * argv[]) {
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   int rank = initObj->getWorldRank();

   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", argv[0]);
         fprintf(stderr, "This test hard-codes the necessary params file.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   PV::ParamGroupHandler * customGroupHandlers[2];
   customGroupHandlers[0] = new PVMLearning::MLearningGroupHandler;
   customGroupHandlers[1] = new PVMLearning::CustomGroupHandler;

   PV_Arguments * arguments = initObj->getArguments();

   int status = PV_SUCCESS;

   const char* paramsFile = "input/GradientCheckW1.params";
   arguments->setParamsFile(paramsFile);
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/GradientCheckW2.params";
   arguments->setParamsFile(paramsFile);
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/MLPTrain.params";
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/MLPTest.params";
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
      exit(status);
   }

   paramsFile = "input/AlexTrain.params";
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/AlexTest.params";
   status = rebuildandrun(initObj, NULL, NULL, customGroupHandlers, 2);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

//void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
//   void* addedGroup= NULL;
//   if ( !strcmp(keyword, "ComparisonLayer") ) {
//      addedGroup = new ComparisonLayer(groupname, hc);
//   }
//   if ( !strcmp(keyword, "InputLayer") ) {
//      addedGroup = new InputLayer(groupname, hc);
//   }
//   if ( !strcmp(keyword, "GTLayer") ) {
//      addedGroup = new GTLayer(groupname, hc);
//   }
//   if (!addedGroup) {
//      fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
//      exit(EXIT_SUCCESS);
//   }
//   checknewobject((void *) addedGroup, keyword, groupname, hc);
//   return addedGroup;
//}
