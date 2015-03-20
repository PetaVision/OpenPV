/*
 * main.cpp for MLPTest
 *
 */


#include <columns/buildandrun.hpp>
#include "ComparisonLayer.hpp"
#include "InputLayer.hpp"
#include "GTLayer.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int rank;
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI

   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", argv[0]);
         fprintf(stderr, "This test hard-codes the necessary params file.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   int pv_argc = argc+2; // input arguments, plus "-p", plus params file argument
   char ** pv_argv = (char **) malloc((pv_argc+1)*sizeof(char *));
   assert(pv_argv);
   int pv_arg=0;
   for (pv_arg = 0; pv_arg < argc; pv_arg++) {
      pv_argv[pv_arg] = strdup(argv[pv_arg]);
      assert(pv_argv[pv_arg]);
   }
   assert(pv_arg==argc);
   pv_argv[pv_arg] = strdup("-p");
   assert(pv_argv[pv_arg]!=NULL);
   int paramFileArgIndex = pv_arg+1;
   pv_argv[paramFileArgIndex] = NULL; // this will hold params filename
   pv_argv[pv_argc]=NULL;

   int status = PV_SUCCESS;

   free(pv_argv[paramFileArgIndex]);
   pv_argv[paramFileArgIndex] = strdup("input/MLPTrain.params");
   assert(pv_argv[paramFileArgIndex]);
   status = buildandrun(pv_argc, pv_argv, NULL, NULL, &addcustomgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", pv_argv[0], pv_argv[2], status);
   }
   pv_argv[paramFileArgIndex] = strdup("input/MLPTest.params");
   assert(pv_argv[paramFileArgIndex]);
   status = buildandrun(pv_argc, pv_argv, NULL, NULL, &addcustomgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", pv_argv[0], pv_argv[2], status);
      exit(status);
   }

   free(pv_argv[paramFileArgIndex]);
   pv_argv[2] = strdup("input/AlexTrain.params");
   assert(pv_argv[paramFileArgIndex]);
   status = buildandrun(pv_argc, pv_argv, NULL, NULL, &addcustomgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", pv_argv[0], pv_argv[2], status);
   }

   free(pv_argv[paramFileArgIndex]);
   pv_argv[paramFileArgIndex] = strdup("input/AlexTest.params");
   assert(pv_argv[paramFileArgIndex]);
   status = buildandrun(pv_argc, pv_argv, NULL, NULL, &addcustomgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", pv_argv[0], pv_argv[2], status);
   }

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif

   for (int i=0; i<pv_argc; i++) {
      free(pv_argv[i]);
   }
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   void* addedGroup= NULL;
   if ( !strcmp(keyword, "ComparisonLayer") ) {
      addedGroup = new ComparisonLayer(groupname, hc);
   }
   if ( !strcmp(keyword, "InputLayer") ) {
      addedGroup = new InputLayer(groupname, hc);
   }
   if ( !strcmp(keyword, "GTLayer") ) {
      addedGroup = new GTLayer(groupname, hc);
   }
   if (!addedGroup) {
      fprintf(stderr, "Group \"%s\": Unable to create %s\n", groupname, keyword);
      exit(EXIT_SUCCESS);
   }
   checknewobject((void *) addedGroup, keyword, groupname, hc);
   return addedGroup;
}
