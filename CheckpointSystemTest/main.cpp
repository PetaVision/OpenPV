/*
 * pv.cpp
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "CPTestInputLayer.hpp"
#include "VaryingKernelConn.hpp"
#include "VaryingHyPerConn.hpp"

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
int customexit(HyPerCol * hc, int argc, char * argv[]);


int main(int argc, char * argv[]) {

   if( argc > 1) {
      fprintf(stderr, "%s: run without input arguments; the necessary arguments are hardcoded.\n", argv[0]);
      exit(EXIT_FAILURE);
   }
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI

#undef REQUIRE_RETURN
#ifdef REQUIRE_RETURN
   int charhit;
   fflush(stdout);
   if( rank == 0 ) {
      printf("Hit enter to begin! ");
      fflush(stdout);
      charhit = getc(stdin);
   }
#ifdef PV_USE_MPI
   int ierr;
   ierr = MPI_Bcast(&charhit, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif // PV_USE_MPI
#endif // REQUIRE_RETURN

   int status;
   char * cl_args[3];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/CheckpointParameters1.params");
   status = buildandrun(3, cl_args, NULL, NULL, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
      exit(status);
   }

   free(cl_args[2]);
   cl_args[2] = strdup("input/CheckpointParameters2.params");
   status = buildandrun(3, cl_args, NULL, &customexit, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
      exit(status);
   }

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif // PV_USE_MPI

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   PVParams * params = hc->parameters();
   if( !strcmp(keyword, "CPTestInputLayer") ) {
      addedGroup = (void *) new CPTestInputLayer(name, hc);
   }
   if( !strcmp(keyword, "VaryingKernelConn") ) {
      HyPerLayer * preLayer;
      HyPerLayer * postLayer;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         InitWeights *weightInitializer;
         ChannelType channelType;
         int channelNo = (int) params->value(name, "channelCode", -1);
         if( decodeChannel( channelNo, &channelType ) != PV_SUCCESS) {
            fprintf(stderr, "Group \"%s\": Parameter group for class %s must set parameter channelCode.\n", name, keyword);
            return NULL;
         }
         weightInitializer = createInitWeightsObject(name, hc, preLayer, postLayer, channelType);
         if( weightInitializer == NULL ) {
            weightInitializer = getDefaultInitWeightsMethod(keyword);
         }
         const char * fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedGroup = (void * ) new VaryingKernelConn(name, hc, preLayer, postLayer, channelType, fileName, weightInitializer);
      }
   }
   if( !strcmp(keyword, "VaryingHyPerConn") ) {
      HyPerLayer * preLayer;
      HyPerLayer * postLayer;
      getPreAndPostLayers(name, hc, &preLayer, &postLayer);
      if( preLayer && postLayer ) {
         InitWeights *weightInitializer;
         ChannelType channelType;
         int channelNo = (int) params->value(name, "channelCode", -1);
         if( decodeChannel( channelNo, &channelType ) != PV_SUCCESS) {
            fprintf(stderr, "Group \"%s\": Parameter group for class %s must set parameter channelCode.\n", name, keyword);
            return NULL;
         }
         weightInitializer = createInitWeightsObject(name, hc, preLayer, postLayer, channelType);
         if( weightInitializer == NULL ) {
            weightInitializer = getDefaultInitWeightsMethod(keyword);
         }
         const char * fileName = getStringValueFromParameterGroup(name, params, "initWeightsFile", false);
         addedGroup = (void * ) new VaryingHyPerConn(name, hc, preLayer, postLayer, channelType, fileName, weightInitializer);
      }
   }
   return addedGroup;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status;
   int rank = hc->icCommunicator()->commRank();
   int rootproc = 0;
   if( rank == rootproc ) {
      int index = hc->parameters()->value("column", "numSteps");
      const char * cpdir1 = hc->parameters()->stringValue("column", "checkpointReadDir");
      const char * cpdir2 = hc->parameters()->stringValue("column", "checkpointWriteDir");
      if(cpdir1 == NULL || cpdir2 == NULL) {
         fprintf(stderr, "%s: unable to allocate memory for names of checkpoint directories", argv[0]);
         exit(EXIT_FAILURE);
      }
      char * shellcommand;
      char c;
      const char * fmtstr = "diff -r -q %s/Checkpoint%d %s/Checkpoint%d";
      int len = snprintf(&c, 1, fmtstr, cpdir1, index, cpdir2, index);
      shellcommand = (char *) malloc(len+1);
      if( shellcommand == NULL) {
         fprintf(stderr, "%s: unable to allocate memory for shell diff command.\n", argv[0]);
         exit(EXIT_FAILURE);
      }
      assert( snprintf(shellcommand, len+1, fmtstr, cpdir1, index, cpdir2, index) == len );
      status = system(shellcommand);
      if( status != 0 ) {
         fprintf(stderr, "system(\"%s\") returned %d\n", shellcommand, status);
         // Because system() seems to return the result of the shell command multiplied by 256,
         // and Unix only sees the 8 least-significant bits of the value returned by a C/C++ program,
         // simply returning the result of the system call doesn't work.
         // I haven't found the mult-by-256 behavior in the documentation, so I'm not sure what's
         // going on.
         return EXIT_FAILURE;
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
#endif // PV_USE_MPI
   return status;
}
