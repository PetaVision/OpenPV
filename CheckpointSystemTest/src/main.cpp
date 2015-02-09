/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "CPTestInputLayer.hpp"
#include "VaryingHyPerConn.hpp"

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
int customexit(HyPerCol * hc, int argc, char * argv[]);


int main(int argc, char * argv[]) {
   int rank = 0;
   bool argerr = false;
   int reqrtn = 0;
   int usethreads = 0;
   int threadargno = -1;
   for (int k=1; k<argc; k++) {
      if (!strcmp(argv[k], "--require-return")) {
         reqrtn = 1;
      }
      else if (!strcmp(argv[k], "-t")) {
         usethreads = 1;
         if (k<argc-1 && argv[k+1][0] != '-') {
            k++;
            threadargno = k;
         }
      }
      else {
         argerr = true;
         break;
      }
   }
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (argerr) {
      if (rank==0) {
         fprintf(stderr, "%s: run without input arguments (except for --require-return); the necessary arguments are hardcoded.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(EXIT_FAILURE);
   }
#else // PV_USE_MPI
   if (argerr) {
      fprintf(stderr, "%s: run without input arguments (except for --require-return); the necessary arguments are hardcoded.\n", argv[0]);
      exit(EXIT_FAILURE);
   }
#endif // PV_USE_MPI

   int status = 0;
   if (rank==0) {
      char const * rmcommand = "rm -rf checkpoints1 checkpoints2 output";
      status = system(rmcommand);
      if (status != 0) {
         fprintf(stderr, "deleting old checkpoints and output directories failed: \"%s\" returned %d\n", rmcommand, status);
         exit(EXIT_FAILURE);
      }
   }

   assert(reqrtn==0 || reqrtn==1);
   assert(usethreads==0 || usethreads==1);
   size_t cl_argc = 3+reqrtn+usethreads+(threadargno>0);
   char ** cl_args = (char **) malloc((cl_argc+1)*sizeof(char *));
   assert(cl_args!=NULL);
   int cl_arg = 0;
   cl_args[cl_arg++] = strdup(argv[0]);
   cl_args[cl_arg++] = strdup("-p");
   cl_args[cl_arg++] = strdup("input/CheckpointParameters1.params");
   if (reqrtn) {
      cl_args[cl_arg++] = strdup("--require-return");
   }
   if (usethreads) {
      cl_args[cl_arg++] = strdup("-t");
      if (threadargno>0) {
         cl_args[cl_arg++] = strdup(argv[threadargno]);
      }
   }
   assert(cl_arg==cl_argc);
   cl_args[cl_arg] = NULL;
   status = buildandrun((int) cl_argc, cl_args, NULL, NULL, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
      exit(status);
   }
   for (size_t arg=0; arg<cl_argc; arg++) {
       free(cl_args[arg]);
   }
   free(cl_args);

   cl_argc = 5+reqrtn+usethreads+(threadargno>0);;
   cl_args = (char **) malloc((cl_argc+1)*sizeof(char *));
   assert(cl_args);
   cl_arg = 0;
   cl_args[cl_arg++] = strdup(argv[0]);
   cl_args[cl_arg++] = strdup("-p");
   cl_args[cl_arg++] = strdup("input/CheckpointParameters2.params");
   cl_args[cl_arg++] = strdup("-c");
   cl_args[cl_arg++] = strdup("checkpoints1/Checkpoint12");
   if (reqrtn) {
      cl_args[cl_arg++] = strdup("--require-return");
   }
   if (usethreads) {
      cl_args[cl_arg++] = strdup("-t");
      if (threadargno>0) {
         cl_args[cl_arg++] = strdup(argv[threadargno]);
      }
   }
   assert(cl_arg==cl_argc);
   cl_args[cl_arg++] = NULL;
   status = buildandrun(cl_argc, cl_args, NULL, &customexit, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
   }

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif

   for (size_t arg=0; arg<cl_argc; arg++) {
       free(cl_args[arg]);
   }
   free(cl_args);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   PVParams * params = hc->parameters();
   char * preLayerName = NULL;
   char * postLayerName = NULL;
   if( !strcmp(keyword, "CPTestInputLayer") ) {
      addedGroup = (void *) new CPTestInputLayer(name, hc);
   }
   if( !strcmp(keyword, "VaryingHyPerConn") ) {
      addedGroup = (void * ) new VaryingHyPerConn(name, hc);
   }
   free(preLayerName); preLayerName = NULL;
   free(postLayerName); postLayerName = NULL;
   return addedGroup;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   int rank = hc->icCommunicator()->commRank();
   int rootproc = 0;
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * cpdir1 = "checkpoints1";
      const char * cpdir2 = hc->parameters()->stringValue("column", "checkpointWriteDir");
      if(cpdir1 == NULL || cpdir2 == NULL) {
         fprintf(stderr, "%s: unable to allocate memory for names of checkpoint directories", argv[0]);
         exit(EXIT_FAILURE);
      }
      char * shellcommand;
      char c;
      const char * fmtstr = "diff -r -q -x timers.txt %s/Checkpoint%d %s/Checkpoint%d";
      int len = snprintf(&c, 1, fmtstr, cpdir1, index, cpdir2, index);
      shellcommand = (char *) malloc(len+1);
      if( shellcommand == NULL) {
         fprintf(stderr, "%s: unable to allocate memory for shell diff command.\n", argv[0]);
         status = PV_FAILURE;
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
         status = PV_FAILURE;
      }
      free(shellcommand); shellcommand = NULL;
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
#endif
   return status;
}
