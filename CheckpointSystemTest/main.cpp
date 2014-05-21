/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "CPTestInputLayer.hpp"
#include "VaryingKernelConn.hpp"
#include "VaryingHyPerConn.hpp"

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);
int customexit(HyPerCol * hc, int argc, char * argv[]);


int main(int argc, char * argv[]) {

   int rank;
   bool argerr = false;
   int reqrtn = 0;
   int threading = 0;

   if (argc > 1){
      for(int i = 1; i < argc; i++){
         std::cout << "i: " << i << "argc: " << argc << "\n";
         //Allowed arguments are -t and --require-return
         if(strcmp(argv[i], "-t") == 0){
            if(i+1 >= argc || argv[i+1][0] == '-'){
               //Do nothing
            }
            else{
               int numthreads = atoi(argv[i+1]);
               if(numthreads != 4){
                  std::cout << "Hardcoding to 4 threads\n";
               }
               i++;
            }
            threading = 2;
            argerr |= false;
         }
         else if(strcmp(argv[i], "--require-return") == 0){
            reqrtn = 1;
            argerr |= false;
         }
         else{
            //error
            argerr = true;
         }
      }
   }
   if (argerr) {
      fprintf(stderr, "%s: run without input arguments (except for --require-return or -t); the necessary arguments are hardcoded.\n", argv[0]);
      exit(EXIT_FAILURE);
   }
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
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
   MPI_Bcast(&charhit, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
#endif // REQUIRE_RETURN

   int status;
   assert(reqrtn==0 || reqrtn==1);
   assert(threading == 0 || threading == 2);
   int cl_argc = 3+reqrtn+threading;
   char * cl_args[cl_argc];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/CheckpointParameters1.params");
   if(threading && reqrtn){
      assert(cl_argc==6);
      cl_args[3] = strdup("-t");
      cl_args[4] = strdup("4");
      cl_args[5] = strdup("--require-return");
      std::cout << "Requiring Return and Threading!\n";
   }
   else if (threading) {
      assert(cl_argc==5);
      cl_args[3] = strdup("-t");
      cl_args[4] = strdup("4");
      std::cout << "Threading!\n";
   }
   else if (reqrtn){
      assert(cl_argc==4);
      cl_args[3] = strdup("--require-return");
      std::cout << "Requiring Return!\n";
   }
   status = buildandrun(cl_argc, cl_args, NULL, NULL, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
      exit(status);
   }

   free(cl_args[2]);
   cl_args[2] = strdup("input/CheckpointParameters2.params");
   status = buildandrun(cl_argc, cl_args, NULL, &customexit, &customgroup);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], cl_args[2], status);
   }

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif

   for (int i=0; i<cl_argc; i++) {
      free(cl_args[i]);
   }
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
   if( !strcmp(keyword, "VaryingKernelConn") ) {
      addedGroup = (void * ) new VaryingKernelConn(name, hc);
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
      const char * cpdir1 = hc->parameters()->stringValue("column", "checkpointReadDir");
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
