/*
 * main .cpp file for CheckpointSystemTest
 *
 */


#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>

class CustomGroupHandler : public ParamGroupHandler {
public:
   CustomGroupHandler() {}
   virtual ~CustomGroupHandler() {}
   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      if (keyword==NULL) { result = UnrecognizedGroupType; }
      //else if (!strcmp(keyword, "CPTestInputLayer")) { result = LayerGroupType; }
      //else if (!strcmp(keyword, "VaryingHyPerConn")) { result = ConnectionGroupType; }
      //else { result = UnrecognizedGroupType; }
      return result;
   }
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc) {
      HyPerLayer * addedLayer = NULL;
      bool matched = false;
      if (keyword==NULL) { addedLayer = NULL; }
      //else if (!strcmp(keyword, "CPTestInputLayer")) {
      //   matched = true;
      //   addedLayer = new CPTestInputLayer(name, hc);
      //}
      else { addedLayer = NULL; }
      if (matched && !addedLayer) {
         fprintf(stderr, "Rank %d process unable to create %s \"%s\".\n", hc->columnId(), keyword, name);
         exit(EXIT_FAILURE);
      }
      return addedLayer;
   }
   virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
      BaseConnection * addedConn = NULL;
      bool matched = false;
      if (keyword==NULL) { addedConn = NULL; }
      //else if (!strcmp(keyword, "VaryingHyPerConn")) {
      //   matched = true;
      //   addedConn = new VaryingHyPerConn(name, hc, weightInitializer, weightNormalizer);
      //}
      else { addedConn = NULL; }
      if (matched && !addedConn) {
         fprintf(stderr, "Rank %d process unable to create %s \"%s\".\n", hc->columnId(), keyword, name);
         exit(EXIT_FAILURE);
      }
      return addedConn;
   }
}; // class CustomGroupHandler

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int rank = 0;
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI
   char const * paramFile1 = "input/timeBatch.params";
   char const * paramFile2 = "input/dimBatch.params";
   int status = PV_SUCCESS;
   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt_str(argc, argv, "-c", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-r", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         fprintf(stderr, "This test uses two hard-coded params files, %s and %s. The second run is started from a checkpoint from the first run, and the results of the two runs are compared.\n",
               paramFile1, paramFile2);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   if (rank==0) {
      char const * rmcommand = "rm -rf checkpoints1 checkpoints2 output";
      status = system(rmcommand);
      if (status != 0) {
         fprintf(stderr, "deleting old checkpoints and output directories failed: \"%s\" returned %d\n", rmcommand, status);
         exit(EXIT_FAILURE);
      }
   }

   ParamGroupHandler * customGroupHandler = new CustomGroupHandler;

   int pv_argc = 2 + argc; // command line arguments, plus "-p" plus paramFile1

   char ** pv_argv = (char **) calloc((pv_argc+1), sizeof(char *));
   assert(pv_argv!=NULL);
   int pv_arg=0;
   for (pv_arg = 0; pv_arg < argc; pv_arg++) {
      pv_argv[pv_arg] = strdup(argv[pv_arg]);
      assert(pv_argv[pv_arg]);
   }
   assert(pv_arg==argc);
   pv_argv[pv_arg++] = strdup("-p");
   pv_argv[pv_arg++] = strdup(paramFile1);

   status = buildandrun((int) pv_argc, pv_argv, NULL, NULL, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", pv_argv[0], rank, paramFile1, status);
      exit(status);
   }

   free(pv_argv[argc+1]);
   pv_argv[argc+1] = strdup(paramFile2);
   assert(pv_argv[argc+1]);
   assert(pv_arg==argc+2);

   status = buildandrun(pv_argc, pv_argv, NULL, &customexit, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", pv_argv[0], rank, paramFile2, status);
   }

   delete customGroupHandler;

   for (size_t arg=0; arg<pv_argc; arg++) {
       free(pv_argv[arg]);
   }
   free(pv_argv);

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   int rank = hc->icCommunicator()->commRank();
   int rootproc = 0;
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * file1 = "outputTime/Last/plasticConn_W.pvp";
      const char * file2 = "outputDim/Last/plasticConn_W.pvp";
      if(file1 == NULL || file2 == NULL) {
         fprintf(stderr, "%s: unable to allocate memory for names of checkpoint directories", argv[0]);
         exit(EXIT_FAILURE);
      }

      FILE * fp1 = fopen(file1, "r");
      FILE * fp2 = fopen(file2, "r");
#define NUM_WGT_PARAMS (NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS)
      //Seek past the header
      fseek(fp1, NUM_WGT_PARAMS * sizeof(int), SEEK_SET);
      fseek(fp2, NUM_WGT_PARAMS * sizeof(int), SEEK_SET);
      char ch1, ch2;
      int flag = 0;
      while(((ch1 = fgetc(fp1)) != EOF) && ((ch2 = fgetc(fp2)) != EOF)){
         //Character comparison
         if(ch1 == ch2){
            flag = 1;
            continue;
         }
         //If characters do not match up
         else{
            std::cout << "File " << file1 << " and " << file2 << " are different\n";
            exit(-1);
         }
      }
      
      //char * shellcommand;
      //char c;
      //const char * fmtstr = "diff -q %s/plasticConn_W.pvp %s/plasticConn_W.pvp";
      //int len = snprintf(&c, 1, fmtstr, cpdir1, cpdir2);
      //shellcommand = (char *) malloc(len+1);
      //if( shellcommand == NULL) {
      //   fprintf(stderr, "%s: unable to allocate memory for shell diff command.\n", argv[0]);
      //   status = PV_FAILURE;
      //}
      //assert( snprintf(shellcommand, len+1, fmtstr, cpdir1, cpdir2) == len );
      //status = system(shellcommand);
      //if( status != 0 ) {
      //   fprintf(stderr, "system(\"%s\") returned %d\n", shellcommand, status);
      //   // Because system() seems to return the result of the shell command multiplied by 256,
      //   // and Unix only sees the 8 least-significant bits of the value returned by a C/C++ program,
      //   // simply returning the result of the system call doesn't work.
      //   // I haven't found the mult-by-256 behavior in the documentation, so I'm not sure what's
      //   // going on.
      //   status = PV_FAILURE;
      //}
      //free(shellcommand); shellcommand = NULL;
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
#endif
   return status;
}
