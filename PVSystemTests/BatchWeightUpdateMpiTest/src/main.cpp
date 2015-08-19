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
   int size = 0;
   PV_Init* initObj = new PV_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //if(size != 5){
   //   std::cout << "BatchWeightUpdateMpiTest must be ran with 16 mpi processes\n";
   //   exit(-1);
   //}

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
   if (pv_getopt(argc, argv, "-rows", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the rows argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-columns", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the columns argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-batchwidth", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the batchwidth argument.\n", argv[0]);
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

   int pv_argc = 8 + argc; // command line arguments, plus "-p" plus paramFile1, -rows + num, -columns + num, -batchwidth + num

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
   pv_argv[pv_arg++] = strdup("-rows");
   pv_argv[pv_arg++] = strdup("1");
   pv_argv[pv_arg++] = strdup("-columns");
   pv_argv[pv_arg++] = strdup("1");
   pv_argv[pv_arg++] = strdup("-batchwidth");
   pv_argv[pv_arg++] = strdup("1");

   status = rebuildandrun((int) pv_argc, pv_argv, initObj, NULL, NULL, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", pv_argv[0], rank, paramFile1, status);
      exit(status);
   }

   free(pv_argv[argc+1]);
   pv_argv[argc+1] = strdup(paramFile2);
   free(pv_argv[argc+7]);
   pv_argv[argc+7] = strdup("5");

   assert(pv_argv[argc+1]);
   assert(pv_arg==argc+8);

   status = rebuildandrun(pv_argc, pv_argv, initObj, NULL, &customexit, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", pv_argv[0], rank, paramFile2, status);
   }

   delete customGroupHandler;

   for (size_t arg=0; arg<pv_argc; arg++) {
       free(pv_argv[arg]);
   }
   free(pv_argv);

//#ifdef PV_USE_MPI
//   MPI_Finalize();
//#endif
   delete initObj;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int compareFiles(const char* file1, const char* file2){
   if(file1 == NULL || file2 == NULL) {
      fprintf(stderr, "Unable to allocate memory for names of checkpoint directories");
      exit(EXIT_FAILURE);
   }

   FILE * fp1 = fopen(file1, "r");
   FILE * fp2 = fopen(file2, "r");
#define NUM_WGT_PARAMS (NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS)
   //Seek past the header
   fseek(fp1, NUM_WGT_PARAMS * sizeof(int), SEEK_SET);
   fseek(fp2, NUM_WGT_PARAMS * sizeof(int), SEEK_SET);
   
   float f1, f2;
   int flag = 0;
   while(!feof(fp1) && !feof(fp2)){
      //Read floating point numbers
      int check1 = fread(&f1, sizeof(float), 1, fp1);
      int check2 = fread(&f2, sizeof(float), 1, fp2);
      if(check1 == 0 && check2 == 0){
         //Both files end of file
         break;
      }
      if(check1 != 1){
         std::cout << "Value returned from fread is " << check1 << " as opposed to 1\n";
         exit(-1);
      }
      if(check2 != 1){
         std::cout << "Value returned from fread is " << check2 << " as opposed to 1\n";
         exit(-1);
      }
      //Floating piont comparison
      if(fabs(f1-f2) <= 1e-5){
         flag = 1;
         continue;
      }
      //If characters do not match up
      else{
         std::cout << "File " << file1 << " and " << file2 << " are different\n";
         exit(-1);
      }
   }
   return PV_SUCCESS;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   int rank = hc->icCommunicator()->globalCommRank();
   int rootproc = 0;
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * filetime = "outputTime/Last/plasticConn_W.pvp";
      const char * filebatch0 = "output_batchsweep_00/Last/plasticConn_W.pvp";
      const char * filebatch1 = "output_batchsweep_01/Last/plasticConn_W.pvp";
      const char * filebatch2 = "output_batchsweep_02/Last/plasticConn_W.pvp";
      const char * filebatch3 = "output_batchsweep_03/Last/plasticConn_W.pvp";
      const char * filebatch4 = "output_batchsweep_04/Last/plasticConn_W.pvp";
      status = compareFiles(filetime, filebatch0);
      status = compareFiles(filetime, filebatch1);
      status = compareFiles(filetime, filebatch2);
      status = compareFiles(filetime, filebatch3);
      status = compareFiles(filetime, filebatch4);
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
#endif
   return status;
}
