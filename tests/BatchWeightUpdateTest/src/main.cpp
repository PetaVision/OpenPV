/*
 * main .cpp file for CheckpointSystemTest
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int rank = 0;
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   rank = initObj.getWorldRank();
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

   PV_Arguments * arguments = initObj.getArguments();
   arguments->setParamsFile(paramFile1);

   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile1, status);
      exit(status);
   }

   arguments->setParamsFile(paramFile2);

   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile2, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   int rank = hc->icCommunicator()->globalCommRank();
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
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->globalCommunicator());
   return status;
}
