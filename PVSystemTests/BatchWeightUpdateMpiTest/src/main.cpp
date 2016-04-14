/*
 * main .cpp file for CheckpointSystemTest
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int rank = 0;
   int size = 0;
   PV_Init initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //if(size != 5){
   //   std::cout << "BatchWeightUpdateMpiTest must be ran with 16 mpi processes\n";
   //   exit(-1);
   //}

   char const * paramFile1 = "input/timeBatch.params";
   char const * paramFile2 = "input/dimBatch.params";
   int status = PV_SUCCESS;
   PV_Arguments * arguments = initObj.getArguments();
   if (arguments->getParamsFile()!=NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getCheckpointReadDir()!=NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the checkpoint directory argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getRestartFlag()) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the restart flag.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getNumRows()!=0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the rows argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getNumColumns()!=0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the columns argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getBatchWidth()!=0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the batchwidth argument.\n", arguments->getProgramName());
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

   arguments->setParamsFile(paramFile1);
   arguments->setNumRows(1);
   arguments->setNumColumns(2);
   arguments->setBatchWidth(1);

   status = buildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile1, status);
      exit(status);
   }

   arguments->setParamsFile(paramFile2);
   arguments->setBatchWidth(5);

   status = buildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile2, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int compareFiles(const char* file1, const char* file2){
   if(file1 == NULL || file2 == NULL) {
      fprintf(stderr, "Unable to allocate memory for names of checkpoint directories");
      exit(EXIT_FAILURE);
   }

   FILE * fp1 = fopen(file1, "r");
   if(!fp1){
      std::cout << "Unable to open file " << file1 << ": " << strerror(errno) << ". Retrying.\n";
      fp1 = fopen(file1, "r");
      if (!fp1) {
         sleep(1U);
         std::cout << "Still unable to open file " << file1 << ": " << strerror(errno) << ". Test failed.\n";
      }
   }
   FILE * fp2 = fopen(file2, "r");
   if(!fp2){
      std::cout << "Unable to open file " << file2 << ": " << strerror(errno) << ". Retrying.\n";
      if (!fp2) {
         sleep(1U);
         std::cout << "Still unable to open file " << file2 << ": " << strerror(errno) << ". Test failed.\n";
      }
   }
   if (!fp1 || !fp2) {
      exit(EXIT_FAILURE);
   }
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
         std::cout << "Value returned from fread of file \"" << file1 << "\" is " << check1 << " as opposed to 1\n";
         exit(-1);
      }
      if(check2 != 1){
         std::cout << "Value returned from fread of file \"" << file2 << "\" is " << check2 << " as opposed to 1\n";
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
      const char * filebatch0 = "outputDim/batchsweep_00/Last/plasticConn_W.pvp";
      const char * filebatch1 = "outputDim/batchsweep_01/Last/plasticConn_W.pvp";
      const char * filebatch2 = "outputDim/batchsweep_02/Last/plasticConn_W.pvp";
      const char * filebatch3 = "outputDim/batchsweep_03/Last/plasticConn_W.pvp";
      const char * filebatch4 = "outputDim/batchsweep_04/Last/plasticConn_W.pvp";
      status = compareFiles(filetime, filebatch0);
      status = compareFiles(filetime, filebatch1);
      status = compareFiles(filetime, filebatch2);
      status = compareFiles(filetime, filebatch3);
      status = compareFiles(filetime, filebatch4);
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
   return status;
}
