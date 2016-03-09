/*
 * main .cpp file for CheckpointSystemTest
 *
 */


#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include "CPTestInputLayer.hpp"
#include "VaryingHyPerConn.hpp"

class CustomGroupHandler : public ParamGroupHandler {
public:
   CustomGroupHandler() {}
   virtual ~CustomGroupHandler() {}
   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      if (keyword==NULL) { result = UnrecognizedGroupType; }
      else if (!strcmp(keyword, "CPTestInputLayer")) { result = LayerGroupType; }
      else if (!strcmp(keyword, "VaryingHyPerConn")) { result = ConnectionGroupType; }
      else { result = UnrecognizedGroupType; }
      return result;
   }
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc) {
      HyPerLayer * addedLayer = NULL;
      bool matched = false;
      if (keyword==NULL) { addedLayer = NULL; }
      else if (!strcmp(keyword, "CPTestInputLayer")) {
         matched = true;
         addedLayer = new CPTestInputLayer(name, hc);
      }
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
      else if (!strcmp(keyword, "VaryingHyPerConn")) {
         matched = true;
         addedConn = new VaryingHyPerConn(name, hc, weightInitializer, weightNormalizer);
      }
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
   PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char const * paramFile1 = "input/CheckpointParameters1.params";
   char const * paramFile2 = "input/CheckpointParameters2.params";
   int status = PV_SUCCESS;
   PV_Arguments * arguments = initObj->getArguments();
   if (arguments->getParamsFile()!=NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", arguments->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (arguments->getCheckpointReadDir()!=NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (arguments->getRestartFlag()) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the restart flag.\n", argv[0]);
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

   arguments->setParamsFile(paramFile1);
   arguments->setBatchWidth(2);

   status = rebuildandrun(initObj, NULL, NULL, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile1, status);
      exit(status);
   }

   arguments->setParamsFile(paramFile2);
   arguments->setCheckpointReadDir("checkpoints1/batchsweep_00/Checkpoint12:checkpoints1/batchsweep_01/Checkpoint12");

   status = rebuildandrun(initObj, NULL, &customexit, &customGroupHandler, 1);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile2, status);
   }

   delete customGroupHandler;

   delete initObj;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int diffDirs(const char* cpdir1, const char* cpdir2, int index){
   int status = PV_SUCCESS;
   if(cpdir1 == NULL || cpdir2 == NULL) {
      fprintf(stderr, "unable to allocate memory for names of checkpoint directories");
      exit(EXIT_FAILURE);
   }
   const int max_buf_len = 1024;
   char shellcommand[max_buf_len];
   const char * fmtstr = "diff -r -q -x timers.txt -x pv.params -x pv.params.lua %s/Checkpoint%d %s/Checkpoint%d";
   snprintf(shellcommand, max_buf_len, fmtstr, cpdir1, index, cpdir2, index);
   status = system(shellcommand);
   if( status != 0 ) {
      // Allow for possibility that the file system hasn't finished sync'ing yet
      sleep(1);
      status = system(shellcommand);
      if (status != 0) {
         fprintf(stderr, "system(\"%s\") returned %d\n", shellcommand, status);
      }
      status = PV_FAILURE;
   }
   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int rank = hc->icCommunicator()->commRank();
   int rootproc = 0;
   int status = PV_SUCCESS;
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * cpdir1 = "checkpoints1/batchsweep_00";
      const char * cpdir2 = "checkpoints2/batchsweep_00";
      status = diffDirs(cpdir1, cpdir2, index);
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
   assert(status == PV_SUCCESS);
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * cpdir1 = "checkpoints1/batchsweep_01";
      const char * cpdir2 = "checkpoints2/batchsweep_01";
      status = diffDirs(cpdir1, cpdir2, index);
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
   assert(status == PV_SUCCESS);
   return status;
}
