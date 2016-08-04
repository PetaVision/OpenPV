/*
 * main .cpp file for CheckpointSystemTest
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "CPTestInputLayer.hpp"
#include "VaryingHyPerConn.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int rank = 0;
   PV_Init initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char const * paramFile1 = "input/CheckpointParameters1.params";
   char const * paramFile2 = "input/CheckpointParameters2.params";
   int status = PV_SUCCESS;
   if (initObj.getParamsFile()!=NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the params file argument.\n", initObj.getProgramName());
      }
      status = PV_FAILURE;
   }
   if (initObj.getCheckpointReadDir()!=NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (initObj.getRestartFlag()) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the restart flag.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         pvErrorNoExit().printf("This test uses two hard-coded params files, %s and %s. The second run is started from a checkpoint from the first run, and the results of the two runs are compared.\n",
               paramFile1, paramFile2);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   if (rank==0) {
      char const * rmcommand = "rm -rf checkpoints1 checkpoints2 output";
      status = system(rmcommand);
      if (status != 0) {
         pvError().printf("deleting old checkpoints and output directories failed: \"%s\" returned %d\n", rmcommand, status);
      }
   }

   initObj.registerKeyword("CPTestInputLayer", Factory::create<CPTestInputLayer>);
   initObj.registerKeyword("VaryingHyPerConn", Factory::create<VaryingHyPerConn>);
 
   initObj.setMPIConfiguration(0/*numRows unspecified*/, 0/*numColumns unspecified*/, 2/*batchWidth*/);
   initObj.setParams(paramFile1);

   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      pvError().printf("%s: rank %d running with params file %s returned error %d.\n", initObj.getProgramName(), rank, paramFile1, status);
   }

   initObj.setParams(paramFile2);
   initObj.setCheckpointReadDir("checkpoints1/batchsweep_00/Checkpoint12:checkpoints1/batchsweep_01/Checkpoint12");

   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      pvError().printf("%s: rank %d running with params file %s returned error %d.\n", initObj.getProgramName(), rank, paramFile2, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int diffDirs(const char* cpdir1, const char* cpdir2, int index){
   int status = PV_SUCCESS;
   if(cpdir1 == NULL || cpdir2 == NULL) {
      pvError().printf("unable to allocate memory for names of checkpoint directories");
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
         pvErrorNoExit().printf("system(\"%s\") returned %d\n", shellcommand, status);
      }
      status = PV_FAILURE;
   }
   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int rank = hc->getCommunicator()->commRank();
   int rootproc = 0;
   int status = PV_SUCCESS;
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * cpdir1 = "checkpoints1/batchsweep_00";
      const char * cpdir2 = "checkpoints2/batchsweep_00";
      status = diffDirs(cpdir1, cpdir2, index);
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->getCommunicator()->communicator());
   pvErrorIf(!(status == PV_SUCCESS), "Test failed.\n");
   if( rank == rootproc ) {
      int index = hc->getFinalStep()-hc->getInitialStep();
      const char * cpdir1 = "checkpoints1/batchsweep_01";
      const char * cpdir2 = "checkpoints2/batchsweep_01";
      status = diffDirs(cpdir1, cpdir2, index);
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->getCommunicator()->communicator());
   pvErrorIf(!(status == PV_SUCCESS), "Test failed.\n");
   return status;
}
