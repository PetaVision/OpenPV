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
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char const * paramFile1 = "input/CheckpointParameters1.params";
   char const * paramFile2 = "input/CheckpointParameters2.params";
   int status = PV_SUCCESS;
   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the params file argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt_str(argc, argv, "-c", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-r", NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the checkpoint directory argument.\n", argv[0]);
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

   initObj.registerKeyword("CPTestInputLayer", createCPTestInputLayer);
   initObj.registerKeyword("VaryingHyPerConn", createVaryingHyPerConn);
 
   PV_Arguments * arguments = initObj.getArguments();

   arguments->setParamsFile(paramFile1);
   initObj.initialize();
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      pvError().printf("%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile1, status);
   }

   initObj.getArguments()->setParamsFile(paramFile2);
   initObj.initialize();
   initObj.getArguments()->setCheckpointReadDir("checkpoints1/Checkpoint12");

   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      pvError().printf("%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile2, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
         pvError().printf("%s: unable to allocate memory for names of checkpoint directories", argv[0]);
      }
      const int max_buf_len = 1024;
      char shellcommand[max_buf_len];
      const char * fmtstr = "diff -r -q -x timers.txt -x pv.params -x pv.params.lua %s/Checkpoint%d %s/Checkpoint%d";
      snprintf(shellcommand, max_buf_len, fmtstr, cpdir1, index, cpdir2, index);
      status = system(shellcommand);
      if( status != 0 ) {
         pvErrorNoExit().printf("system(\"%s\") returned %d\n", shellcommand, WEXITSTATUS(status));
         status = PV_FAILURE;
      }
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->icCommunicator()->communicator());
   return status;
}
