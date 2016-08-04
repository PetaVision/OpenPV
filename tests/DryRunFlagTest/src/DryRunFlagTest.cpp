/*
 * DryRunFlagTest.cpp
 *
 */

#include <sys/types.h>
#include <unistd.h>
#include <columns/buildandrun.hpp>

#define PROCESSED_PARAMS "processed.params"

int deleteGeneratedFiles(PV::PV_Init * pv_obj);
int compareOutputs();
int checkDryRunSet(HyPerCol * hc, int argc, char * argv[]);
int checkDryRunCleared(HyPerCol * hc, int argc, char * argv[]);
int checkNumTimesteps(HyPerCol * hc, char const * programName);

int deleteFile(char const * path, PV::PV_Init * pv_obj);

int main(int argc, char * argv[]) {

   int status = PV_SUCCESS;

   PV::PV_Init pv_obj(&argc, &argv, false/*allowUnrecognizedArguments*/);

   if (status != PV_SUCCESS) {
      pvError().printf("%s: PV_Init::initialize() failed on process with PID=%d\n", argv[0], getpid()); 
   }

   pv_obj.setDryRunFlag(true);

   if (pv_obj.getParamsFile()==NULL) {
      pv_obj.setParams("input/DryRunFlagTest.params");
   }

   if (pv_obj.isExtraProc()) { return EXIT_SUCCESS; }

   int rank = pv_obj.getCommunicator()->globalCommRank();

   status = deleteGeneratedFiles(&pv_obj);
   if (status!=PV_SUCCESS) {
      pvError().printf("%s: error cleaning generated files from any previous run.\n", argv[0]);
   }

   status = rebuildandrun(&pv_obj, NULL, checkDryRunSet);

   if (status!=PV_SUCCESS) {
      pvError().printf("%s: running with dry-run flag set failed on process %d.\n", argv[0], rank);
   }

   // Re-run, without the dry-run flag.
   pv_obj.setDryRunFlag(false);
   pv_obj.setOutputPath("output-generate");
   status = rebuildandrun(&pv_obj, NULL, checkDryRunCleared);
   if (status != PV_SUCCESS) {
      pvError().printf("%s: running with dry-run flag cleared failed on process %d\n", argv[0], rank);
   }

   // Run the column with the cleaned-up params file, sending output to directory "output-verify/"
   pv_obj.setOutputPath("output-verify");
   pv_obj.setParams("output/pv.params");
   status = rebuildandrun(&pv_obj, NULL, checkDryRunCleared);
   if (status != PV_SUCCESS) {
      pvError().printf("%s: running with processed params file failed on process %d\n", argv[0], rank);
   }

   if (rank==0) {
      status = compareOutputs();
      if (status != PV_SUCCESS) {
         pvError().printf("%s: compareOutputs() failed with return code %d.\n", argv[0], status);
      }
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int deleteGeneratedFiles(PV::PV_Init * pv_obj) {

   int status = PV_SUCCESS;
   if (pv_obj->getCommunicator()->globalCommRank()==0) {
      char const * filename = NULL;

      if (deleteFile(PROCESSED_PARAMS, pv_obj) != PV_SUCCESS) { status = PV_FAILURE; }
      if (deleteFile(PROCESSED_PARAMS ".lua", pv_obj) != PV_SUCCESS) { status = PV_FAILURE; }
      if (system("rm -rf output") != PV_SUCCESS) { status = PV_FAILURE; }
      if (system("rm -rf output-generate") != PV_SUCCESS) { status = PV_FAILURE; }
      if (system("rm -rf output-verify") != PV_SUCCESS) { status = PV_FAILURE; }
   }
   MPI_Bcast(&status, 1, MPI_INT, 0, pv_obj->getCommunicator()->communicator());
   return status;
}

int compareOutputs() {
   // Only root process calls this function, since it does I/O
#ifndef NDEBUG
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   pvErrorIf(!(rank==0), "Test failed.\n");
#endif // NDEBUG

   int status = PV_SUCCESS;
   char const * diffcmd = "diff -r -q -x pv.params -x pv.params.lua -x timers.txt output-generate output-verify";
   status = system(diffcmd);
   if (status != 0) {
      // If the diff command fails, it may be only that the file system hasn't caught up yet.  
      pvWarn().printf("diff command returned %d: waiting 1 second and trying again...\n", WEXITSTATUS(status));
      pvWarn().flush();
      sleep(1);
      status = system(diffcmd);
      if (status!=0) { status = WEXITSTATUS(status); }
   }
   return status;
}

int deleteFile(char const * path, PV::PV_Init * pv_obj) {
   // Only root process calls this function, since it does I/O
#ifndef NDEBUG
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   pvErrorIf(!(rank==0), "Test failed.\n");
#endif // NDEBUG

   int status = unlink(path);
   if (status != 0 && errno != ENOENT) {
      pvErrorNoExit().printf("%s: error deleting %s: %s\n", pv_obj->getProgramName(), path, strerror(errno));
      status = PV_FAILURE;
   }
   else {
      status = PV_SUCCESS;
   }
   return PV_SUCCESS;
}

int checkDryRunSet(HyPerCol * hc, int argc, char * argv[]) {
   pvErrorIf(!(argc>=1), "Test failed.\n");
   int status = checkNumTimesteps(hc, argv[0]);
   if (hc->getCurrentStep() != hc->getInitialStep()) {
      if (hc->columnId()==0) {
         pvErrorNoExit().printf("%s failed: with dry-run flag set, initialStep was %ld but currentStep was %ld\n",
                 argv[0], hc->getInitialStep(), hc->getCurrentStep());
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
      status = PV_FAILURE;
   }
   return status;
}

int checkDryRunCleared(HyPerCol * hc, int argc, char * argv[]) {
   pvErrorIf(!(argc>=1), "Test failed.\n");
   int status = checkNumTimesteps(hc, argv[0]);
   if (hc->getCurrentStep() != hc->getFinalStep()) {
      if (hc->columnId()==0) {
         pvErrorNoExit().printf("%s failed: with dry-run flag cleared, finalStep was %ld but currentStep was %ld\n",
                 argv[0], hc->getFinalStep(), hc->getCurrentStep());
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
      status = PV_FAILURE;
   }
   return status;
}

int checkNumTimesteps(HyPerCol * hc, char const * programName) {
   int status;
   if (hc->getInitialStep() == hc->getFinalStep()) {
      if (hc->columnId()==0) {
         pvError().printf("HyPerCol has same initial step and final step (%ld): unable to test dry-run flag.\n", hc->getInitialStep());
      }
      status = PV_FAILURE;
   }
   else {
      status = PV_SUCCESS;
   }
   return status;
}
