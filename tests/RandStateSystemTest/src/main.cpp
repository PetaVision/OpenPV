/*
 * pv.cpp
 *
 */

#include <iostream>
#include <sstream>
#include <arch/mpi/mpi.h>
#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   const char * paramfile1 = "input/RandStateSystemTest1.params";
   const char * paramfile2 = "input/RandStateSystemTest2.params";

   int rank=0;
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Can't use `initObj.getComm()->communicator()` because initObj.initialize hasn't been called. */

   if (initObj.getParamsFile() != NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("%s does not take -p as an option.  Instead the necessary params files are hard-coded.\n", initObj.getProgramName());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   initObj.setParams(paramfile1);
   int status1 = rebuildandrun(&initObj, NULL, NULL);
   if (status1 != PV_SUCCESS) {
      pvError().printf("%s failed on param file %s with return code %d.\n", initObj.getProgramName(), paramfile1, status1);
   }

   initObj.setParams(paramfile2);
   int status2 = rebuildandrun(&initObj, NULL, &customexit);
   if (status2 != PV_SUCCESS) {
      pvErrorNoExit().printf("%s failed on param file %s.\n", initObj.getProgramName(), paramfile2);
   }
   int status = status1==PV_SUCCESS && status2==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;

#ifdef PV_USE_MPI
   if (status == EXIT_SUCCESS) {
      pvInfo().printf("Test complete.  %s passed on process rank %d.\n", initObj.getProgramName(), rank);
   }
   else {
      pvErrorNoExit().printf("Test complete.  %s FAILED on process rank %d.\n", initObj.getProgramName(), rank);
   }
#else
   if (status == EXIT_SUCCESS) {
      pvInfo().printf("Test complete.  %s passed.\n", initObj.getProgramName());
   }
   else {
      pvErrorNoExit().printf("Test complete.  %s FAILED.\n", initObj.getProgramName());
   }
#endif // PV_USE_MPI

   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   if (hc->columnId()==0) {
      std::string cmd("diff -r -q -x timers.txt -x pv.params -x pv.params.lua checkpoints*/Checkpoint");
      std::stringstream stepnumber;
      stepnumber << hc->getFinalStep();
      cmd += stepnumber.str();
      const char * cmdstr = cmd.c_str();
      status = system(cmdstr);
      if (status != 0) {
         pvErrorNoExit().printf("%s failed: system command \"%s\" returned %d\n", argv[0], cmdstr, status);
      }
   }
   return status;
}
