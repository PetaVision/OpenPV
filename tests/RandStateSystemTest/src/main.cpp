/*
 * pv.cpp
 *
 */

#include <iostream>
#include <sstream>
#include <arch/mpi/mpi.h>
#include <columns/buildandrun.hpp>
#include "ColumnArchive.hpp"

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

   float tolerance = 1.0e-7;

   initObj.setParams(paramfile1);
   HyPerCol * hc1 = build(&initObj);
   int status1 = hc1->run();
   if (status1 != PV_SUCCESS) {
      pvError().printf("%s failed to run on param file %s with return code %d.\n", initObj.getProgramName(), paramfile1, status1);
   }
   ColumnArchive columnArchive1(hc1, tolerance, tolerance);

   initObj.setParams(paramfile2);
   HyPerCol * hc2 = build(&initObj);
   int status2 = hc2->run();
   if (status2 != PV_SUCCESS) {
      pvErrorNoExit().printf("%s failed to run on param file %s.\n", initObj.getProgramName(), paramfile2);
   }
   ColumnArchive columnArchive2(hc1, tolerance, tolerance);

   int status3 = columnArchive1==columnArchive2 ? PV_SUCCESS : PV_FAILURE;
   if (status3 != PV_SUCCESS) {
      pvErrorNoExit().printf("%s failed comparing params files %s and %s.\n", initObj.getProgramName(), paramfile1, paramfile2);
   }
   int status = status1==PV_SUCCESS && status2==PV_SUCCESS && status3==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;

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
