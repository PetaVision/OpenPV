/*
 * pv.cpp
 *
 */

#include <iostream>
#include <sstream>
#include <mpi.h>
#include "../PetaVision/src/columns/buildandrun.hpp"

#undef MAIN_USES_CUSTOMGROUPS

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   const char * paramfile1 = "input/RandStateSystemTest1.params";
   const char * paramfile2 = "input/RandStateSystemTest2.params";

#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI

   int cl_argc = 3;
   char * cl_argv[3];
   cl_argv[0] = argv[0];
   cl_argv[1] = strdup("-p");
   cl_argv[2] = strdup(paramfile1);

   int status1 = buildandrun(cl_argc, cl_argv, NULL, NULL, NULL);
   if (status1 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s with return code %d.\n", cl_argv[0], cl_argv[2], status1);
      return EXIT_FAILURE;
   }

   free(cl_argv[2]);
   cl_argv[2] = strdup(paramfile2);
   int status2 = buildandrun(cl_argc, cl_argv, NULL, &customexit, NULL);
   if (status2 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s.\n", cl_argv[0], cl_argv[2]);
   }
   int status = status1==PV_SUCCESS && status2==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   free(cl_argv[2]);
   free(cl_argv[1]);

#ifdef PV_USE_MPI
   MPI_Finalize();
   if (status == EXIT_SUCCESS) {
      printf("Test complete.  %s passed on process rank %d.\n", cl_argv[0], rank);
   }
   else {
      fprintf(stderr, "Test complete.  %s FAILED on process rank %d.\n", cl_argv[0], rank);
   }
#else
   if (status == EXIT_SUCCESS) {
      printf("Test complete.  %s passed.\n", cl_argv[0]);
   }
   else {
      fprintf(stderr, "Test complete.  %s FAILED.\n", cl_argv[0]);
   }
#endif // PV_USE_MPI


   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   if (hc->columnId()==0) {
      std::string cmd("diff -r -q checkpoints*/Checkpoint");
      std::stringstream stepnumber;
      stepnumber << hc->numberOfTimeSteps();
      cmd += stepnumber.str();
      const char * cmdstr = cmd.c_str();
      status = system(cmdstr);
      if (status != 0) {
         fprintf(stderr, "%s failed: system command \"%s\" returned %d\n", argv[0], cmdstr, status);
      }
   }
   return status;
}
