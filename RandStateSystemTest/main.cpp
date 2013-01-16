/*
 * pv.cpp
 *
 */


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

   int status1 = buildandrun(cl_argc, cl_argv, NULL, &customexit, NULL);
   if (status1 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s with return code %d.\n", cl_argv[0], cl_argv[2], status1);
      return EXIT_FAILURE;
   }

   free(cl_argv[2]);
   cl_argv[2] = strdup(paramfile2);
   int status2 = buildandrun(cl_argc, cl_argv, NULL, &customexit, NULL);
   if (status2 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s with return code %d.\n", cl_argv[0], cl_argv[2], status2);
      return EXIT_FAILURE;
   }
   free(cl_argv[2]);
   free(cl_argv[1]);

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif // PV_USE_MPI

   int status = system("diff -r -q checkpoints*/Checkpoint10");

   return status ? EXIT_FAILURE : EXIT_SUCCESS;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   return PV_SUCCESS;
}
