/*
 * GPUSystemTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "GPUSystemTestProbe.hpp"
#include "identicalBatchProbe.hpp"

#define MAIN_USES_CUSTOM_GROUPS

int main(int argc, char * argv[]) {

#if !defined(PV_USE_CUDA)
   MPI_Init(&argc, &argv);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank==0) {
      printf("%s was compiled without GPUs.  Exiting\n", argv[0]);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return EXIT_FAILURE;
#endif // !defined(PV_USE_CUDA)

   int status;
#ifdef MAIN_USES_CUSTOM_GROUPS
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("GPUSystemTestProbe", createGPUSystemTestProbe);
   pv_initObj.registerKeyword("identicalBatchProbe", create_identicalBatchProbe);
   status = buildandrun(&pv_initObj);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOM_GROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
