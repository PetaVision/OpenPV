/*
 * GPUSystemTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "GPUSystemTestProbe.hpp"
#include "identicalBatchProbe.hpp"

#define MAIN_USES_CUSTOMGROUPS

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   MPI_Init(&argc, &argv);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank==0) {
      printf("%s was compiled without GPUs.  Exiting\n", argv[0]);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return EXIT_SUCCESS;
#endif

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (strcmp(keyword, "GPUSystemTestProbe") == 0){
      addedGroup = new GPUSystemTestProbe(name, hc);
   }
   else if (strcmp(keyword, "identicalBatchProbe") == 0){
      addedGroup = new identicalBatchProbe(name, hc);
   }
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
