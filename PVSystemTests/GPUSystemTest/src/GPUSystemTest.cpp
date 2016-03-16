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
#include <io/ParamGroupHandler.hpp>
// CustomGroupHandler is for adding objects not supported by CoreParamGroupHandler().
class CustomGroupHandler: public ParamGroupHandler {
public:
   CustomGroupHandler() {}

   virtual ~CustomGroupHandler() {}

   virtual ParamGroupType getGroupType(char const * keyword) {
      ParamGroupType result = UnrecognizedGroupType;
      //
      // This routine should compare keyword to the list of keywords handled by CustomGroupHandler and return one of
      // LayerGroupType, ConnectionGroupType, ProbeGroupType, ColProbeGroupType, WeightInitializerGroupType, or
      // WeightNormalizerGroupType
      // according to the keyword, or UnrecognizedGroupType if this ParamGroupHandler object does not know the keyword.
      //
      if (keyword==NULL) {
         return result;
      }
      else if (!strcmp(keyword, "GPUSystemTestProbe")) {
         result = ProbeGroupType;
      }
      else if (!strcmp(keyword, "identicalBatchProbe")) {
         result = ProbeGroupType;
      }
      else {
         result = UnrecognizedGroupType;
      }
      return result;
   }
   // A CustomGroupHandler group should override createLayer, createConnection, etc., as appropriate, if there are custom
   // objects
   // corresponding to that group type.
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) {
      BaseProbe * addedProbe = NULL;
      bool errorFound = false;
      if (keyword==NULL) {
         return addedProbe;
      }
      else if (!strcmp(keyword, "GPUSystemTestProbe")) {
         addedProbe = new GPUSystemTestProbe(name, hc);
         if (addedProbe==NULL) { errorFound = true; }
      }
      else if (!strcmp(keyword, "identicalBatchProbe")) {
         addedProbe = new identicalBatchProbe(name, hc);
         if (addedProbe==NULL) { errorFound = true; }
      }
      else {
         fprintf(stderr, "CustomGroupProbe error creating %s \"%s\": %s is not a recognized keyword.\n", keyword, name,
keyword);
         addedProbe = NULL;
      }

      if (errorFound) {
         assert(addedProbe==NULL);
         fprintf(stderr, "CustomGroupProbe error: unable to create %s \"%s\"\n", keyword, name);
      }
      return addedProbe;
   }
}; /* class CustomGroupHandler */
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
   return EXIT_FAILURE;
#endif

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
#else
   status = buildandrun(argc, argv, NULL, 0/*numGroupHandlers*/);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
