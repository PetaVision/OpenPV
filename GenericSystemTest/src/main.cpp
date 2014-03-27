/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>

#define MAIN_USES_CUSTOMGROUPS

int customexit(HyPerCol * hc, int argc, char * argv[]);

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   status = buildandrun(argc, argv, NULL, &customexit, &customgroup);
#else
   status = buildandrun(argc, argv, NULL, &customexit);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   const char * targetLayerName = "comparison";
   HyPerLayer * layer = hc->getLayerFromName(targetLayerName);
   LayerProbe * probe = NULL;
   int np = layer->getNumProbes();
   for (int p=0; p<np; p++) {
      if (!strcmp(layer->getProbe(p)->getProbeName(), "comparison_test")) {
         probe = layer->getProbe(p);
         break;
      }
   }
   RequireAllZeroActivityProbe * allzeroProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   assert(allzeroProbe);
   if (allzeroProbe->getNonzeroFound()) {
      if (hc->columnId()==0) {
         double t = allzeroProbe->getNonzeroTime();
         fprintf(stderr, "%s \"%s\" had at least one nonzero activity value, beginning at time %f\n",
               hc->parameters()->groupKeywordFromName(targetLayerName), targetLayerName, t);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

#ifdef MAIN_USES_CUSTOMGROUPS
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUPS
