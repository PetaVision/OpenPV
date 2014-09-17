/*
 * Main file for FeedbackDelayTest
 * To run, use arguments -p input/FeedbackDelayTest.params
 *
 */


#include <columns/buildandrun.hpp>
#include "LayerPhaseTestProbe.hpp"

#define MAIN_USES_CUSTOMGROUP

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * name, const char * groupname, HyPerCol * hc);
// customgroups is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUP
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUP
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_CUSTOMGROUP
void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   LayerProbe * addedProbe;
   void * addedGroup = NULL;
   if( !strcmp(keyword, "LayerPhaseTestProbe") ) {
      addedProbe = (LayerProbe *) new LayerPhaseTestProbe(name, hc);
      if( !addedProbe ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         exit(EXIT_FAILURE);
      }
      addedGroup = (void *) addedProbe;
   }
   return addedGroup;
}
#endif // MAIN_USES_CUSTOMGROUP
