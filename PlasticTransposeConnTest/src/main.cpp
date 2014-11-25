/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TestAllZerosProbe.hpp"

void * customgroups(char const * keyword, char const * name, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL/*custominit*/, NULL/*customexit*/, &customgroups);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(char const * keyword, char const * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   bool errorFound = false;
   if ( !strcmp(keyword, "TestAllZerosProbe") ) {
      addedGroup = (void *) new TestAllZerosProbe(name, hc);
      if (!addedGroup) {
         fprintf(stderr, "Group \"%s\": Unable to create %s\n", name, keyword);
         errorFound = true;
      }
   }
   return addedGroup;
}
