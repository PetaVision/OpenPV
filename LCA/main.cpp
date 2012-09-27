/*
 * pv.cpp
 *
 */


#include <src/columns/buildandrun.hpp>
#include "LCALayer.hpp"

void * customgroup(const char * name, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "LCALayer")) {
      addedGroup = (void *) new LCALayer(name, hc, MAX_CHANNELS);
   }
   checknewobject(addedGroup, keyword, name, hc);
   return addedGroup;
}
