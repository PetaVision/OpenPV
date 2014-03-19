/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "TestAllZerosProbe.hpp"

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL, NULL, &addcustomgroup);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * addcustomgroup(const char * keyword, const char * groupname, HyPerCol * hc) {
   void * addedProbe = NULL;

   if ( !strcmp(keyword, "TestAllZerosProbe") ) {
      addedProbe = new TestAllZerosProbe(groupname, hc);
   }
   checknewobject((void *) addedProbe, keyword, groupname, hc);
   return addedProbe;
}
