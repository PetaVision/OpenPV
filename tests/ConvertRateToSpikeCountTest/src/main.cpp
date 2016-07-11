/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv);
   if (status == PV_SUCCESS) {
      pvInfo().printf("%s succeeded.\n", argv[0]);
   }
   else {
      pvError().printf("%s failed with return code %d.\n", argv[0], status);
   }
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
