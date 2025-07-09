/*
 * MaxPoolTest.cpp
 */

#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, NULL, NULL);
   if (status == PV_SUCCESS) {
      InfoLog().printf("Test succeeded.\n");
      return EXIT_SUCCESS;
   }
   else {
      ErrorLog().printf("Test failed.\n");
      return EXIT_FAILURE;
   }
}
