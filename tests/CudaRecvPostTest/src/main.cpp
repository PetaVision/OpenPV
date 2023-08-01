/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <cstdlib>

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv);
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
      return EXIT_SUCCESS;
   }
   else {
      // Whatever caused the test to fail should have generated a log message.
      return EXIT_FAILURE;
   }
}
