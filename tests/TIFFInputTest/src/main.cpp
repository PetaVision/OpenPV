/*
 * main.cpp
 *
 */
#include <columns/PV_Init.hpp>

#include <cMakeHeader.h> // Loads the value of PV_USE_TIFF

#ifdef PV_USE_TIFF

#include <cstdlib>
#include <tiffio.h>
#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <utils/PVLog.hpp>

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = buildandrun(&pv_initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else // PV_USE_TIFF

#include <cstdlib>
#include <columns/PV_Init.hpp>
#include <utils/PVLog.hpp>
int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   ErrorLog().printf(
         "%s requires the PV_USE_TIFF option to be on.\n",
         pv_initObj.returnProgramName());
   return EXIT_FAILURE;
}

#endif // PV_USE_TIFF
