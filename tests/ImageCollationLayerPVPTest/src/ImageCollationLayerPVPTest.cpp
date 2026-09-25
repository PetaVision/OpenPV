/*
 * ImageCollationLayerPVPTest.cpp
 *
 */

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string>

#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <utils/PVLog.hpp>

int main(int argc, char *argv[]) {
   PV_Init pvInitObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);

   std::string paramsFilesPath("input/TestParamsFiles.txt");
   std::ifstream paramsFilesStream(paramsFilesPath);
   FatalIf(
         !paramsFilesStream,
         "Unable to open the list of params files for this test (%s): error %d (%s)\n",
         paramsFilesPath.c_str(),
         errno,
         strerror(errno));
   std::string paramsFile;
   while (std::getline(paramsFilesStream, paramsFile, '\n')) {
      pvInitObj.setParams(paramsFile.c_str());
      int status = buildandrun(&pvInitObj);
      FatalIf(status != PV_SUCCESS, "Params file \"%s\" failed.\n", paramsFile.c_str());
   }

   paramsFile = "input/DensePVPInputPath.params";
   pvInitObj.setParams(paramsFile.c_str());

   int status = buildandrun(&pvInitObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
