/*
 * CommandLineArguments.cpp
 *
 *  Created on: Nov 28, 2016
 *      Author: pschultz
 */

#include "CommandLineArguments.hpp"
#include "include/pv_common.h"
#include "io/io.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <sstream>
#include <string>

namespace PV {

CommandLineArguments::CommandLineArguments(
      int argc,
      char const *const *argv,
      bool allowUnrecognizedArguments) {
   initialize_base();
   initialize(argc, argv, allowUnrecognizedArguments);
}

int CommandLineArguments::initialize_base() { return PV_SUCCESS; }

int CommandLineArguments::initialize(
      int argc,
      char const *const *argv,
      bool allowUnrecognizedArguments) {
   resetState(argc, argv, allowUnrecognizedArguments);
   return PV_SUCCESS;
}

void CommandLineArguments::resetState(
      int argc,
      char const *const *argv,
      bool allowUnrecognizedArguments) {
   bool paramUsage[argc];
   bool requireReturn        = false;
   char *outputPath          = nullptr;
   char *paramsFile          = nullptr;
   char *logFile             = nullptr;
   char *gpuDevices          = nullptr;
   unsigned int randomSeed   = 0U;
   char *workingDir          = nullptr;
   int restart               = 0;
   char *checkpointReadDir   = nullptr;
   bool useDefaultNumThreads = false;
   int numThreads            = 0;
   int numRows               = 0;
   int numColumns            = 0;
   int batchWidth            = 0;
   int dryRun                = 0;
   parse_options(
         argc,
         argv,
         paramUsage,
         &requireReturn,
         &outputPath,
         &paramsFile,
         &logFile,
         &gpuDevices,
         &randomSeed,
         &workingDir,
         &restart,
         &checkpointReadDir,
         &useDefaultNumThreads,
         &numThreads,
         &numRows,
         &numColumns,
         &batchWidth,
         &dryRun);
   if (!allowUnrecognizedArguments) {
      bool allrecognized = true;
      for (int argi = 0; argi < argc; argi++) {
         bool recognized = paramUsage[argi];
         if (!recognized) {
            allrecognized = false;
            ErrorLog().printf(
                  "Argument %d, \"%s\", is unrecognized and AllowUnrecognizedArguments is set to "
                  "false.\n",
                  argi,
                  argv[argi]);
         }
      }
      FatalIf(!allrecognized, "%s called with unrecognized arguments.\n", argv[0]);
   }
   std::string configString = ConfigParser::createString(
         requireReturn,
         std::string{outputPath ? outputPath : ""},
         std::string{paramsFile ? paramsFile : ""},
         std::string{logFile ? logFile : ""},
         std::string{gpuDevices ? gpuDevices : ""},
         randomSeed,
         std::string{workingDir ? workingDir : ""},
         (bool)restart,
         std::string{checkpointReadDir ? checkpointReadDir : ""},
         useDefaultNumThreads,
         numThreads,
         numRows,
         numColumns,
         batchWidth,
         (bool)dryRun);
   std::istringstream configStream{configString};
   Arguments::resetState(configStream, allowUnrecognizedArguments);
   free(outputPath);
   free(paramsFile);
   free(logFile);
   free(gpuDevices);
   free(workingDir);
   free(checkpointReadDir);
}

} /* namespace PV */
