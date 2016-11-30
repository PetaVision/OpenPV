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

namespace PV {

CommandLineArguments::CommandLineArguments(int argc, char *argv[], bool allowUnrecognizedArguments) {
   initialize_base();
   initialize(argc, argv, allowUnrecognizedArguments);
}

int CommandLineArguments::initialize_base() {
   return PV_SUCCESS;
}

int CommandLineArguments::initialize(int argc, char *argv[], bool allowUnrecognizedArguments) {
   resetState(argc, argv, allowUnrecognizedArguments);
   return PV_SUCCESS;
}

void CommandLineArguments::resetState(int argc, char *argv[], bool allowUnrecognizedArguments) {
   bool paramUsage[argc];
   bool requireReturn = false;
   char *outputPath = nullptr;
   char *paramsFile = nullptr;
   char *logFile = nullptr;
   char *gpuDevices = nullptr;
   unsigned int randomSeed = 0U;
   char *workingDir = nullptr;
   int restart = 0;
   char *checkpointReadDir = nullptr;
   bool useDefaultNumThreads = false;
   int numThreads = 0;
   int numRows = 0;
   int numColumns = 0;
   int batchWidth = 0;
   int dryRun = 0;
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
   std::string outputPathString{outputPath ? outputPath : ""};
   std::string paramsFileString{paramsFile ? paramsFile : ""};
   std::string logFileString{logFile ? logFile : ""};
   std::string gpuDevicesString{gpuDevices ? gpuDevices : ""};
   std::string workingDirString{workingDir ? workingDir : ""};
   std::string checkpointReadDirString{checkpointReadDir ? checkpointReadDir : ""};
   std::string configString = ConfigParser::createString(
      requireReturn,
      outputPathString,
      paramsFileString,
      logFileString,
      gpuDevicesString,
      randomSeed,
      workingDirString,
      restart,
      checkpointReadDirString,
      useDefaultNumThreads,
      numThreads,
      numRows,
      numColumns,
      batchWidth,
      dryRun);
   std::istringstream configStream{configString};
   Arguments::resetState(configStream, allowUnrecognizedArguments);
}

} /* namespace PV */
