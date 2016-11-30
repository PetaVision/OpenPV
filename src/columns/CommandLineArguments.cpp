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
   char *paramFile = nullptr;
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
      &paramFile,
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
   std::string configString;
   if (requireReturn) {
      configString.append("RequireReturn:true\n");
   }
   if (outputPath) {
      configString.append("OutputPath:").append(outputPath).append("\n");
   }
   if (paramFile) {
      configString.append("ParamsFile:").append(paramFile).append("\n");
   }
   if (logFile) {
      configString.append("LogFile:").append(logFile).append("\n");
   }
   if (gpuDevices) {
      configString.append("GpuDevices:").append(gpuDevices).append("\n");
   }
   if (randomSeed) {
      configString.append("RandomSeed:").append(std::to_string(randomSeed)).append("\n");
   }
   if (workingDir) {
      configString.append("WorkingDirectory:").append(workingDir).append("\n");
   }
   if (restart) {
      configString.append("Restart:true\n");
   }
   if (checkpointReadDir) {
      configString.append("CheckpointReadDirectory:").append(checkpointReadDir).append("\n");
   }
   if (useDefaultNumThreads) {
      configString.append("NumThreads:-\n");
   }
   else {
      configString.append("NumThreads:").append(std::to_string(numThreads)).append("\n");
   }
   if (numRows) {
      configString.append("NumRows:").append(std::to_string(numRows)).append("\n");
   }
   if (numColumns) {
      configString.append("NumColumns:").append(std::to_string(numColumns)).append("\n");
   }
   if (batchWidth) {
      configString.append("BatchWidth:").append(std::to_string(batchWidth)).append("\n");
   }
   if (dryRun) {
      configString.append("DryRun:true\n");
   }

   std::istringstream configStream{configString};
   Arguments::resetState(configStream, allowUnrecognizedArguments);
}

} /* namespace PV */
