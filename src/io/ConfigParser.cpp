/*
 * ConfigParser.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: Pete Schultz
 */

#include "ConfigParser.hpp"
#include "utils/PVLog.hpp"
#include <cctype>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace PV {

ConfigParser::ConfigParser(std::istream &configStream, bool allowUnrecognizedArguments) {
   initialize(configStream, allowUnrecognizedArguments);
   if (!configStream.eof() && configStream.fail()) {
      Fatal() << "Reading configuration stream failed.\n";
   }
}

void ConfigParser::initialize(std::istream &configStream, bool allowUnrecognizedArguments) {
   mAllowUnrecognizedArguments = allowUnrecognizedArguments;
   std::string line;
   unsigned int lineNumber = 0;
   while (std::getline(configStream, line)) {
      lineNumber++;
      auto start = line.begin();
      // seek to first non-whitespace character
      while (start != line.end() && std::isspace((int)(*start))) {
         ++start;
      }
      if (start >= line.end()) { // Skip if line is all whitespace
         continue;
      }
      if (*start == '#') { // Skip if line is a comment
         continue;
      }
      auto colonPosition = line.find(':', (std::size_t)(start - line.begin()));
      FatalIf(
            colonPosition == std::string::npos,
            "configuration line %d, \"%s\" does not have the format \"argument:value\".\n",
            lineNumber,
            line.c_str());

      std::string argument{start, line.begin() + colonPosition};
      argument = stripLeadingTrailingWhitespace(argument);
      std::string value{line.begin() + colonPosition + 1, line.end()};
      value = stripLeadingTrailingWhitespace(value);
      try {
         bool status = mConfig.setArgumentUsingString(argument, value);
         if (!status) {
            handleUnrecognized(argument, value, lineNumber);
         }
      } catch (...) {
         Fatal() << "Unable to parse config file line number " << lineNumber << ", " << argument
                 << ":" << value << "\n";
      }
   }
   bool restartFlag                    = mConfig.getBooleanArgument("Restart");
   std::string checkpointReadDirectory = mConfig.getStringArgument("CheckpointReadDirectory");
   bool checkpointReadConflict         = restartFlag && !checkpointReadDirectory.empty();
   FatalIf(
         checkpointReadConflict,
         "ConfigParser: cannot set both the restart flag and the checkpoint read directory.\n");
}

void ConfigParser::handleUnrecognized(
      std::string const &argument,
      std::string const &value,
      int lineNumber) {
   if (!mAllowUnrecognizedArguments) {
      Fatal() << "allowUnrecognizedArguments is false but line number " << lineNumber << " ("
              << argument << ":" << value << ") is not a recognized argument.\n";
   }
}

std::string ConfigParser::stripLeadingTrailingWhitespace(std::string const &inString) {
   auto start = inString.begin();
   auto stop  = inString.end();
   while (start < stop && std::isspace((int)(*start))) {
      ++start;
   }
   while (stop > start && std::isspace((int)(*(stop - 1)))) {
      --stop;
   }
   std::string outString{start, stop};
   return outString;
}

std::string ConfigParser::createString(
      bool requireReturnFlag,
      std::string const &outputPath,
      std::string const &paramsFile,
      std::string const &logFile,
      std::string const &gpuDevices,
      unsigned int randomSeed,
      std::string const &workingDir,
      bool restartFlag,
      std::string const &checkpointReadDir,
      bool useDefaultNumThreads,
      int numThreads,
      int numRows,
      int numColumns,
      int batchWidth,
      bool dryRunFlag) {
   std::string configString;
   FatalIf(
         restartFlag && !checkpointReadDir.empty(),
         "ConfigParser::createStream called with both Restart and CheckpointReadDirectory set.\n");
   if (requireReturnFlag) {
      configString.append("RequireReturn:true\n");
   }
   if (!outputPath.empty()) {
      configString.append("OutputPath:").append(outputPath).append("\n");
   }
   if (!paramsFile.empty()) {
      configString.append("ParamsFile:").append(paramsFile).append("\n");
   }
   if (!logFile.empty()) {
      configString.append("LogFile:").append(logFile).append("\n");
   }
   if (!gpuDevices.empty()) {
      configString.append("GPUDevices:").append(gpuDevices).append("\n");
   }
   if (randomSeed) {
      configString.append("RandomSeed:").append(std::to_string(randomSeed)).append("\n");
   }
   if (!workingDir.empty()) {
      configString.append("WorkingDirectory:").append(workingDir).append("\n");
   }
   if (restartFlag) {
      configString.append("Restart:true\n");
   }
   if (!checkpointReadDir.empty()) {
      configString.append("CheckpointReadDirectory:").append(checkpointReadDir).append("\n");
   }
   if (useDefaultNumThreads) {
      configString.append("NumThreads:-\n");
   }
   else if (numThreads >= 0) {
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
   if (dryRunFlag) {
      configString.append("DryRun:true\n");
   }
   return configString;
}

} // namespace PV
