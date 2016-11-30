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

ConfigParser::ConfigParser(
      std::istream &configStream,
      bool allowUnrecognizedArguments) {
   initialize(configStream, allowUnrecognizedArguments);
   if (!configStream.eof() && configStream.fail()) {
      Fatal() << "Reading configuration stream failed.\n";
   }
}

void ConfigParser::initialize(
      std::istream &configStream,
      bool allowUnrecognizedArguments) {
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
            "configuration line \"%s\" does not have the format \"argument:value\".\n");

      std::string argument{start, line.begin() + colonPosition};
      argument = stripLeadingTrailingWhitespace(argument);
      std::string value{line.begin() + colonPosition + 1, line.end()};
      value = stripLeadingTrailingWhitespace(value);
      bool found = false;
      try {
         found |= parseBoolean(argument, "RequireReturn", value, &mRequireReturn);
         found |= parseString(argument, "OutputPath", value, &mOutputPath);
         found |= parseString(argument, "ParamsFile", value, &mParamsFile);
         found |= parseString(argument, "LogFile", value, &mLogFile);
         found |= parseString(argument, "GPUDevices", value, &mGpuDevices);
         found |= parseUnsignedInt(argument, "RandomSeed", value, &mRandomSeed);
         found |= parseString(argument, "WorkingDirectory", value, &mWorkingDir);
         found |= parseBoolean(argument, "Restart", value, &mRestart);
         found |= parseString(argument, "CheckpointReadDirectory", value, &mCheckpointReadDir);
         found |= parseIntOptional(argument, "NumThreads", value, &mNumThreads, &mUseDefaultNumThreads);
         found |= parseInteger(argument, "NumRows", value, &mNumRows);
         found |= parseInteger(argument, "NumColumns", value, &mNumColumns);
         found |= parseInteger(argument, "BatchWidth", value, &mBatchWidth);
         found |= parseBoolean(argument, "DryRun", value, &mDryRun);
      } catch (...) {
         Fatal() << "Unable to parse config file line number " << lineNumber << ", " << argument
                 << ":" << value << "\n";
      }
      if (!(found || mAllowUnrecognizedArguments)) {
         Fatal() << "allowUnrecognizedArguments is false but line number " << lineNumber << " (" <<  argument << ":" << value << ") is not a recognized argument.\n";
      }
   }
   FatalIf(mRestart && !mCheckpointReadDir.empty(), "ConfigParser: cannot set both the restart flag and the checkpoint read directory.\n");
}

bool ConfigParser::parseBoolean(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      bool *returnValue) {
   bool found = false;
   if (argument == selectedArgument) {
      if (configValue == "true" || configValue == "T" || configValue == "1") {
         found = true;
         *returnValue = true;
      }
      else if (configValue == "false" || configValue == "F" || configValue == "0") {
         found = true;
         *returnValue = false;
      }
      else {
         throw std::invalid_argument("parseBoolean");
      }
   }
   return found;
}

bool ConfigParser::parseInteger(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue) {
   bool found = false;
   if (argument == selectedArgument) {
      found = true;
      *returnValue = std::stoi(configValue);
   }
   return found;
}

bool ConfigParser::parseUnsignedInt(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      unsigned int *returnValue) {
   bool found = false;
   if (argument == selectedArgument) {
      found = true;
      *returnValue = (unsigned int)std::stoul(configValue);
   }
   return found;
}

bool ConfigParser::parseString(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      std::string *returnValue) {
   bool found = false;
   if (argument == selectedArgument) {
      found = true;
      *returnValue = configValue.c_str();
   }
   return found;
}

bool ConfigParser::parseIntOptional(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue,
      bool *usingDefault) {
   bool found = false;
   if (argument == selectedArgument) {
      found = true;
      if (configValue.empty() || configValue == "-") {
         *returnValue  = -1;
         *usingDefault = true;
      }
      else {
         *returnValue  = std::stoi(configValue);
         *usingDefault = false;
      }
   }
   return found;
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
   FatalIf(restartFlag && !checkpointReadDir.empty(), "ConfigParser::createStream called with both Restart and CheckpointReadDirectory set.\n");
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
