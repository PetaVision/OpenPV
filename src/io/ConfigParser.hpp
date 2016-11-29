#ifndef CONFIGPARSER_HPP_
#define CONFIGPARSER_HPP_

#include <istream>
#include <string>

namespace PV {

class ConfigParser {
public:

   ConfigParser(std::istream &configStream, bool allowUnrecognizedArguments);

   // get-methods
   bool getAllowUnrecognizedArguments() const { return mAllowUnrecognizedArguments; }
   bool getRequireReturn() const { return mRequireReturn; }
   bool getRestart() const { return mRestart; }
   bool getDryRun() const { return mDryRun; }
   unsigned int getRandomSeed() const { return mRandomSeed; }
   bool getUseDefaultNumThreads() const { return mUseDefaultNumThreads; }
   int getNumThreads() const { return mNumThreads; }
   int getNumRows() const { return mNumRows; }
   int getNumColumns() const { return mNumColumns; }
   int getBatchWidth() const { return mBatchWidth; }
   std::string const &getOutputPath() const { return mOutputPath; }
   std::string const &getParamsFile() const { return mParamsFile; }
   std::string const &getLogFile() const { return mLogFile; }
   std::string const &getGpuDevices() const { return mGpuDevices; }
   std::string const &getWorkingDir() const { return mWorkingDir; }
   std::string const &getCheckpointReadDir() const { return mCheckpointReadDir; }

private:

   void initialize(std::istream &inputStream, bool allowUnrecognizedArguments);

bool parseBoolean(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      bool *returnValue);

bool parseInteger(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue);

bool parseUnsignedInt(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      unsigned int *returnValue);

bool parseString(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      std::string *returnValue);

bool parseIntOptional(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue,
      bool *usingDefault);

std::string stripLeadingTrailingWhitespace(std::string const &inString);

private:
      bool mAllowUnrecognizedArguments = false;

      // parsed values
      bool mRequireReturn = false;
      bool mRestart = false;
      bool mDryRun = false;
      unsigned int mRandomSeed = 0U;
      bool mUseDefaultNumThreads = false;
      int mNumThreads = -1;
      int mNumRows = 0;
      int mNumColumns = 0;
      int mBatchWidth = 0;
      std::string mOutputPath;
      std::string mParamsFile;
      std::string mLogFile;
      std::string mGpuDevices;
      std::string mWorkingDir;
      std::string mCheckpointReadDir;
};

} // end namespace PV

#endif // CONFIGPARSER_HPP_
