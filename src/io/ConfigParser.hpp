#ifndef CONFIGPARSER_HPP_
#define CONFIGPARSER_HPP_

#include <istream>
#include <string>

namespace PV {

/**
 * A class to parse an input stream for configuration options.
 */
class ConfigParser {
public:
   /**
    * The public constructor for ConfigParser.
    * configStream is an input stream with the contents of the configuration.
    * See the initialize method on how the stream contents are parsed.
    * If allowUnrecognizedArguments is true, any unrecognized configuration
    * arguments are skipped. If it is false, unrecognized arguments cause
    * an error.
    */
   ConfigParser(std::istream &configStream, bool allowUnrecognizedArguments);

   /**
    * Returns the value of allowUnrecognizedArguments used in instantiation.
    */
   bool getAllowUnrecognizedArguments() const { return mAllowUnrecognizedArguments; }

   /**
    * Returns the boolean value associated with RequireReturn in the configuration.
    */
   bool getRequireReturn() const { return mRequireReturn; }

   /**
    * Returns the boolean value associated with Restart in the configuration.
    */
   bool getRestart() const { return mRestart; }

   /**
    * Returns the boolean value associated with DryRun in the configuration.
    */
   bool getDryRun() const { return mDryRun; }

   /**
    * Returns the unsigned integer associated with RandomSeed in the configuration.
    */
   unsigned int getRandomSeed() const { return mRandomSeed; }

   /**
    * Returns true if the configuration specified to use the default number of threads;
    * false if the configuration left it unspecified or specified a specific number.
    */
   bool getUseDefaultNumThreads() const { return mUseDefaultNumThreads; }

   /**
    * Returns the integer value associated with NumThreads in the configuration.
    * If the configuration specified to use the default number of threads, or if
    * the configuration left NumThreads unspecified, this method returns -1.
    */
   int getNumThreads() const { return mNumThreads; }

   /**
    * Returns the integer associated with NumRows in the configuration.
    */
   int getNumRows() const { return mNumRows; }

   /**
    * Returns the integer associated with NumColumns in the configuration.
    */
   int getNumColumns() const { return mNumColumns; }

   /**
    * Returns the integer associated with BatchWidth in the configuration.
    */
   int getBatchWidth() const { return mBatchWidth; }

   /**
    * Returns the string associated with OutputPath in the configuration.
    */
   std::string const &getOutputPath() const { return mOutputPath; }

   /**
    * Returns the string associated with ParamsFile in the configuration.
    */
   std::string const &getParamsFile() const { return mParamsFile; }

   /**
    * Returns the string associated with LogFile in the configuration.
    */
   std::string const &getLogFile() const { return mLogFile; }

   /**
    * Returns the string associated with GpuDevices in the configuration.
    */
   std::string const &getGpuDevices() const { return mGpuDevices; }

   /**
    * Returns the string associated with WorkingDirectory in the configuration.
    */
   std::string const &getWorkingDir() const { return mWorkingDir; }

   /**
    * Returns the string associated with CheckpointReadDirectory in the configuration.
    */
   std::string const &getCheckpointReadDir() const { return mCheckpointReadDir; }

   /**
    * A static method for creating an string based on the given arguments
    * The return value can be used to instantiate a ConfigParser object.
    * Its main use is to have the same class that parses a config stream be
    * the class that generates a string in the format it expects.
    */
   static std::string createString(
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
      bool dryRunFlag);

private:

   /**
    * Called by the ConfigParser constructor.
    * Each line is read until the end of the inputStream, and processed as
    * follows:
    * Blank lines and lines consisting solely of whitespace are skipped.
    * Lines whose first nonwhitespace character is '#' (comments) are skipped.
    * The remaining lines are scanned for a colon ':'. It is an error for a
    * nonblank, noncomment line to have no colons.
    * The part of the line before the first colon, with leading and trailing
    * whitespace deleted, is the argument name.
    * The part after the first colon, with leading and trailing whitespace
    * deleted, is the argument value.
    * Recognized parameters and the method used for each are:
    *   RequireReturn (parseBoolean)
    *   OutputPath (parseString)
    *   ParamsFile (parseString)
    *   LogFile (parseString)
    *   GPUDevices (parseString)
    *   RandomSeed (parseUnsignedInt)
    *   WorkingDirectory (parseString)
    *   Restart (parseBoolean)
    *   CheckpointReadDirectory (parseString)
    *   NumThreads (parseIntOptional)
    *   NumRows (parseInteger)
    *   NumColumns (parseInteger)
    *   BatchWidth (parseInteger)
    *   DryRun (parseBoolean)
    * Any other argument names are ignored if the allowUnrecognizedArguments
    * flag is true, and cause an error if the flag is false.
    *
    * initialize() expects that if the parsing method it calls encounter
    * a bad argument value, it will throw an exception. initialize() catches
    * any such exception and exits with an error.
    */
   void initialize(std::istream &inputStream, bool allowUnrecognizedArguments);

/**
 * A method called by initialize() to read a boolean value.
 * argument is the argument name retrieved from the configuration stream.
 * selectedArgument is the argument name to check for.
 * configValue is the argument value retrieved from the configuration stream.
 * returnValue points to the variable to hold the result of parsing the value.
 *
 * If selectedArgument and argument are not the same, the program returns
 * false with returnValue unchanged.
 * If selectedArgument and argument are equal, configValue must be equal to one
 * of the following strings (case-sensitive):
 * "true", "T", "1" for true, or "false", "F", or "0" for false.
 * returnValue is set accordingly and the routine returns true.
 *
 * If configValue is not one of the above strings, parseBoolean() throws an
 * exception.
 */
bool parseBoolean(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      bool *returnValue);

/**
 * A method called by initialize() to read a integer value.
 * argument is the argument name retrieved from the configuration stream.
 * selectedArgument is the argument name to check for.
 * configValue is the argument value retrieved from the configuration stream.
 * returnValue points to the variable to hold the result of parsing the value.
 *
 * If selectedArgument and argument are not the same, the program returns
 * false with returnValue unchanged.
 * If selectedArgument and argument are equal, configValue is converted to
 * an integer using the C++11 std::stoi function. Any exceptions thrown by
 * stoi are passed up to the calling function. If stoi succeeds, the converted
 * value is placed in returnValue and the function returns true.
 */
bool parseInteger(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue);

/**
 * A method called by initialize() to read an unsigned integer value.
 * argument is the argument name retrieved from the configuration stream.
 * selectedArgument is the argument name to check for.
 * configValue is the argument value retrieved from the configuration stream.
 * returnValue points to the variable to hold the result of parsing the value.
 *
 * If selectedArgument and argument are not the same, the program returns
 * false with returnValue unchanged.
 * If selectedArgument and argument are equal, configValue is converted to
 * an integer using the C++11 std::stoul function. Any exceptions thrown by
 * stoul are passed up to the calling function. If stoul succeeds, the converted
 * value is cast to unsigned int and placed in returnValue and the function
 * returns true.
 */
bool parseUnsignedInt(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      unsigned int *returnValue);

/**
 * A method called by initialize() to read a string.
 * argument is the argument name retrieved from the configuration stream.
 * selectedArgument is the argument name to check for.
 * configValue is the argument value retrieved from the configuration stream.
 * returnValue points to the string to hold the result of parsing the value.
 *
 * If selectedArgument and argument are not the same, the program returns
 * false with returnValue unchanged.
 * If selectedArgument and argument are equal, configValue is copied into
 * *returnValue and the method returns true.
 */
bool parseString(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      std::string *returnValue);

/**
 * A method called by initialize() to read an integer, when a default
 * value can be specified. Note that the default is not handled by
 * the ConfigParser class; it is up to the calling routine to interpret
 * the meaning of the default setting.
 *
 * argument is the argument name retrieved from the configuration stream.
 * selectedArgument is the argument name to check for.
 * configValue is the argument value retrieved from the configuration stream.
 * returnValue points to the integer to hold the result of parsing the value.
 * usingDefault points to the boolean that tells whether the default
 * should be used.
 *
 * If selectedArgument and argument are not the same, the program returns
 * false with returnValue and usingDefault unchanged.
 * If selectedArgument and argument are equal, configValue must be either:
 * - a string that can be converted to an integer using std::stoi, or
 * - the empty string, or
 * - the string consisting of a single hyphen "-".
 *
 * In the first instance, usingDefault is set to false and returnValue is
 * set to the value of the string.
 * In the other two instances, usingDefault is set to true and returnValue
 * is set to -1 (although the value should not be used if usingDefault is true)
 * The method then returns true.
 */
bool parseIntOptional(
      std::string const &argument,
      char const *selectedArgument,
      std::string const &configValue,
      int *returnValue,
      bool *usingDefault);

/**
 * This method returns a string whose contents are the input string,
 * with any whitespace at the beginning or end of the string removed.
 * Whitespace in the middle of the string remains untouched.
 */
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
