#ifndef CONFIGPARSER_HPP_
#define CONFIGPARSER_HPP_

#include "io/Configuration.hpp"
#include <istream>
#include <string>

namespace PV {

/**
 * A class to parse an input stream for configuration options.
 * The main data member is a Configuration object. ConfigParser
 * has get-methods which are passed to the Configuration object,
 * but it does not have set-methods because the object, once
 * instantiated should always reflect the configuration of the
 * input stream. The intended use case is to allow the Arguments
 * class to modify some configuration settings but be able, at
 * any time, to reset to the state from the original stream
 * (using the Arguments::resetState(void) method).
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
    * If the configuration recognizes the string in 'name' as a boolean
    * argument, this method returns the value of that argument.
    * If the name is not a boolean argument, throws an invalid_argument
    * exception.
    */
   bool const &getBooleanArgument(std::string const &name) const {
      return mConfig.getBooleanArgument(name);
   }

   /**
    * If the configuration recognizes the string in 'name' as an integer
    * argument, this method returns the value of that argument.
    * If the name is not an integer argument, throws an invalid_argument
    * exception.
    */
   int const &getIntegerArgument(std::string const &name) const {
      return mConfig.getIntegerArgument(name);
   }

   /**
    * If the configuration recognizes the string in 'name' as an unsigned
    * integer argument, this method returns the value of that argument.
    * If the name is not an unsigned integer argument, throws an
    * invalid_argument exception.
    */
   unsigned int const &getUnsignedIntArgument(std::string const &name) const {
      return mConfig.getUnsignedIntArgument(name);
   }

   /**
    * If the configuration recognizes the string in 'name' as a string
    * argument, this method returns the value of that argument.
    * If the name is not a string argument, throws an
    * invalid_argument exception.
    */
   std::string getStringArgument(std::string const &name) const {
      return mConfig.getStringArgument(name);
   }

   /**
    * If the configuration recognizes the string in 'name' as an integer
    * argument with a default option, this method returns the value of that
    * argument. If the name is not an integer-with-default argument,
    * throws an invalid_argument exception.
    */
   Configuration::IntOptional getIntOptionalArgument(std::string const &name) const {
      return mConfig.getIntOptionalArgument(name);
   }

   /**
    * Returns a string consisting of all the configuration settings.
    * If this string is converted to an input stream, it can be used to
    * instantiate another ConfigParser object.
    */
   std::string printConfig() const { return mConfig.printConfig(); }

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

   /**
    * Returns a constant reference to the underlying Configuration object.
    */
   Configuration const &getConfig() const { return mConfig; }

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
    * If allowUnrecognizedArguments is true, this routine does nothing.
    * If it is false, it exits with an error message containing the arguments.
    */
   void handleUnrecognized(std::string const &argument, std::string const &value, int linenumber);

   /**
    * This method returns a string whose contents are the input string,
    * with any whitespace at the beginning or end of the string removed.
    * Whitespace in the middle of the string remains untouched.
    */
   std::string stripLeadingTrailingWhitespace(std::string const &inString);

  private:
   bool mAllowUnrecognizedArguments = false;
   Configuration mConfig;
};

} // end namespace PV

#endif // CONFIGPARSER_HPP_
