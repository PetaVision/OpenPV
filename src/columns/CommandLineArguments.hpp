/*
 * PVArguments.hpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#ifndef COMMANDLINEARGUMENTS_HPP_
#define COMMANDLINEARGUMENTS_HPP_

#include "Arguments.hpp"
#include <vector>

namespace PV {

/**
 * A class for parsing a configuration from command-line arguments and
 * storing the results. It is derived from the Arguments class,
 * and converts the command-line arguments into a stringstream that
 * can be handled by the Arguments::resetState method.
 */
class CommandLineArguments : public Arguments {
  public:
   /**
    * The constructor for CommandLineArguments. It calls the initialize
    * method, which calls resetState, which converts the argv array
    * into an input stringstream that gets passed to the ConfigParser class.
    * See resetState(int, char **, bool) for details on how the argv array
    * is interpreted.
    */
   CommandLineArguments(int argc, char *argv[], bool allowUnrecognizedArguments);

   /*
    * The destructor for CommandLineArguments.
    */
   virtual ~CommandLineArguments() {}

   /**
    * Reinitializes the object's state based on the given argv array.
    * The argv array is scanned to create a string in the format understood
    * by the ConfigParser class.
    * The strings argv[0], argv[1], ..., argv[argc-1] are processed as follows:
    * argv[0] is ignored (but can be retrieved using the getProgramName method).
    * argv[1] through argv[argc-1] are scanned for an exact match with each of
    * the following:
    *    "-c": the next argument is used as the CheckpointReadDirectory string.
    *    "-d": the next argument is used as the GpuDevices string.
    *    "-l": the next argument is used as the LogFile string.
    *    "-o": the next argument is used as the OutputPath string.
    *    "-p": the next argument is used as the ParamsFile string.
    *    "-r": the line "Restart:true" is added to the configuration.
    *    "-s": the next argument is parsed as an unsigned integer and used as
    * the RandomSeed setting.
    *    "-t": if the next argument is a nonnegative integer, it is used as the
    * NumThreads setting. If the next argument is not a nonnegative argument
    * or "-t" is the last argument, "NumThreads:-" is added to the
    * configuration.
    *    "-w": the next argument is used as the WorkingDirectory string.
    *    "-rows": the next argument is parsed as an integer and used as the
    * NumRows setting.
    *    "-columns": the next argument is parsed as an integer and used as the
    * NumColumns setting.
    *    "-batchwidth": the next argument is parsed as an integer and used as
    * the BatchWidth setting.
    *    "-n": the line "DryRun:true" is added to the configuration.
    *    "--require-return": the line "RequireReturn:true" is added to the
    * configuration.
    * It is an error to have both the -r and -c options.
    *
    * Note that all arguments have a single hyphen, except for
    * "--require-return".
    *
    * If an option depends on the next argument but there is no next argument,
    * the corresponding
    * internal variable is not set,
    * with the exception of "-t", as noted above.
    *
    * If an option occurs more than once, later occurrences supersede earlier
    * occurrences.
    *
    * If allowUnrecognizedArguments is set to false, the constructor fails if any
    * string in the argv array is not recoginzed.
    */
   void resetState(int argc, char *argv[], bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the
    * internal state before initialize() is called.
    */
   int initialize_base();

   /**
    * initialize() is called by the constructor. It calls resetState
    * to set the initial values of the Arguments data members.
    */
   int initialize(int argc, char *argv[], bool allowUnrecognizedArguments);
};

} /* namespace PV */

#endif /* COMMANDLINEARGUMENTS_HPP_ */
