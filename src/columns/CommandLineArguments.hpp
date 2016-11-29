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
 * A class for parsing a config file and storing the results.
 * Internally, the Arguments object contains a require-return flag,
 * an output path string, a params file string, a log file string,
 * a gpu devices string, a working directory string, a checkpointRead directory
 * string,
 * a restart flag, an unsigned integer indicating a random seed,
 * and integers indicating the number of threads, the number of MPI rows,
 * the number of MPI columns, and the batch width.
 * After initialization, individual settings can be modified with set-methods,
 * or reset to the original argc/argv settings.
 *
 * It is an error to set both the restart flag and the checkpointRead directory
 * string.
 * Arguments does not check whether directory strings point at existing
 * directories,
 * or do any other sanity checking of the arguments.
 * Typically under MPI, each mpi process will call PV_Init with the same
 * arguments, and each process's PV_Init object creates a Arguments object
 * that it, HyPerCol, and other objects can use to get the command-line
 * arguments.
 */
class CommandLineArguments : public Arguments {
  public:
   /**
    * The constructor for CommandLineArguments.
    * The given argc and argv are assembled into a config-file like stream and
    * parsed using Argument::initialize().
    */
   CommandLineArguments(int argc, char *argv[], bool allowUnrecognizedArguments);

   /*
    * The destructor for Arguments.
    */
   virtual ~CommandLineArguments() {}

   /**
    * Reinitializes the object's state based on the given argv array.
    * The allowUnrecognizedArguments flag has the same effect as in the
    * constructor. Any previous pointers returned by get-methods become
    * invalid.
    */
   void resetState(int argc, char *argv[], bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the internal
    * variables to false for flags, zero for integers, and empty for strings.
    */
   int initialize_base();

   /**
    * initialize() is called by the constructor internal variables accordingly.
    */
   int initialize(int argc, char *argv[], bool allowUnrecognizedArguments);

   // Member variables
  private:
   int mArgC;
   std::vector<std::string> mArgV;
   bool mRequireReturnFlag;
   bool mRestartFlag;
   bool mDryRunFlag;
   unsigned int mRandomSeed;
   int mNumThreads;
   bool mUseDefaultNumThreads;
   int mNumRows;
   int mNumColumns;
   int mBatchWidth;
   std::string mOutputPath;
   std::string mParamsFile;
   std::string mLogFile;
   std::string mGpuDevices;
   std::string mWorkingDir;
   std::string mCheckpointReadDir;
};

} /* namespace PV */

#endif /* COMMANDLINEARGUMENTS_HPP_ */
