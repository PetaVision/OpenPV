/*
 * PVArguments.hpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#ifndef ARGUMENTS_HPP_
#define ARGUMENTS_HPP_

#include "io/ConfigParser.hpp"
#include <string>

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
class Arguments {
  public:
   /**
    * The constructor for Arguments. The arguments are passed to initialize().
    */
   Arguments(std::istream &configStream, bool allowUnrecognizedArguments);

   /*
    * The destructor for Arguments.
    */
   virtual ~Arguments();

   /**
    * Returns true if the require-return flag was set.
    */
   bool getRequireReturnFlag() const { return mRequireReturnFlag; }

   /**
    * Returns the output path string.
    */
   std::string const &getOutputPath() const { return mOutputPath; }

   /**
    * Returns the params file string.
    */
   std::string const &getParamsFile() const { return mParamsFile; }

   /**
    * Returns the log file string.
    */
   std::string const &getLogFile() const { return mLogFile; }

   /**
    * Returns the gpu devices string.
    */
   std::string const &getGPUDevices() const { return mGpuDevices; }

   /**
    * Returns the random seed.
    */
   unsigned int getRandomSeed() const { return mRandomSeed; }

   /**
    * Returns the working directory string.
    */
   std::string const &getWorkingDir() const { return mWorkingDir; }

   /**
    * Returns true if the restart flag was set.
    */
   bool getRestartFlag() const { return mRestartFlag; }

   /**
    * Returns the checkpointRead directory string.
    */
   std::string const &getCheckpointReadDir() const { return mCheckpointReadDir; }

   /**
    * Returns the useDefaultNumThreads flag.
    */
   bool getUseDefaultNumThreads() const { return mUseDefaultNumThreads; }

   /**
    * Returns the number of threads.
    */
   int getNumThreads() const { return mNumThreads; }

   /**
    * Returns the number of rows.
    */
   int getNumRows() const { return mNumRows; }

   /**
    * Returns the number of columns.
    */
   int getNumColumns() const { return mNumColumns; }

   /**
    * Returns the batch width.
    */
   int getBatchWidth() const { return mBatchWidth; }

   /**
    * Returns true if the dry-run flag was set.
    */
   bool getDryRunFlag() const { return mDryRunFlag; }

   /**
    * Sets the value of the require-return flag.
    * The return value is the new value of the require-return flag.
    */
   bool setRequireReturnFlag(bool val);

   /**
    * Sets the value of the output path string.
    */
   void setOutputPath(char const *val);

   /**
    * Sets the value of the params file string to a copy of the input argument.
    */
   void setParamsFile(char const *val);

   /**
    * Sets the value of the log file string to a copy of the input argument.
    */
   void setLogFile(char const *val);

   /**
    * Sets the value of the gpu devices string to a copy of the input argument.
    */
   void setGPUDevices(char const *val);

   /**
    * Sets the value of the random seed to the input argument.  The old value is
    * discarded.
    * The return value is the new value of the random seed.
    */
   unsigned int setRandomSeed(unsigned int val);

   /**
    * Sets the value of the working directory string to a copy of the input
    * argument.
    */
   void setWorkingDir(char const *val);

   /**
    * Sets the value of the restart flag.
    * The return value is the new value of the restart flag.
    */
   bool setRestartFlag(bool val);

   /**
    * Sets the value of the checkpointRead directory string to a copy of the
    * input argument.
    */
   void setCheckpointReadDir(char const *val);

   /**
    * Sets the useDefaultNumThreads flag to the given argument.  The old value is
    * discarded.
    * The return value is the new useDefaultNumThreads flag.
    * If the argument is true, numThreads is set to zero.
    */
   bool setUseDefaultNumThreads(bool val);

   /**
    * Sets the number of threads to the input argument.  The old value is
    * discarded.
    * The return value is the new number of threads.
    * Additionally, the useDefaultNumThreads flag is set to false.
    */
   int setNumThreads(int val);

   /**
    * Sets the number of rows to the input argument.  The old value is discarded.
    * The return value is the new number of rows.
    */
   int setNumRows(int val);

   /**
    * Sets the number of columns to the input argument.  The old value is
    * discarded.
    * The return value is the new number of columns.
    */
   int setNumColumns(int val);

   /**
    * Sets the batch width to the input argument.  The old value is discarded.
    * The return value is the new batch width.
    */
   int setBatchWidth(int val);

   /**
    * Sets the dry-run flag to the new argument.  The old value is discarded.
    * The return value is the new value of the dry-run flag.
    *
    */
   bool setDryRunFlag(bool val);

   /**
    * Reinitializes the object's state based on the given input stream.
    * The allowUnrecognizedArguments flag has the same effect as in the
    * constructor. Any previous pointers returned by get-methods become
    * invalid.
    */
   void resetState(std::istream &configStream, bool allowUnrecognizedArguments);

   /**
    * Resets all member variables to their state at the time the object was
    * instantiated. That is, the arguments in argv are parsed but the effect
    * of any set-method that had been called is discarded.  Any previous
    * pointers returned by get-methods, except for getArgs or getArgsCopy
    * are no longer valid.
    */
   void resetState();

   /**
    * Prints the effective command line based on the arguments used in
    * instantiation, and any set-methods used since then.
    */
   int printState() const;

  protected:
   /**
    * The constructor that should be called by classes derived from Arguments.
    * It calls initialize_base() but not initialize. The derived class's
    * own initialization should call Arguments::initialize().
    */
   Arguments() { initialize_base(); }

   /**
    * initialize() is called by the constructor, and sets the data members.
    * The given config stream is parsed using ParseConfigFile().
    * The lines of the config file are processed as follows:
    * Lines consisting of all whitespace or whose first nonwhitespace
    * character is '#' are ignored.
    * Other lines must be in the format
    * argumentName ':' argumentValue 
    * argumentName and argumentValue can have leading or trailing whitespace,
    * which is ignored.
    * The recognized argumentNames are:
    *    RequireReturn
    *    OutputPath
    *    ParamsFile
    *    LogFile
    *    GPUDevices
    *    RandomSeed
    *    WorkingDirectory
    *    Restart
    *    CheckpointReadDirectory
    *    NumThreads
    *    NumRows
    *    NumColumns
    *    BatchWidth
    *    DryRun
    * The argument values can be retrieved using getRequireReturn(), etc.
    *
    * RequireReturn, Restart, and DryRun are boolean (see parseBoolean()).
    * RandomSeed is an unsigned integer (see parseUnsignedInt()).
    * NumRows, NumColumns, and BatchWidth are integers (see parseInteger()).
    * OutputPath, ParamsFile, LogFile, WorkingDirectory, and
    *    CheckpointReadDirectory are strings (see parseString()).
    * NumThreads can be either an integer, the empty string, or the string "-".
    *    In the latter two cases, getUseDefaultNumThreads() returns true
    *    and getNumThreads() returns zero. The expectation is that the calling
    *    routine will determine the optimal number of threads.
    *    If NumThreads parses as an integer, getUseDefaultNumThreads() retuns
    *    false and getNumThreads() returns the given value.
    *
    * It is an error for the config file to set both Restart and
    *    CheckpointReadDirectory
    *
    * If an argument occurs more than once, later instances supersede earlier
    * instances.
    *
    * If allowUnrecognizedArguments is set to false, the constructor fails if any
    * argument in the config file is not recognized.
    * If it is set to true, any unrecognized arguments are silently skipped.
    */
   int initialize(std::istream &configStream, bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the internal
    * variables
    * to false for flags, zero for integers, and nullptr for strings.
    */
   int initialize_base();

   /**
    * clearState() frees all memory allocated for member variables except for the
    * copy of argv, and resets all member variables to their default values
    * (booleans to false, integers to zero, strings to empty).
    */
   void clearState();

   // Member variables
  private:
   ConfigParser *mConfigFromStream = nullptr;
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

#endif /* ARGUMENTS_HPP_ */
