/*
 * PVArguments.hpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#ifndef PV_ARGUMENTS_HPP_
#define PV_ARGUMENTS_HPP_

namespace PV {

/**
 * A class for parsing the command line arguments and storing the results.
 * Internally, the PV_Arguments object contains a require-return flag,
 * an output path string, a params file string, a log file string,
 * a gpu devices string, a working directory string, a checkpointRead directory string,
 * a restart flag, an unsigned integer indicating a random seed,
 * and integers indicating the number of threads, the number of MPI rows,
 * the number of MPI columns, and the batch width.
 * After initialization, individual settings can be modified with set-methods,
 * or reset to the original argc/argv settings.
 *
 * It is an error to set both the restart flag and the checkpointRead directory string.
 * PV_Arguments does not check whether directory strings point at existing directories,
 * or do any other sanity checking of the arguments.
 * Typically under MPI, each mpi process will call PV_Init with the same
 * arguments, and each process's PV_Init object creates a PV_Arguments object
 * that it, HyPerCol, and other objects can use to get the command-line arguments.
 */
class PV_Arguments {
public:
   /**
    * The standard constructor for PV_Arguments.
    * The strings argv[0], argv[1], ..., argv[argc-1] are processed as follows:
    * argv[0] is ignored.
    * argv[1] through argv[argc-1] are scanned for an exact match with each of the following:
    *    "-c": the next argument is copied into the checkpointRead directory string.
    *    "-d": the next argument is copied into the gpu devices argument.
    *    "-l": the next argument is copied into the log file string.
    *    "-o": the next argument is copied into the output path string.
    *    "-p": the next argument is copied into the params file string.
    *    "-r": the restart flag is set to true.
    *    "-s": the next argument is parsed as an unsigned integer and stored into the random seed.
    *    "-t": the next argument is parsed as an integer and stored as the number of threads.
    *    "-w": the next argument is copied into the working directory string.
    *    "-rows": the next argument is parsed as an integer and stored as the number of MPI rows.
    *    "-columns": the next argument is parsed as an integer and stored as the number of MPI columns.
    *    "-batchwidth": the next argument is parsed as an integer and stored as the batch width.
    *    "--require-return": the require-return flag is set to true.
    * If the last argument is "-t" and "-t" did not appear earlier, set numThreads to the max possible
    *    (omp_get_max_threads() if PV_USE_OPENMP_THREADS is on; 1 if PV_USE_OPENMP_THREADS is off).
    * It is an error to have both the -r and -c options.
    *
    * Note that all arguments have a single hyphen, except for "--require-return".
    *
    * If an option depends on the next argument but there is no next argument, the corresponding internal variable is not set,
    * with the exception of "-t", as noted above.
    *
    * If an option occurs more than once, only the first occurrence is used to set the value.
    *
    * If allowUnrecognizedArguments is set to false, the constructor fails if any string
    * in the argv array is not recoginzed.
    */
   PV_Arguments(int argc, char * argv[], bool allowUnrecognizedArguments);

   /*
    * The destructor for PV_Arguments.
    */
   virtual ~PV_Arguments();

   /**
    * Returns the string passed as argv[0] to the constructor,
    * typically the name of the program being run.
    */
   char const * getProgramName() { return args ? args[0] : NULL; }

   /**
    * getArgument(i) returns the string passed as argv[i] to the constructor.
    * If i is out of bounds, returns null.
    */
   char const * getArgument(int i) { return (i>=0 && i<numArgs) ? args[i] : NULL; }

   /**
    * Returns a read-only pointer to the args array.
    * Note that getArgsConst()[i] and getArgsConst()[i][j] are both const;
    * this method does not provide a way to modify the args array.
    */

   char const * const * getArgsConst() { return args; }
   /**
    * Returns a copy of the args array.  It uses malloc and strdup, so the caller
    * is responsible for freeing getArgs()[k] for each k and for freeing getArgs()
    * (the simplest way to free all the memory at once is to call the
    * static method PV_Argument::freeArgs)
    * The length of the returned array is argc+1, and getArgs()[argc] is NULL.
    */
   char ** getArgsCopy();

   /**
    * Deallocates an array, assuming it was created by a call to getArgs().
    * It frees argv[0], argv[1], ..., argv[argc-1], and then frees argv.
    */
   static void freeArgs(int argc, char ** argv);

   /**
    * Returns the length of the array returned by getUnusedArgArray(); i.e. the argc argument passed to the constructor.
    */
   int getNumArgs() { return numArgs; }

   /**
    * Returns true if the require-return flag was set.
    */
   bool getRequireReturnFlag() { return requireReturnFlag; }

   /**
    * Returns the output path string.
    */
   char const * getOutputPath() { return outputPath; }

   /**
    * Returns the params file string.
    */
   char const * getParamsFile() { return paramsFile; }

   /**
    * Returns the log file string.
    */
   char const * getLogFile() { return logFile; }

   /**
    * Returns the gpu devices string.
    */
   char const * getGPUDevices() { return gpuDevices; }

   /**
    * Returns the random seed.
    */
   unsigned int getRandomSeed() { return randomSeed; }

   /**
    * Returns the working directory string.
    */
   char const * getWorkingDir() { return workingDir; }

   /**
    * Returns true if the restart flag was set.
    */
   bool getRestartFlag() { return restartFlag; }

   /**
    * Returns the checkpointRead directory string.
    */
   char const * getCheckpointReadDir() { return checkpointReadDir; }

   /**
    * Returns the useDefaultNumThreads flag.
    */
   bool getUseDefaultNumThreads() { return useDefaultNumThreads; }

   /**
    * Returns the number of threads.
    */
   int getNumThreads() { return numThreads; }

   /**
    * Returns the number of rows.
    */
   int getNumRows() { return numRows; }

   /**
    * Returns the number of columns.
    */
   int getNumColumns() { return numColumns; }

   /**
    * Returns the batch width.
    */
   int getBatchWidth() { return batchWidth; }

   /**
    * Sets the value of the require-return flag.
    * The return value is the new value of the require-return flag.
    */
   bool setRequireReturnFlag(bool val);

   /**
    * Sets the value of the output path string.
    * The return value is the new value of the output path string.
    */
   char const * setOutputPath(char const * val);

   /**
    * Sets the value of the params file string to a copy of the input argument.
    * If there is an error setting the string, the old value is untouched and
    * the return value is NULL.
    * On success, the old value is deallocated and the return value is a
    * pointer to the new params file string.
    */
   char const * setParamsFile(char const * val);

   /**
    * Sets the value of the log file string to a copy of the input argument.
    * If there is an error setting the string, the old value is untouched and
    * the return value is NULL.
    * On success, the old value is deallocated and the return value is a
    * pointer to the new log file string.
    */
   char const * setLogFile(char const * val);

   /**
    * Sets the value of the gpu devices string to a copy of the input argument.
    * If there is an error setting the string, the old value is untouched and
    * the return value is NULL.
    * On success, the old value is deallocated and the return value is a
    * pointer to the new gpu devices string.
    */
   char const * setGPUDevices(char const * val);

   /**
    * Sets the value of the random seed to the input argument.  The old value is discarded.
    * The return value is the new value of the random seed.
    */
   unsigned int setRandomSeed(unsigned int val);

   /**
    * Sets the value of the working directory string to a copy of the input argument.
    * If there is an error setting the string, the old value is untouched and
    * the return value is NULL.
    * On success, the old value is deallocated and the return value is a
    * pointer to the new working directory string.
    */
   char const * setWorkingDir(char const * val);

   /**
    * Sets the value of the restart flag.
    * The return value is the new value of the restart flag.
    */
   bool setRestartFlag(bool val);

   /**
    * Sets the value of the checkpointRead directory string to a copy of the input argument.
    * If there is an error setting the string, the old value is untouched and
    * the return value is NULL.
    * On success, the old value is deallocated and the return value is a
    * pointer to the new checkpointRead directory string.
    */
   char const * setCheckpointReadDir(char const * val);

   /**
    * Sets the useDefaultNumThreads flag to the given argument.  The old value is discarded.
    * The return value is the new useDefaultNumThreads flag.
    * If the argument is true, numThreads is set to zero.
    */
   bool setUseDefaultNumThreads(bool val);

   /**
    * Sets the number of threads to the input argument.  The old value is discarded.
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
    * Sets the number of columns to the input argument.  The old value is discarded.
    * The return value is the new number of columns.
    */
   int setNumColumns(int val);

   /**
    * Sets the batch width to the input argument.  The old value is discarded.
    * The return value is the new batch width.
    */
   int setBatchWidth(int val);

   /**
    * Reinitializes the object's state based on the new argc and argv array.
    * The allowUnrecognizedArguments flag has the same effect as in the
    * constructor. Any previous pointers returned by get-methods become
    * invalid.
    */
   int resetState(int argc, char * argv[], bool allowUnrecognizedArguments);

   /**
    * Resets all member variables to their state after the object was
    * instantiated. That is, the arguments in argv are parsed but the effect
    * of any set-method is discarded.  Any previous pointers returned
    * by get-methods, except for getArgsConst, getArgs, .
    */
   int resetState();

   /**
    * Prints the effective command line based on the argc/argv arguments used in instantiation,
    * and any set-methods used since then.
    */
   int printState();

private:
   /**
    * initialize_base() is called by the constructor to initialize the internal variables
    * to false for flags, zero for integers, and NULL for strings.
    */
   int initialize_base();

   /**
    * initializeState() sets all member variables except numArgs and args
    * (the copies of argc and argv passed to the constructor) to zero or NULL.
    * It is called by initialize_base() and clearState().
    */
   int initializeState();

   /**
    * initialize() is called by the constructor to parse the argv array and set
    * internal variables accordingly.
    */
   int initialize(int argc, char * argv[], bool allowUnrecognizedArguments);

   /**
    * copyArgs() is used internally both by the constructor, to copy the argv array into a member variable,
    * and by the getArgs get-method to return a copy of the args array.
    * It returns an array of length argc+1, where the first argc strings are copied from argv
    * and the last one is NULL (i.e. copyArgs()[argc]==NULL).
    */
   static char ** copyArgs(int argc, char const * const * argv);

   /**
    * setString() is used internally as a common interface for setting the internal string variables.
    * If it encounters an error, it prints an error message and returns NULL with the old value of
    * the parameter untouched.
    */
   char const * setString(char ** parameter, char const * string, char const * parameterName);

   /**
    * errorSettingString() is used internally to provide a common interface for reporting an error encountered by setString.
    * It uses the value of errno.
    */
   int errorSettingString(char const * parameterName, char const * attempted_value);

   /**
    * clearState() frees all memory allocated for member variables except for the copy of argv, and resets
    * all member variables to their
    */
   int clearState();

   int setStateFromCmdLineArgs(bool allowUnrecognizedArguments);

   /**
    * prints a message to stderr and returns PV_FAILURE if any error in the state is detected;
    * returns PV_SUCCESS otherwise.
    * Currently, it only checks whether restartFlag and checkpointReadDir are not both set.
    */
   int errorChecking();

   // Member variables
private:
   int numArgs;
   char ** args;
   bool requireReturnFlag;
   char * outputPath;
   char * paramsFile;
   char * logFile;
   char * gpuDevices;
   unsigned int randomSeed;
   char * workingDir;
   bool restartFlag;
   char * checkpointReadDir;
   bool useDefaultNumThreads;
   int numThreads;
   int numRows;
   int numColumns;
   int batchWidth;
};

} /* namespace PV */

#endif /* PV_ARGUMENTS_HPP_ */
