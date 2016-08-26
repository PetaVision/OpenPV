/*
 * PVArguments.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#include <cstdlib>
#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif
#include <string.h>
#include "PV_Arguments.hpp"
#include "io/io.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

PV_Arguments::PV_Arguments(int argc, char * argv[], bool allowUnrecognizedArguments) {
   initialize_base();
   initialize(argc, argv, allowUnrecognizedArguments);
}

int PV_Arguments::initialize_base() {
   initializeState();
   numArgs = 0;
   args = NULL;
   return PV_SUCCESS;
}

int PV_Arguments::initializeState() {
   requireReturnFlag = false;
   outputPath = NULL;
   paramsFile = NULL;
   logFile = NULL;
   gpuDevices = NULL;
   randomSeed = 0U;
   workingDir = NULL;
   restartFlag = false;
   checkpointReadDir = NULL;
   useDefaultNumThreads = false;
   numThreads = 0;
   numRows = 0;
   numColumns = 0;
   batchWidth = 0;
   dryRunFlag = false;
   return PV_SUCCESS;
}

int PV_Arguments::initialize(int argc, char * argv[], bool allowUnrecognizedArguments) {
   if (argc<=0) {
      pvError().printf("PV_Arguments: argc must be positive (called with argc=%d)\n", argc);
   }
   numArgs = argc;
   args = copyArgs(argc, argv);
   return setStateFromCmdLineArgs(allowUnrecognizedArguments);
}

char ** PV_Arguments::copyArgs(int argc, char const * const * argv) {
   char ** argumentArray = (char **) malloc((size_t) (argc+1) * sizeof(char *));
   if (argumentArray==NULL) {
      pvError().printf("PV_Arguments error: unable to allocate memory for %d arguments: %s\n",
            argc, strerror(errno));
   }
   for (int a=0; a<argc; a++) {
      char const * arga = argv[a];
      if (arga) {
         char * copied = strdup(arga);
         if (!copied) {
            pvErrorNoExit().printf("PV_Arguments unable to store argument %d: %s\n", a, strerror(errno));
            pvError().printf("Argument was \"%s\".\n", arga);
         }
         argumentArray[a] = copied;
      }
      else {
         argumentArray[a] = NULL;
      }
   }
   argumentArray[argc] = NULL;
   return argumentArray;
}

void PV_Arguments::freeArgs(int argc, char ** argv) {
   for (int k=0; k<argc; k++) { free(argv[k]); }
   free(argv);
   return;
}

char ** PV_Arguments::getArgsCopy() const {
   return copyArgs(numArgs, args);
}

bool PV_Arguments::setRequireReturnFlag(bool val) {
   requireReturnFlag = val;
   return requireReturnFlag;
}
char const * PV_Arguments::setOutputPath(char const * val) {
   return setString(&outputPath, val, "output path");
}
char const * PV_Arguments::setParamsFile(char const * val) {
   return setString(&paramsFile, val, "params file");
}
char const * PV_Arguments::setLogFile(char const * val) {
   return setString(&logFile, val, "log file");
}
char const * PV_Arguments::setGPUDevices(char const * val) {
   return setString(&gpuDevices, val, "GPU devices string");
}
unsigned int PV_Arguments::setRandomSeed(unsigned int val) {
   randomSeed = val;
   return randomSeed;
}
char const * PV_Arguments::setWorkingDir(char const * val) {
   return setString(&workingDir, val, "working directory");
}
bool PV_Arguments::setRestartFlag(bool val) {
   restartFlag = val;
   return restartFlag;
}
char const * PV_Arguments::setCheckpointReadDir(char const * val) {
   return setString(&checkpointReadDir, val, "checkpointRead directory");
}
bool PV_Arguments::setUseDefaultNumThreads(bool val) {
   useDefaultNumThreads = val;
   if(val) numThreads = 0;
   return useDefaultNumThreads;
}
int PV_Arguments::setNumThreads(int val) {
   numThreads = val;
   useDefaultNumThreads = false;
   return numThreads;
}
int PV_Arguments::setNumRows(int val) {
   numRows = val;
   return numRows;
}
int PV_Arguments::setNumColumns(int val) {
   numColumns = val;
   return numColumns;
}
int PV_Arguments::setBatchWidth(int val) {
   batchWidth = val;
   return batchWidth;
}
bool PV_Arguments::setDryRunFlag(bool val) {
   dryRunFlag = val;
   return dryRunFlag;
}

char const * PV_Arguments::setString(char ** parameter, char const * string, char const * parameterName) {
   int status = PV_SUCCESS;
   char * newParameter = NULL;
   if (string!=NULL) {
      newParameter = strdup(string);
      if (newParameter==NULL) {
         pvError().printf("PV_Arguments error setting %s to \"%s\": %s\n",
               parameterName, string, strerror(errno));
         status = PV_FAILURE;
      }
   }
   if (status==PV_SUCCESS) {
      free(*parameter);
      *parameter = newParameter;
   }
   return newParameter;
}

int PV_Arguments::resetState(int argc, char * argv[], bool allowUnrecognizedArguments) {
   int status = clearState();
   pvAssert(status == PV_SUCCESS);
   freeArgs(numArgs, args); args = NULL;
   return initialize(argc, argv, allowUnrecognizedArguments);
}

int PV_Arguments::resetState() {
   int status = clearState();
   pvAssert(status == PV_SUCCESS);
   return setStateFromCmdLineArgs(true);
   /* If unrecognized arguments were not allowed in the constructor and there were unrecognized args in argv,
    * the error would have taken place during the constructor. */
}

int PV_Arguments::clearState() {
   requireReturnFlag = false;
   free(outputPath); outputPath = NULL;
   free(paramsFile); paramsFile = NULL;
   free(logFile); logFile = NULL;
   free(gpuDevices); gpuDevices = NULL;
   randomSeed = 0U;
   free(workingDir); workingDir = NULL;
   restartFlag = false;
   free(checkpointReadDir); checkpointReadDir = NULL;
   numThreads = 0;
   numRows = 0;
   numColumns = 0;
   batchWidth = 0;
   return PV_SUCCESS;
}

int PV_Arguments::setStateFromCmdLineArgs(bool allowUnrecognizedArguments) {
   pvAssert(numArgs>0);
   bool * usedArgArray = (bool *) calloc((size_t) numArgs, sizeof(bool));
   if (usedArgArray==NULL) {
      pvError().printf("PV_Arguments::setStateFromCmdLineArgs unable to allocate memory for usedArgArray: %s\n", strerror(errno));
   }
   usedArgArray[0] = true; // Always use the program name

   int restart = (int) restartFlag;
   int dryrun = (int) dryRunFlag;
   int status = parse_options(numArgs, args,
         usedArgArray, &requireReturnFlag, &outputPath, &paramsFile, &logFile,
         &gpuDevices, &randomSeed, &workingDir,
         &restart, &checkpointReadDir, &useDefaultNumThreads, &numThreads,
         &numRows, &numColumns, &batchWidth, &dryrun);
   restartFlag = restart!=0;
   dryRunFlag = dryrun!=0;

   // Error out if both -r and -c are used
   if (restartFlag && checkpointReadDir) {
      pvError().printf("PV_Arguments: cannot set both the restart flag and the checkpoint read directory.\n");
   }

   if (!allowUnrecognizedArguments) {
      bool anyUnusedArgs = false;
      for (int a=0; a<numArgs; a++) {
         if(!usedArgArray[a]) {
            pvErrorNoExit().printf("%s: argument %d, \"%s\", is not recognized.\n",
                  getProgramName(), a, args[a]);
            anyUnusedArgs = true;
         }
      }
      if (anyUnusedArgs) {
         exit(EXIT_FAILURE);
      }
   }
   free(usedArgArray);
   return PV_SUCCESS;
}

int PV_Arguments::printState() const {
   pvInfo().printf("%s", getProgramName());
   if (requireReturnFlag) { pvInfo().printf(" --require-return"); }
   if (outputPath) { pvInfo().printf(" -o %s", outputPath); }
   if (paramsFile) { pvInfo().printf(" -p %s", paramsFile); }
   if (logFile) { pvInfo().printf(" -l %s", logFile); }
   if (gpuDevices) { pvInfo().printf(" -d %s", gpuDevices); }
   if (randomSeed) { pvInfo().printf(" -s %u", randomSeed); }
   if (workingDir) { pvInfo().printf(" -w %s", workingDir); }
   pvAssert(!(restartFlag && checkpointReadDir));
   if (restartFlag) { pvInfo().printf(" -r"); }
   if (checkpointReadDir) { pvInfo().printf(" -c %s", checkpointReadDir); }
   if (numThreads>=0) { pvInfo().printf(" -t %d", numThreads); }
   if (numRows) { pvInfo().printf(" -rows %d", numRows); }
   if (numColumns) { pvInfo().printf(" -columns %d", numColumns); }
   if (batchWidth) { pvInfo().printf(" -batchwidth %d", batchWidth); }
   if (dryRunFlag) { pvInfo().printf(" -n"); }
   pvInfo().printf("\n");
   return PV_SUCCESS;
}

PV_Arguments::~PV_Arguments() {
   for(int a=0; a<numArgs; a++) {
      free(args[a]);
   }
   free(args);
   clearState();
}

} /* namespace PV */
