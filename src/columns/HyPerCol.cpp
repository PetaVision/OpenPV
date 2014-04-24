/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#define TIMER_ON
#define TIMESTEP_OUTPUT

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../io/clock.h"
#include "../io/imageio.hpp"
#include "../io/io.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>
#include <time.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fts.h>

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[], PVParams * params) {
   initialize_base();
   initialize(name, argc, argv, params);
}

HyPerCol::~HyPerCol()
{
   int n;

#ifdef PV_USE_OPENCL
   finalizeThreads();
#endif // PV_USE_OPENCL

   if (image_file != NULL) free(image_file);

   for (n = 0; n < numConnections; n++) {
      delete connections[n];
   }

   for (n = 0; n < numLayers; n++) {
      if (layers[n] != NULL) {
         delete layers[n];
      }
   }

   if (ownsParams) delete params;

   if (ownsInterColComm) {
      delete icComm;
   }
   else {
      icComm->clearPublishers();
   }

   if (columnId()==0) {
      runTimer->fprint_time(stdout);
      fflush(stdout);
   }
   delete runTimer;

   free(connections);
   free(layers);
   for (int k=0; k<numProbes; k++) {
      delete probes[k];
   }
   free(probes);
   free(name);
   free(outputPath);
   free(printParamsFilename);
   // free(outputNamesOfLayersAndConns);
   if (checkpointWriteFlag) {
      free(checkpointWriteDir); checkpointWriteDir = NULL;
      free(checkpointWriteTriggerModeString); checkpointWriteTriggerModeString = NULL;
   }
   if (checkpointReadFlag) {
      free(checkpointReadDir); checkpointReadDir = NULL;
      free(checkpointReadDirBase); checkpointReadDirBase = NULL;
   }
}


#define DEFAULT_NUMSTEPS 1
int HyPerCol::initialize_base() {
   // Initialize all member variables to safe values.  They will be set to their actual values in initialize()
   warmStart = false;
   currentStep = 0;
   layerArraySize = INITIAL_LAYER_ARRAY_SIZE;
   numLayers = 0;
   numPhases = 0;
   connectionArraySize = INITIAL_CONNECTION_ARRAY_SIZE;
   numConnections = 0;
   checkpointReadFlag = false;
   checkpointWriteFlag = false;
   checkpointReadDir = NULL;
   checkpointReadDirBase = NULL;
   cpReadDirIndex = -1L;
   checkpointWriteDir = NULL;
   checkpointWriteTriggerMode = CPWRITE_TRIGGER_STEP;
   cpWriteStepInterval = -1L;
   nextCPWriteStep = 0L;
   cpWriteTimeInterval = -1.0;
   nextCPWriteTime = 0.0;
   deleteOlderCheckpoints = false;
   memset(lastCheckpointDir, 0, PV_PATH_MAX);
   suppressLastOutput = false;
   simTime = 0.0;
   startTime = 0.0;
   stopTime = 0.0;
   deltaTime = DELTA_T;
   deltaTimeBase = DELTA_T;
   // progressStep = 1L; // deprecated Dec 18, 2013
   progressInterval = 1.0;
   writeProgressToErr = false;
   clDevice = NULL;
   layers = NULL;
   connections = NULL;
   name = NULL;
   outputPath = NULL;
   // outputNamesOfLayersAndConns = NULL;
   printParamsFilename = NULL;
   printParamsStream = NULL;
   image_file = NULL;
   nxGlobal = 0;
   nyGlobal = 0;
   ownsParams = true;
   ownsInterColComm = true;
   params = NULL;
   icComm = NULL;
   runDelegate = NULL;
   runTimer = NULL;
   numProbes = 0;
   probes = NULL;
   numLayerProbes = 0;
   layerProbes = NULL;
   // numConnProbes = 0;
   // connProbes = NULL;
   filenamesContainLayerNames = 0;
   filenamesContainConnectionNames = 0;
   random_seed = 0;
   random_seed_obj = 0;

   return PV_SUCCESS;
}

int HyPerCol::initialize(const char * name, int argc, char ** argv, PVParams * params)
{
   ownsInterColComm = (params==NULL || params->getInterColComm()==NULL);
   if (ownsInterColComm) {
      icComm = new InterColComm(&argc, &argv);
   }
   else {
      icComm = params->getInterColComm();
   }
   int rank = icComm->commRank();

#ifdef PVP_DEBUG
   bool reqrtn = false;
   for(int arg=1; arg<argc; arg++) {
      if( !strcmp(argv[arg], "--require-return")) {
         reqrtn = true;
         break;
      }
   }
   if( reqrtn ) {
      if( rank == 0 ) {
         printf("Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while(charhit != '\n') {
            charhit = getc(stdin);
         }
      }
#ifdef PV_USE_MPI
      MPI_Barrier(icComm->communicator());
#endif // PV_USE_MPI
   }
#endif // PVP_DEBUG

   this->name = strdup(name);
   this->runTimer = new Timer(name, "column", "run    ");

   layers = (HyPerLayer **) malloc(layerArraySize * sizeof(HyPerLayer *));
   connections = (HyPerConn **) malloc(connectionArraySize * sizeof(HyPerConn *));

   int opencl_device = 0;  // default to GPU for now
   char * param_file = NULL;
   char * working_dir = NULL;
   int restart = 0;
   parse_options(argc, argv, &outputPath, &param_file,
                 &opencl_device, &random_seed, &working_dir, &restart, &checkpointReadDir);
   warmStart = (restart!=0);
   if(working_dir && columnId()==0) {
      int status = chdir(working_dir);
      if(status) {
         fprintf(stderr, "Unable to switch directory to \"%s\"\n", working_dir);
         fprintf(stderr, "chdir error: %s\n", strerror(errno));
         exit(status);
      }
   }

   ownsParams = params==NULL;
   if (ownsParams) {
      size_t groupArraySize = 2*(layerArraySize + connectionArraySize);
      params = new PVParams(param_file, groupArraySize, icComm);  // PVParams::addGroup can resize if initialGroups is exceeded
   }
   this->params = params;
   free(param_file);
   param_file = NULL;

#ifdef PV_USE_MPI // Fail if there was a parsing error, but make sure nonroot processes don't kill the root process before the root process reaches the syntax error
   int parsedStatus;
   int rootproc = 0;
   if( rank == rootproc ) {
      parsedStatus = params->getParseStatus();
   }
   MPI_Bcast(&parsedStatus, 1, MPI_INT, rootproc, icCommunicator()->communicator());
#else
   int parsedStatus = params->getParseStatus();
#endif
   if( parsedStatus != 0 ) {
      exit(parsedStatus);
   }

   ioParams(PARAMS_IO_READ);

   ensureDirExists(outputPath);

   simTime = startTime;
   initialStep = (long int) nearbyint(startTime/deltaTimeBase);
   currentStep = initialStep;
   finalStep = (long int) nearbyint(stopTime/deltaTimeBase);
   nextProgressTime = startTime + progressInterval;

   if(checkpointWriteFlag && checkpointWriteTriggerMode == CPWRITE_TRIGGER_STEP) {
      switch (checkpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         nextCPWriteStep = initialStep;
         nextCPWriteTime = startTime; // Should be unnecessary
         cpWriteTimeInterval = -1;
         break;
      case CPWRITE_TRIGGER_TIME:
         nextCPWriteStep = initialStep; // Should be unnecessary
         nextCPWriteTime = startTime;
         cpWriteStepInterval = -1;
         break;
      case CPWRITE_TRIGGER_CLOCK:
         assert(0); // Using clock time to checkpoint has not been implemented yet.
         break;
      default:
         assert(0); // All cases of checkpointWriteTriggerMode should have been covered above.
         break;
      }
   }

   if (warmStart && checkpointReadDir) {
      if (columnId()==0) {
         fprintf(stderr, "%s error: cannot set both -r and -c.\n", argv[0]);
      }
#if PV_USE_MPI
      MPI_Barrier(icComm->communicator());
#endif // PV_USE_MPI
      exit(EXIT_FAILURE);
   }
   if (warmStart) {
      // parse_options() and ioParams() must have both been called at this point, so that we have the correct outputPath and checkpointWriteFlag
      assert(checkpointReadDir==NULL);
      checkpointReadDir = (char *) calloc(PV_PATH_MAX, sizeof(char));
      if(checkpointReadDir==NULL) {
         fprintf(stderr, "%s error: unable to allocate memory for path to checkpoint read directory.\n", argv[0]);
         exit(EXIT_FAILURE);
      }
      if (columnId()==0) {
         struct stat statbuf;
         // Look for directory "Last" in outputPath directory
         std::string cpDirString = outputPath;
         cpDirString += "/";
         cpDirString += "Last";
         if (PV_stat(cpDirString.c_str(), &statbuf)==0) {
            if (statbuf.st_mode & S_IFDIR) {
               strncpy(checkpointReadDir, cpDirString.c_str(), PV_PATH_MAX);
               if (checkpointReadDir[PV_PATH_MAX-1]) {
                  fprintf(stderr, "%s error: checkpoint read directory \"%s\" too long.\n", argv[0], cpDirString.c_str());
                  exit(EXIT_FAILURE);
               }
            }
            else {
               fprintf(stderr, "%s error: checkpoint read directory \"%s\" is not a directory.\n", argv[0], cpDirString.c_str());
               exit(EXIT_FAILURE);
            }
         }
         else if (checkpointWriteFlag) {
            // Last directory didn't exist; now look for checkpointWriteDir
            assert(checkpointWriteDir);
            cpDirString = checkpointWriteDir;
            if (cpDirString.c_str()[cpDirString.length()-1] != '/') {
               cpDirString += "/";
            }
            int statstatus = PV_stat(cpDirString.c_str(), &statbuf);
            if (statstatus==0) {
               if (statbuf.st_mode & S_IFDIR) {
                  char *dirs[] = {checkpointWriteDir, NULL};
                  FTS * fts = fts_open(dirs, FTS_LOGICAL, NULL);
                  FTSENT * ftsent = fts_read(fts);
                  bool found = false;
                  long int cp_index = LONG_MIN;
                  for (ftsent = fts_children(fts, 0); ftsent!=NULL; ftsent=ftsent->fts_link) {
                     if (ftsent->fts_statp->st_mode & S_IFDIR) {
                        long int x;
                        int k = sscanf(ftsent->fts_name, "Checkpoint%ld", &x);
                        if (x>cp_index) {
                           cp_index = x;
                           found = true;
                        }
                     }
                  }
                  if (!found) {
                     fprintf(stderr, "%s error: restarting but Last directory does not exist and checkpointWriteDir directory \"%s\" does not have any checkpoints\n",
                           argv[0], checkpointWriteDir);
                     exit(EXIT_FAILURE);
                  }
                  int pathlen=snprintf(checkpointReadDir, PV_PATH_MAX, "%sCheckpoint%ld", cpDirString.c_str(), cp_index);
                  if (pathlen>PV_PATH_MAX) {
                     fprintf(stderr, "%s error: checkpoint read directory \"%s\" too long.\n", argv[0], cpDirString.c_str());
                     exit(EXIT_FAILURE);
                  }

               }
               else {
                  fprintf(stderr, "%s error: checkpoint read directory \"%s\" is not a directory.\n", argv[0], checkpointWriteDir);
                  exit(EXIT_FAILURE);
               }
            }
            else if (errno == ENOENT) {
               fprintf(stderr, "%s error: restarting but neither Last nor checkpointWriteDir directory \"%s\" exists.\n", argv[0], checkpointWriteDir);
               exit(EXIT_FAILURE);
            }
         }
         else {
            fprintf(stderr, "%s error: restarting but Last directory does not exist and checkpointWriteDir is not defined (checkpointWrite=false)\n", argv[0]);
         }

      }
#if PV_USE_MPI
      MPI_Bcast(checkpointReadDir, PV_PATH_MAX, MPI_CHAR, 0, icComm->communicator());
#endif // PV_USE_MPI
   }
   if (checkpointReadDir) {
      checkpointReadFlag = true;
      printf("Rank %d process setting checkpointReadDir to %s.\n", columnId(), checkpointReadDir);
   }

   // run only on GPU for now
#ifdef PV_USE_OPENCL
   initializeThreads(opencl_device);
   clDevice->query_device_info();
#endif

   runDelegate = NULL;

   return PV_SUCCESS;
}

int HyPerCol::ioParams(enum ParamsIOFlag ioFlag) {
   ioParamsStartGroup(ioFlag, name);
   ioParamsFillGroup(ioFlag);
   ioParamsFinishGroup(ioFlag);

   return PV_SUCCESS;
}

int HyPerCol::ioParamsStartGroup(enum ParamsIOFlag ioFlag, const char * group_name) {
   if (ioFlag == PARAMS_IO_WRITE && columnId()==0) {
      assert(printParamsStream);
      const char * keyword = params->groupKeywordFromName(group_name);
      fprintf(printParamsStream->fp, "\n");
      fprintf(printParamsStream->fp, "%s \"%s\" = {\n", keyword, group_name);
   }
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_startTime(ioFlag);
   ioParam_dt(ioFlag);
   ioParam_stopTime(ioFlag);
   ioParam_progressInterval(ioFlag);
   ioParam_writeProgressToErr(ioFlag);
   ioParam_outputPath(ioFlag);
   ioParam_printParamsFilename(ioFlag);
   ioParam_randomSeed(ioFlag);
   ioParam_nx(ioFlag);
   ioParam_ny(ioFlag);
   ioParam_filenamesContainLayerNames(ioFlag);
   ioParam_filenamesContainConnectionNames(ioFlag);
   ioParam_checkpointRead(ioFlag);
   ioParam_checkpointWrite(ioFlag);
   ioParam_checkpointWriteDir(ioFlag);
   ioParam_checkpointWriteTriggerMode(ioFlag);
   ioParam_checkpointWriteStepInterval(ioFlag);
   ioParam_checkpointWriteTimeInterval(ioFlag);
   ioParam_deleteOlderCheckpoints(ioFlag);
   ioParam_suppressLastOutput(ioFlag);
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFinishGroup(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_WRITE && columnId()==0) {
      assert(printParamsStream);
      fprintf(printParamsStream->fp, "};\n");
   }
   return PV_SUCCESS;
}

template <typename T>
void HyPerCol::ioParamValueRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * value) {
   switch(ioFlag) {
   case PARAMS_IO_READ:
      *value = params->value(group_name, param_name);
      break;
   case PARAMS_IO_WRITE:
      writeParam(param_name, *value);
      break;
   }
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
// template void HyPerCol::ioParamValueRequired<pvdata_t>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, pvdata_t * value);
template void HyPerCol::ioParamValueRequired<float>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, float * value);
template void HyPerCol::ioParamValueRequired<double>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, double * value);
template void HyPerCol::ioParamValueRequired<int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, int * value);
template void HyPerCol::ioParamValueRequired<unsigned int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, unsigned int * value);
template void HyPerCol::ioParamValueRequired<bool>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, bool * value);

template <typename T>
void HyPerCol::ioParamValue(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * value, T defaultValue, bool warnIfAbsent) {
   switch(ioFlag) {
   case PARAMS_IO_READ:
      *value = (T) params->value(group_name, param_name, defaultValue, warnIfAbsent);
      break;
   case PARAMS_IO_WRITE:
      writeParam(param_name, *value);
      break;
   }
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
// template void HyPerCol::ioParamValue<pvdata_t>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, pvdata_t * value, pvdata_t defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<float>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, float * value, float defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<double>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, double * value, double defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, int * value, int defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<unsigned int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, unsigned int * value, unsigned int defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<bool>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, bool * value, bool defaultValue, bool warnIfAbsent);
template void HyPerCol::ioParamValue<long>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, long * value, long defaultValue, bool warnIfAbsent);

void HyPerCol::ioParamString(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value, const char * defaultValue, bool warnIfAbsent) {
   const char * param_string = NULL;
   switch(ioFlag) {
   case PARAMS_IO_READ:
      param_string = params->stringValue(group_name, param_name, warnIfAbsent);
      if (param_string==NULL && defaultValue !=NULL) {
         if (columnId()==0 && warnIfAbsent==true) {
            fprintf(stderr, "Using default value \"%s\" for string parameter \"%s\" in group \"%s\"\n", defaultValue, param_name, group_name);
         }
         param_string = defaultValue;
      }
      if (param_string!=NULL) {
         *value = strdup(param_string);
         if (*value==NULL) {
            fprintf(stderr, "Rank %d process unable to copy param %s in group \"%s\": %s\n", columnId(), param_name, group_name, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
      else {
         *value = NULL;
      }
      break;
   case PARAMS_IO_WRITE:
      writeParamString(param_name, *value);
   }
}

void HyPerCol::ioParamStringRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value) {
   const char * param_string = NULL;
   switch(ioFlag) {
   case PARAMS_IO_READ:
      param_string = params->stringValue(group_name, param_name, false/*warnIfAbsent*/);
      if (param_string!=NULL) {
         *value = strdup(param_string);
         if (*value==NULL) {
            fprintf(stderr, "Rank %d process unable to copy param %s in group \"%s\": %s\n", columnId(), param_name, group_name, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
      else {
         if (columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: string parameter \"%s\" is required.\n",
                            params->groupKeywordFromName(group_name), group_name, param_name);
         }
#if PV_USE_MPI
         MPI_Barrier(icComm->communicator());
#endif
         exit(EXIT_SUCCESS);
      }
      break;
   case PARAMS_IO_WRITE:
      writeParamString(param_name, *value);
   }

}

template <typename T>
void HyPerCol::ioParamArray(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T ** value, int * arraysize) {
    if(ioFlag==PARAMS_IO_READ) {
       const double * param_array = params->arrayValuesDbl(group_name, param_name, arraysize);
       assert(*arraysize>=0);
       if (*arraysize>0) {
          *value = (T *) calloc((size_t) *arraysize, sizeof(T));
          if (*value==NULL) {
             fprintf(stderr, "%s \"%s\" error: rank %d process unable to copy array parameter %s: %s\n",
                   parameters()->groupKeywordFromName(name), name, columnId(), param_name, strerror(errno));
          }
          for (int k=0; k<*arraysize; k++) {
             (*value)[k] = (T) param_array[k];
          }
       }
    }
    else if (ioFlag==PARAMS_IO_WRITE) {
       writeParamArray(param_name, *value, *arraysize);
    }
    else {
       assert(0); // All possibilities for ioFlag handled above
    }
}
template void HyPerCol::ioParamArray<float>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, float ** value, int * arraysize);

void HyPerCol::ioParam_startTime(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "startTime", &startTime, startTime);
}

void HyPerCol::ioParam_dt(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "dt", &deltaTime, deltaTime);
   deltaTimeBase = deltaTime;  // use param value as base
}

void HyPerCol::ioParam_stopTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !params->present(name, "stopTime") && params->present(name, "numSteps")) {
      assert(!params->presentAndNotBeenRead(name, "startTime"));
      assert(!params->presentAndNotBeenRead(name, "deltaTime"));
      long int numSteps = params->value(name, "numSteps");
      stopTime = startTime + numSteps * deltaTimeBase;
      if (columnId()==0) {
         fprintf(stderr, "Warning: numSteps is deprecated.  Use startTime, stopTime and deltaTime instead.\n");
         fprintf(stderr, "    stopTime set to %f\n", stopTime);
      }
      return;
   }
   // numSteps was deprecated Dec 12, 2013
   // When support for numSteps is removed entirely, remove the above if-statement and keep the ioParamValue call below.
   ioParamValue(ioFlag, name, "stopTime", &stopTime, stopTime);
}

void HyPerCol::ioParam_progressInterval(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !params->present(name, "progressInterval") && params->present(name, "progressStep")) {
      long int progressStep = (long int) params->value(name, "progressStep");
      progressInterval = progressStep/deltaTimeBase;
      if (columnId()==0) {
         fprintf(stderr, "Warning: progressStep is deprecated.  Use progressInterval instead.\n");
         fprintf(stderr, "    progressInterval set to %f\n", progressInterval);
      }
      return;
   }
   // progressStep was deprecated Dec 18, 2013
   // When support for progressStep is removed entirely, remove the above if-statement and keep the ioParamValue call below.
   ioParamValue(ioFlag, name, "progressInterval", &progressInterval, progressInterval);
}

void HyPerCol::ioParam_writeProgressToErr(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "writeProgressToErr", &writeProgressToErr, writeProgressToErr);
}

void HyPerCol::ioParam_outputPath(enum ParamsIOFlag ioFlag) {
   // outputPath can be set on the command line.
   switch(ioFlag) {
   case PARAMS_IO_READ:
      if (outputPath==NULL) {
         if( params->stringPresent(name, "outputPath") ) {
            outputPath = strdup(params->stringValue(name, "outputPath"));
            assert(outputPath != NULL);
         }
         else {
            outputPath = strdup(OUTPUT_PATH);
            assert(outputPath != NULL);
            printf("Output path specified neither in command line nor in params file.\n"
                   "Output path set to default \"%s\"\n", OUTPUT_PATH);
         }
      }
      break;
   case PARAMS_IO_WRITE:
      writeParamString("outputPath", outputPath);
      break;
   default:
      assert(0);
      break;
   }
}

void HyPerCol::ioParam_printParamsFilename(enum ParamsIOFlag ioFlag) {
   ioParamString(ioFlag, name, "printParamsFilename", &printParamsFilename, "pv.params");
}

void HyPerCol::ioParam_randomSeed(enum ParamsIOFlag ioFlag) {
   switch(ioFlag) {
   // randomSeed can be set on the command line, from the params file, or from the system clock
   case PARAMS_IO_READ:
      // set random seed if it wasn't set in the command line
      // bool seedfromclock = false;
      if( !random_seed ) {
         if( params->present(name, "randomSeed") ) {
            random_seed = (unsigned long) params->value(name, "randomSeed");
         }
         else {
            random_seed = getRandomSeed();
         }
      }
      if (random_seed < 10000000) {
         fprintf(stderr, "Error: random seed %u is too small. Use a seed of at least 10000000.\n", random_seed);
         abort();
      }

      random_seed_obj = random_seed;
      break;
   case PARAMS_IO_WRITE:
      writeParam("randomSeed", random_seed);
      break;
   default:
      assert(0);
      break;
   }
}

void HyPerCol::ioParam_nx(enum ParamsIOFlag ioFlag) {
   ioParamValueRequired(ioFlag, name, "nx", &nxGlobal);
}

void HyPerCol::ioParam_ny(enum ParamsIOFlag ioFlag) {
   ioParamValueRequired(ioFlag, name, "ny", &nyGlobal);
}

void HyPerCol::ioParam_filenamesContainLayerNames(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "filenamesContainLayerNames", &filenamesContainLayerNames, 0);
   if(filenamesContainLayerNames < 0 || filenamesContainLayerNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainLayerNames must have the value 0, 1, or 2.\n", name);
      abort();
   }
}

void HyPerCol::ioParam_filenamesContainConnectionNames(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "filenamesContainConnectionNames", &filenamesContainConnectionNames, 0);
   if(filenamesContainConnectionNames < 0 || filenamesContainConnectionNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainConnectionNames must have the value 0, 1, or 2.\n", name);
      abort();
   }
}

void HyPerCol::ioParam_checkpointRead(enum ParamsIOFlag ioFlag) {
   // checkpointRead, checkpointReadDir, and checkpointReadDirIndex parameters were deprecated on Mar 27, 2014.
   // Instead of setting checkpointRead=true; checkpointReadDir="foo"; checkpointReadDirIndex=100,
   // pass the option <-c foo/Checkpoint100> on the command line.
   // If "-c" was passed then checkpointReadDir will have been set by HyPerCol::initialize's call to parse_options.
   // If "-r" was passed then restartFromCheckpoint will  have been set.
   if (!checkpointReadDir && !warmStart) {
      ioParamValue(ioFlag, name, "checkpointRead", &checkpointReadFlag, false/*default value*/, false/*warnIfAbsent*/);
      if (checkpointReadFlag) {
         ioParamStringRequired(ioFlag, name, "checkpointReadDir", &checkpointReadDirBase);
         ioParamValueRequired(ioFlag, name, "checkpointReadDirIndex", &cpReadDirIndex);
         if (ioFlag==PARAMS_IO_READ) {
            int str_len = snprintf(NULL, 0, "%s/Checkpoint%ld", checkpointReadDirBase, cpReadDirIndex);
            size_t str_size = (size_t) (str_len+1);
            checkpointReadDir = (char *) malloc( str_size*sizeof(char) );
            snprintf(checkpointReadDir, str_size, "%s/Checkpoint%ld", checkpointReadDirBase, cpReadDirIndex);
         }
      }
      else {
         checkpointReadDirBase = NULL;
      }
      if (ioFlag==PARAMS_IO_READ && columnId()==0 && params->present(name, "checkpointRead")) {
         fprintf(stderr, "%s \"%s\" warning: checkpointRead parameter is deprecated.\n",
               params->groupKeywordFromName(name), name);
         if (params->value(name, "checkpointRead")!=0) {
            fprintf(stderr, "    Instead, pass the option on the command line:  -c \"%s\".\n", checkpointReadDir);
         }
      }
   }
}

void HyPerCol::ioParam_checkpointWrite(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "checkpointWrite", &checkpointWriteFlag, false/*default value*/);
}

void HyPerCol::ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      ioParamStringRequired(ioFlag, name, "checkpointWriteDir", &checkpointWriteDir);
   }
   else {
      checkpointWriteDir = NULL;
   }
}

void HyPerCol::ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag ) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      ioParamString(ioFlag, name, "checkpointWriteTriggerMode", &checkpointWriteTriggerModeString, "step");
      if (ioFlag==PARAMS_IO_READ) {
         assert(checkpointWriteTriggerModeString);
         if (!strcmp(checkpointWriteTriggerModeString, "step") || !strcmp(checkpointWriteTriggerModeString, "Step") || !strcmp(checkpointWriteTriggerModeString, "STEP")) {
            checkpointWriteTriggerMode = CPWRITE_TRIGGER_STEP;
         }
         else if (!strcmp(checkpointWriteTriggerModeString, "time") || !strcmp(checkpointWriteTriggerModeString, "Time") || !strcmp(checkpointWriteTriggerModeString, "TIME")) {
            checkpointWriteTriggerMode = CPWRITE_TRIGGER_TIME;
         }
         else if (!strcmp(checkpointWriteTriggerModeString, "clock") || !strcmp(checkpointWriteTriggerModeString, "Clock") || !strcmp(checkpointWriteTriggerModeString, "CLOCK")) {
            checkpointWriteTriggerMode = CPWRITE_TRIGGER_CLOCK;
            if (columnId()==0) {
               fprintf(stderr, "HyPerCol \"%s\": checkpointWriteTriggerMode \"clock\" has not been implemented yet.\n", name);
            }
#if PV_USE_MPI
            MPI_Barrier(icCommunicator()->communicator());
#endif
            exit(EXIT_FAILURE);
         }
         else {
            if (columnId()==0) {
               fprintf(stderr, "HyPerCol \"%s\": checkpointWriteTriggerMode \"%s\" is not recognized.\n", name, checkpointWriteTriggerModeString);
            }
#if PV_USE_MPI
            MPI_Barrier(icCommunicator()->communicator());
#endif
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HyPerCol::ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
   if(checkpointWriteFlag && checkpointWriteTriggerMode == CPWRITE_TRIGGER_STEP) {
      ioParamValue(ioFlag, name, "checkpointWriteStepInterval", &cpWriteStepInterval, 1L);
   }
}

void HyPerCol::ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
   if(checkpointWriteFlag && checkpointWriteTriggerMode == CPWRITE_TRIGGER_TIME) {
      ioParamValue(ioFlag, name, "checkpointWriteTimeInterval", &cpWriteTimeInterval, deltaTimeBase);
   }
}

void HyPerCol::ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      ioParamValue(ioFlag, name, "deleteOlderCheckpoints", &deleteOlderCheckpoints, false/*default value*/);
   }
}

void HyPerCol::ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (!checkpointWriteFlag) {
      ioParamValue(ioFlag, name, "suppressLastOutput", &suppressLastOutput, false/*default value*/);
   }
}

template <typename T>
void HyPerCol::writeParam(const char * param_name, T value) {
   if (columnId()==0) {
      assert(printParamsStream && printParamsStream->fp);
      std::stringstream vstr("");
      if (typeid(value)==typeid(false)) {
         vstr << (value ? "true" : "false");
      }
      else {
         vstr << value;
      }
      fprintf(printParamsStream->fp, "    %-35s = %s;\n", param_name, vstr.str().c_str()); // Check: does vstr.str().c_str() work?
   }
}
// Declare the instantiations of writeParam that occur in other .cpp files; otherwise you'll get linker errors.
template void HyPerCol::writeParam<float>(const char * param_name, float value);
template void HyPerCol::writeParam<int>(const char * param_name, int value);
template void HyPerCol::writeParam<unsigned int>(const char * param_name, unsigned int value);
template void HyPerCol::writeParam<bool>(const char * param_name, bool value);

void HyPerCol::writeParamString(const char * param_name, const char * svalue) {
   if (columnId()==0) {
      assert(printParamsStream!=NULL && printParamsStream->fp!=NULL);
      if (svalue!=NULL) {
         fprintf(printParamsStream->fp, "    %-35s = \"%s\";\n", param_name, svalue);
      }
      else {
         fprintf(printParamsStream->fp, "    // %-35s was set to (NULL);\n", param_name);
      }
   }
}

template <typename T>
void HyPerCol::writeParamArray(const char * param_name, const T * array, int arraysize) {
   if (columnId()==0) {
      assert(printParamsStream!=NULL && printParamsStream->fp!=NULL && arraysize>=0);
      assert(arraysize>=0);
      if (arraysize>0) {
         fprintf(printParamsStream->fp, "    %-35s = [", param_name);
         for (int k=0; k<arraysize-1; k++) {
            fprintf(printParamsStream->fp, "%f,", array[k]);
         }
         fprintf(printParamsStream->fp, "%f];\n", array[arraysize-1]);
      }
   }
}
// Declare the instantiations of writeParam that occur in other .cpp files; otherwise you'll get linker errors.
template void HyPerCol::writeParamArray<float>(const char * param_name, const float * array, int arraysize);


int HyPerCol::checkDirExists(const char * dirname, struct stat * pathstat) {
   // check if the given directory name exists for the rank zero process
   // the return value is zero if a successful stat(2) call and the error
   // if unsuccessful.  pathstat contains the result of the buffer from the stat call.
   // The rank zero process is the only one that calls stat(); it then Bcasts the
   // result to the rest of the processes.
   assert(pathstat);

   int rank = icComm->commRank();
   int status;
   int errorcode;
   if( rank == 0 ) {
      status = stat(dirname, pathstat);
      if( status ) errorcode = errno;
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, 0, icCommunicator()->communicator());
   if( status ) {
      MPI_Bcast(&errorcode, 1, MPI_INT, 0, icCommunicator()->communicator());
   }
   MPI_Bcast(pathstat, sizeof(struct stat), MPI_CHAR, 0, icCommunicator()->communicator());
#endif // PV_USE_MPI
   return status ? errorcode : 0;
}

int HyPerCol::ensureDirExists(const char * dirname) {
   // see if path exists, and try to create it if it doesn't.
   // Since only rank 0 process should be reading and writing, only rank 0 does the mkdir call
   int rank = icComm->commRank();
   struct stat pathstat;
   int resultcode = checkDirExists(dirname, &pathstat);
   if( resultcode == 0 ) { // outputPath exists; now check if it's a directory.
      if( !(pathstat.st_mode & S_IFDIR ) ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr, "Path \"%s\" exists but is not a directory\n", dirname);
         }
         exit(EXIT_FAILURE);
      }
   }
   else if( resultcode == ENOENT /* No such file or directory */ ) {
      if( rank == 0 ) {
         printf("Directory \"%s\" does not exist; attempting to create\n", dirname);

         char targetString[PV_PATH_MAX];
         int num_chars_needed = snprintf(targetString,PV_PATH_MAX,"mkdir -p %s",dirname);
         if (num_chars_needed > PV_PATH_MAX) {
            fflush(stdout);
            fprintf(stderr,"Path \"%s\" is too long.",dirname);
            exit(EXIT_FAILURE);
         }
         int mkdirstatus = system(targetString);
         if( mkdirstatus != 0 ) {
            fflush(stdout);
            fprintf(stderr, "Directory \"%s\" could not be created: %s\n", dirname, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      if( rank == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Error checking status of directory \"%s\": %s\n", dirname, strerror(resultcode));
      }
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int HyPerCol::columnId()
{
   return icComm->commRank();
}

int HyPerCol::numberOfColumns()
{
   return icComm->numCommRows() * icComm->numCommColumns();
}

int HyPerCol::commColumn(int colId)
{
   return colId % icComm->numCommColumns();
}

int HyPerCol::commRow(int colId)
{
   return colId / icComm->numCommColumns();
}

int HyPerCol::addLayer(HyPerLayer * l)
{
   assert((size_t) numLayers <= layerArraySize);

   // Check for duplicate layer names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numLayers; k++) {
   //    if( !strcmp(l->getName(), layers[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, l->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }

   if( (size_t) numLayers ==  layerArraySize ) {
      layerArraySize += RESIZE_ARRAY_INCR;
      HyPerLayer ** newLayers = (HyPerLayer **) malloc( layerArraySize * sizeof(HyPerLayer *) );
      assert(newLayers);
      for(int k=0; k<numLayers; k++) {
         newLayers[k] = layers[k];
      }
      free(layers);
      layers = newLayers;
   }
   layers[numLayers++] = l;
   if (l->getPhase() >= numPhases) numPhases = l->getPhase()+1;
   return (numLayers - 1);
}

int HyPerCol::addConnection(HyPerConn * conn)
{
   int connId = numConnections;

   assert((size_t) numConnections <= connectionArraySize);
   // Check for duplicate connection names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numConnections; k++) {
   //    if( !strcmp(conn->getName(), connections[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, conn->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }
   if( (size_t) numConnections == connectionArraySize ) {
      connectionArraySize += RESIZE_ARRAY_INCR;
      HyPerConn ** newConnections = (HyPerConn **) malloc( connectionArraySize * sizeof(HyPerConn *) );
      assert(newConnections);
      for(int k=0; k<numConnections; k++) {
         newConnections[k] = connections[k];
      }
      free(connections);
      connections = newConnections;
   }

   // numConnections is the ID of this connection
   // subscribe call moved to HyPerCol::initPublishers, since it needs to be after the publishers are initialized.
   // icComm->subscribe(conn);

   connections[numConnections++] = conn;

   return connId;
}

  // typically called by buildandrun via HyPerCol::run()
int HyPerCol::run(double start_time, double stop_time, double dt)
{
   startTime = start_time;
   stopTime = stop_time;
   deltaTime = dt;

   int (HyPerCol::*layerInitializationStage)(int) = NULL;
   int (HyPerCol::*connInitializationStage)(int) = NULL;

   // communicateInitInfo stage
   layerInitializationStage = &HyPerCol::layerCommunicateInitInfo;
   connInitializationStage = &HyPerCol::connCommunicateInitInfo;
   doInitializationStage(layerInitializationStage, connInitializationStage, "communicateInitInfo");

   // insert probes
   for (int i=0; i<numLayerProbes; i++) {
      LayerProbe * p = layerProbes[i];
      int pstatus = p->communicateInitInfo();
      if (pstatus==PV_SUCCESS) {
         if (columnId()==0) printf("Layer probe \"%s\" communicateInitInfo completed.\n", p->getProbeName());
      }
      else {
         assert(pstatus == PV_FAILURE); // PV_POSTPONE etc. hasn't been implemented for probes yet.
         exit(EXIT_FAILURE); // Any error message should be printed by probe's communicateInitInfo function
      }
   }
   //for (int i=0; i<numConnProbes; i++) {
   //   BaseConnectionProbe * p = connProbes[i];
   //   p->communicateInitInfo();
   //}

   // allocateDataStructures stage
   layerInitializationStage = &HyPerCol::layerAllocateDataStructures;
   connInitializationStage = &HyPerCol::connAllocateDataStructures;
   doInitializationStage(layerInitializationStage, connInitializationStage, "allocateDataStructures");

#ifdef OBSOLETE // Marked obsolete Aug 9, 2013.  Look everybody, checkMarginWidths is obsolete!
   if( checkMarginWidths() != PV_SUCCESS ) {
      fprintf(stderr, "Margin width failure; unable to continue.\n");
      return PV_MARGINWIDTH_FAILURE;
   }
#endif // OBSOLETE

   const bool exitOnFinish = false;

   initPublishers(); // create the publishers and their data stores

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // Initialize either by loading from checkpoint, or calling initializeState
   // This needs to happen after initPublishers so that we can initialize the values in the data stores,
   // and before the layers' publish calls so that the data in border regions gets copied correctly.
   if ( checkpointReadFlag ) {
      checkpointRead(checkpointReadDir);
   }
   else {
      for ( int l=0; l<numLayers; l++ ) {
         layers[l]->initializeState();
      }
   }

   parameters()->warnUnread();
   if (printParamsFilename!=NULL) outputParams();

   // publish initial conditions
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->publish(icComm, simTime);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      icComm->wait(layers[l]->getLayerId());
   }

   // output initial conditions
   if (!checkpointReadFlag) {
      for (int c = 0; c < numConnections; c++) {
         connections[c]->outputState(simTime);
      }
      for (int l = 0; l < numLayers; l++) {
         layers[l]->outputState(simTime);
      }
   }

   if (runDelegate) {
      // let delegate advance the time
      //
      runDelegate->run(simTime, stopTime);
   }

#ifdef TIMER_ON
   start_clock();
#endif
   // time loop
   //
   long int step = 0;
   int status = PV_SUCCESS;
   while (simTime < stopTime - deltaTime/2.0 && status != PV_EXIT_NORMALLY) {
      if( checkpointWriteFlag && advanceCPWriteTime() ) {
         if ( !checkpointReadFlag || strcmp(checkpointReadDir, checkpointWriteDir) || cpReadDirIndex!=currentStep ) {
            if (icComm->commRank()==0) {
               printf("Checkpointing, simTime = %f\n", simulationTime());
            }

            char cpDir[PV_PATH_MAX];
            int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", checkpointWriteDir, currentStep);
            if(chars_printed >= PV_PATH_MAX) {
               if (icComm->commRank()==0) {
                  fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", checkpointWriteDir, currentStep);
                  abort();
               }
            }
            checkpointWrite(cpDir);
         }
         else {
            if (icComm->commRank()==0) {
               printf("Skipping checkpoint at time %f, since this would clobber the checkpointRead checkpoint.\n", simulationTime());
            }
         }
      }
      status = advanceTime(simTime);

      step += 1;
#ifdef TIMER_ON
      if (step == 10) start_clock();
#endif

   }  // end time loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run done...\n");  fflush(stdout);
   }
#endif

   exitRunLoop(exitOnFinish);

#ifdef TIMER_ON
   stop_clock();
#endif

   return PV_SUCCESS;
}

int HyPerCol::doInitializationStage(int (HyPerCol::*layerInitializationStage)(int), int (HyPerCol::*connInitializationStage)(int), const char * stageName) {
   int status = PV_SUCCESS;
   int * layerStatus = (int *) malloc((size_t) numLayers * sizeof(int));
   assert(layerStatus);
   for (int l=0; l<numLayers; l++) {
      layerStatus[l]=PV_POSTPONE;
   }
   int * connStatus = (int *) malloc((size_t) numConnections * sizeof(int));
   for (int c=0; c<numConnections; c++) {
      connStatus[c]=PV_POSTPONE;
   }
   int numPostponedLayers = numLayers;
   int numPostponedConns = numConnections;
   int prevNumPostponedLayers;
   int prevNumPostponedConns;
   do {
      prevNumPostponedLayers = numPostponedLayers;
      prevNumPostponedConns = numPostponedConns;
      for (int l=0; l<numLayers; l++) {
         if (layerStatus[l]==PV_POSTPONE) {
            int status = (this->*layerInitializationStage)(l);
            switch (status) {
            case PV_SUCCESS:
               layerStatus[l] = PV_SUCCESS;
               numPostponedLayers--;
               assert(numPostponedLayers>=0);
               if (columnId()==0) printf("Layer \"%s\" %s completed.\n", layers[l]->getName(), stageName);
               break;
            case PV_POSTPONE:
               if (columnId()==0) printf("Layer \"%s\": %s postponed.\n", layers[l]->getName(), stageName);
               break;
            case PV_FAILURE:
               exit(EXIT_FAILURE); // Any error message should be printed by layerInitializationStage function.
               break;
            default:
               assert(0); // This shouldn't be possible
            }
         }
      }
      for (int c=0; c<numConnections; c++) {
         if (connStatus[c]==PV_POSTPONE) {
            int status = (this->*connInitializationStage)(c);
            switch (status) {
            case PV_SUCCESS:
               connStatus[c] = PV_SUCCESS;
               numPostponedConns--;
               assert(numPostponedConns>=0);
               if (columnId()==0) printf("Connection \"%s\" %s completed.\n", connections[c]->getName(), stageName);
               break;
            case PV_POSTPONE:
               if (columnId()==0) printf("Connection \"%s\" %s postponed.\n", connections[c]->getName(), stageName);
               break;
            case PV_FAILURE:
               exit(EXIT_FAILURE); // Error message printed in HyPerConn::communicateInitInfo().
               break;
            default:
               assert(0); // This shouldn't be possible
            }
         }
      }
   }
   while (numPostponedLayers < prevNumPostponedLayers || numPostponedConns < prevNumPostponedConns);

   if (numPostponedLayers != 0 || numPostponedConns != 0) {
      printf("%s loop has hung on rank %d process.\n", stageName, columnId());
      for (int l=0; l<numLayers; l++) {
         if (layerStatus[l]==PV_POSTPONE) {
            printf("Layer \"%s\" on rank %d is still postponed.\n", layers[l]->getName(), columnId());
         }
      }
      for (int c=0; c<numConnections; c++) {
         if (layerStatus[c]==PV_POSTPONE) {
            printf("Connection \"%s\" on rank %d is still postponed.\n", connections[c]->getName(), columnId());
         }
      }
      exit(EXIT_FAILURE);
   }
   free(layerStatus); layerStatus = NULL;
   free(connStatus); connStatus = NULL;
   return status;
}

int HyPerCol::layerCommunicateInitInfo(int l) {
   HyPerLayer * layer = layers[l];
   assert(l>=0 && l<numLayers && layer->getInitInfoCommunicatedFlag()==false);
   int status = layer->communicateInitInfo();
   if (status==PV_SUCCESS) layer->setInitInfoCommunicatedFlag();
   return status;
}

int HyPerCol::connCommunicateInitInfo(int c) {
   HyPerConn * conn = connections[c];
   assert(c>=0 && c<numConnections && conn->getInitInfoCommunicatedFlag()==false);
   int status = conn->communicateInitInfo();
   if (status==PV_SUCCESS) conn->setInitInfoCommunicatedFlag();
   return status;
}

int HyPerCol::layerAllocateDataStructures(int l) {
   HyPerLayer * layer = layers[l];
   assert(l>=0 && l<numLayers && layer->getDataStructuresAllocatedFlag()==false);
   int status = layer->allocateDataStructures();
   if (status==PV_SUCCESS) layer->setDataStructuresAllocatedFlag();
   return status;
}

int HyPerCol::connAllocateDataStructures(int c) {
   HyPerConn * conn = connections[c];
   assert(c>=0 && c<numConnections && conn->getDataStructuresAllocatedFlag()==false);
   int status = conn->allocateDataStructures();
   if (status==PV_SUCCESS) conn->setDataStructuresAllocatedFlag();
   return status;
}

int HyPerCol::initPublishers() {
   for( int l=0; l<numLayers; l++ ) {
      // PVLayer * clayer = layers[l]->getCLayer();
      icComm->addPublisher(layers[l], layers[l]->getNumExtended(), layers[l]->getNumDelayLevels());
   }
   for( int c=0; c<numConnections; c++ ) {
      icComm->subscribe(connections[c]);
   }

   return PV_SUCCESS;
}

int HyPerCol::advanceTime(double sim_time)
{
   if (simTime >= nextProgressTime) {
      nextProgressTime += progressInterval;
      if (columnId() == 0) {
         FILE * progressStream = writeProgressToErr ? stderr : stdout;
         fprintf(progressStream, "   time==%f  ", sim_time);
         time_t current_time;
         time(&current_time);
         std::cout << ctime(&current_time);
      }
   }

   runTimer->start();

   // adapt deltaTime
   // query all layers to determine minimum timeScale > 0
   // by default, HyPerLayer::getTimeScale returns -1
   // initialize timeScaleMin to first returned timeScale > 0
   // Movie returns timeScale = 1 when expecting to load a new frame 
   // on next time step based on current value of deltaTime
   deltaTime = deltaTimeBase;
   double timeScale = 1.0;
   double timeScaleMin = -1.0;
   for(int l = 0; l < numLayers; l++) {
     double timeScaleTmp = layers[l]->getTimeScale();
     if (timeScaleTmp > 0.0){
       if (timeScaleMin > 0.0){
	 timeScaleMin = timeScaleTmp < timeScaleMin ? timeScaleTmp : timeScaleMin;
       }
       else{
	 timeScaleMin = timeScaleTmp;
       }
     }
   }
   timeScale = timeScaleMin > 0.0 ? timeScaleMin : 1.0;
   // deltaTimeAdapt is only used internally to set scale of each update step
   double deltaTimeAdapt = timeScale * deltaTimeBase;
   // if (columnId() == 0) {
   //   std::cout << "timeScale = " << timeScale << std::endl;
   // }

   // make sure simTime is updated even if HyPerCol isn't running time loop
   // triggerOffset might fail if simTime does not advance uniformly because
   // simTime could skip over tigger event
   // !!!TODO: fix trigger layer to compute timeScale so as not to allow bypassing trigger event
   simTime = sim_time + deltaTimeBase;
   currentStep++;

   // At this point all activity from the previous time step has
   // been delivered to the data store.
   //

   int status = PV_SUCCESS;
   bool exitAfterUpdate = false;

   // update the connections (weights)
   //
   for (int c = 0; c < numConnections; c++) {
      status = connections[c]->updateStateWrapper(simTime, deltaTimeBase);
      if (!exitAfterUpdate) {
		  exitAfterUpdate = status == PV_EXIT_NORMALLY;
      }
   }
   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(simTime);
   }

   // Each layer's phase establishes a priority for updating
   for (int phase=0; phase<numPhases; phase++) {

      // clear GSyn buffers
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         layers[l]->resetGSynBuffers(simTime, deltaTimeBase);  // deltaTimeAdapt is not used 
         layers[l]->recvAllSynapticInput();
      }
      //    for (int l = 0; l < numLayers; l++) {
      //       // deliver new synaptic activity to any
      //       // postsynaptic layers for which this
      //       // layer is presynaptic.
      //       layers[l]->triggerReceive(icComm);
      //    }

      // Update the layers (activity)
      // We don't put updateState in the same loop over layers as recvAllSynapticInput
      // because we plan to have updateState update the datastore directly, and
      // recvSynapticInput uses the datastore to compute GSyn.
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         status = layers[l]->updateStateWrapper(simTime, deltaTimeAdapt);
		 if (!exitAfterUpdate) {
			 exitAfterUpdate = status == PV_EXIT_NORMALLY;
		 }
      }

      // This loop separate from the update layer loop above
      // to provide time for layer data to be copied from
      // the OpenCL device.
      //
      for (int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         // after updateBorder completes all necessary data has been
         // copied from the device (GPU) to the host (CPU)
         layers[l]->updateBorder(simTime, deltaTimeBase); // TODO rename updateBorder?  deltaTimeAdapt not used here

         // TODO - move this to layer
         // Advance time level so we have a new place in data store
         // to copy the data.  This should be done immediately before
         // publish so there is a place to publish and deliver the data to.
         // No one can access the data store (except to publish) until
         // wait has been called.  This should be fixed so that publish goes
         // to last time level and level is advanced only after wait.
         icComm->increaseTimeLevel(layers[l]->getLayerId());

         layers[l]->publish(icComm, simTime);
         //    }
         //
         //    // wait for all published data to arrive
         //    //
         //    for (int l = 0; l < numLayers; l++) {
         layers[l]->waitOnPublish(icComm);
         //    }
         //
         //    // also calls layer probes
         //    for (int l = 0; l < numLayers; l++) {
         layers[l]->outputState(simTime);
      }

   }

   // double outputTime = simTime; // so that outputState is called with the correct time
   //                             // but doesn't effect runTimer

   runTimer->stop();

   outputState(simTime);

   if (exitAfterUpdate) {
	   status = PV_EXIT_NORMALLY;
   }

   return status;
}

bool HyPerCol::advanceCPWriteTime() {
   // returns true if nextCPWrite{Step,Time} has been advanced
   bool advanceCPTime;
   if( cpWriteStepInterval>0 ) {
      assert(cpWriteTimeInterval<0.0);
      advanceCPTime = currentStep >= nextCPWriteStep;
      if( advanceCPTime ) {
         nextCPWriteStep += cpWriteStepInterval;
      }
   }
   else if( cpWriteTimeInterval>0.0) {
      assert(cpWriteStepInterval<0);
      advanceCPTime = simTime >= nextCPWriteTime;
      if( advanceCPTime ) {
         nextCPWriteTime += cpWriteTimeInterval;
      }
   }
   else {
      assert( false ); // routine should only be called if one of cpWrite{Step,Time}Interval is positive
      advanceCPTime = false;
   }
   return advanceCPTime;
}

int HyPerCol::checkpointRead(const char * cpDir) {
   struct timestamp_struct {
      double time; // time measured in units of dt
      long int step; // step number, usually time/dt
   };
   struct timestamp_struct timestamp;
   size_t timestamp_size = sizeof(struct timestamp_struct);
   assert(sizeof(struct timestamp_struct) == sizeof(long int) + sizeof(double));
   if( icCommunicator()->commRank()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      PV_Stream * timestampfile = PV_fopen(timestamppath,"r");
      if (timestampfile == NULL) {
         fprintf(stderr, "HyPerCol::checkpointRead error: unable to open \"%s\" for reading.\n", timestamppath);
         abort();
      }
      long int startpos = getPV_StreamFilepos(timestampfile);
      PV_fread(&timestamp,1,timestamp_size,timestampfile);
      long int endpos = getPV_StreamFilepos(timestampfile);
      assert(endpos-startpos==(int)timestamp_size);
      PV_fclose(timestampfile);
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&timestamp,(int) timestamp_size,MPI_CHAR,0,icCommunicator()->communicator());
#endif // PV_USE_MPI
   simTime = timestamp.time;
   currentStep = timestamp.step;
   double checkTime;
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointRead(cpDir, &checkTime);
      if (checkTime!=simTime && columnId()==0) {
         fprintf(stderr, "Warning: layer \"%s\" checkpoint has timestamp %f instead of the HyPerCol timestamp %f.\n",
               layers[l]->getName(), checkTime, simTime);
      }
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointRead(cpDir, &checkTime);
      if (checkTime!=simTime && columnId()==0) {
         fprintf(stderr, "Warning: connection \"%s\" checkpoint has timestamp %f instead of the HyPerCol timestamp %f.\n",
               connections[c]->getName(), checkTime, simTime);
      }
   }
   if(checkpointWriteFlag) {
      if( cpWriteStepInterval > 0) {
         assert(cpWriteTimeInterval<0.0f);
         nextCPWriteStep = currentStep; // checkpointWrite should be called before any timesteps,
             // analogous to checkpointWrite being called immediately after initialization on a fresh run.
      }
      else if( cpWriteTimeInterval > 0.0f ) {
         assert(cpWriteStepInterval<0);
         nextCPWriteTime = simTime; // checkpointWrite should be called before any timesteps
      }
      else {
         assert(false); // if checkpointWriteFlag is set, one of cpWrite{Step,Time}Interval should be positive
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::checkpointWrite(const char * cpDir) {
   if (columnId()==0) {
      printf("Checkpointing to directory \"%s\" at simTime = %f\n", cpDir, simTime);
      struct stat timeinfostat;
      char timeinfofilename[PV_PATH_MAX];
      int chars_needed = snprintf(timeinfofilename, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointWrite error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      int statstatus = stat(timeinfofilename, &timeinfostat);
      if (statstatus == 0) {
         fprintf(stderr, "Warning: Checkpoint directory \"%s\" has existing timeinfo.bin, which is now being deleted.\n", cpDir);
         int unlinkstatus = unlink(timeinfofilename);
         if (unlinkstatus != 0) {
            fprintf(stderr, "Error deleting \"%s\": %s\n", timeinfofilename, strerror(errno));
            abort();
         }
      }
   }

   ensureDirExists(cpDir);
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointWrite(cpDir);
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointWrite(cpDir);
   }
   
   // Timers
   if (columnId()==0) {
      std::string timerpathstring = cpDir;
      timerpathstring += "/";
      timerpathstring += "timers.txt";
      const char * timerpath = timerpathstring.c_str();
      PV_Stream * timerstream = PV_fopen(timerpath, "w");
      assert(timerstream); // Lazy; do a proper error message.
      runTimer->fprint_time(timerstream->fp);
      icCommunicator()->fprintTime(timerstream->fp);
      for (int l=0; l<numLayers; l++) {
         layers[l]->checkpointTimers(timerstream);
      }
      for (int c=0; c<numConnections; c++) {
         connections[c]->checkpointTimers(timerstream);
      }
      PV_fclose(timerstream); timerstream = NULL;
   }

   // Note: timeinfo should be done at the end of the checkpointing, so that its presence serves as a flag that the checkpoint has completed.
   if( columnId()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timestampfile = PV_fopen(timestamppath,"w");
      assert(timestampfile);
      PV_fwrite(&simTime,1,sizeof(double),timestampfile);
      PV_fwrite(&currentStep,1,sizeof(long int),timestampfile);
      PV_fclose(timestampfile);
      chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timestampfile = PV_fopen(timestamppath,"w");
      assert(timestampfile);
      fprintf(timestampfile->fp,"time = %g\n", simTime);
      fprintf(timestampfile->fp,"timestep = %ld\n", currentStep);
      PV_fclose(timestampfile);
   }

   if (deleteOlderCheckpoints) {
      assert(checkpointWriteFlag); // checkpointWrite is called by exitRunLoop when checkpointWriteFlag is false; in this case deleteOlderCheckpoints should be false as well.
      if (lastCheckpointDir[0]) {
         if (icComm->commRank()==0) {
            struct stat lcp_stat;
            int statstatus = stat(lastCheckpointDir, &lcp_stat);
            if ( statstatus!=0 || !(lcp_stat.st_mode & S_IFDIR) ) {
               if (statstatus==0) {
                  fprintf(stderr, "Error deleting older checkpoint: failed to stat \"%s\": %s.\n", lastCheckpointDir, strerror(errno));
               }
               else {
                  fprintf(stderr, "Deleting older checkpoint: \"%s\" exists but is not a directory.\n", lastCheckpointDir);
               }
            }
#define RMRFSIZE (PV_PATH_MAX + 13)
            char rmrf_string[RMRFSIZE];
            int chars_needed = snprintf(rmrf_string, RMRFSIZE, "rm -r '%s'", lastCheckpointDir);
            assert(chars_needed < RMRFSIZE);
#undef RMRFSIZE
            system(rmrf_string);
         }
      }
      int chars_needed = snprintf(lastCheckpointDir, PV_PATH_MAX, "%s", cpDir);
      assert(chars_needed < PV_PATH_MAX);
   }

   if (icComm->commRank()==0) {
      fprintf(stderr, "checkpointWrite complete. simTime = %f\n", simTime);
   }
   return PV_SUCCESS;
}

int HyPerCol::outputParams() {
   int status = PV_SUCCESS;
#ifdef PV_USE_MPI
   int rank=icComm->commRank();
#else
   int rank=0;
#endif
   assert(printParamsStream==NULL);
   char printParamsPath[PV_PATH_MAX];
   if( rank == 0 && printParamsFilename != NULL && printParamsFilename[0] != '\0' ) {
      int len = 0;
      if (printParamsFilename[0] == '/') { // filename is absolute path
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s", printParamsFilename);
      }
      else { // filename is relative path from outputPath
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s/%s", outputPath, printParamsFilename);
      }
      if( len >= PV_PATH_MAX ) {
         fprintf(stderr, "outputParams: ");
         if (printParamsFilename[0] != '/') fprintf(stderr, "outputPath + ");
         fprintf(stderr, "printParamsFilename gives too long a filename.  Parameters will not be printed.\n");
      }
      else {
         printParamsStream = PV_fopen(printParamsPath, "w");
         if( printParamsStream == NULL ) {
            status = errno;
            fprintf(stderr, "outputParams error opening \"%s\" for writing: %s\n", printParamsPath, strerror(errno));
         }
      }
   }
   if (printParamsStream != NULL) {
      time_t t = time(NULL);
      fprintf(printParamsStream->fp, "// PetaVision version something-point-something run at %s", ctime(&t)); // newline is included in output of ctime
#ifdef PV_USE_MPI
      fprintf(printParamsStream->fp, "// Compiled with MPI and run using %d rows and %d columns.\n", icComm->numCommRows(), icComm->numCommColumns());
#else // PV_USE_MPI
      fprintf(printParamsStream->fp, "// Compiled without MPI.\n");
#endif // PV_USE_MPI
      if (checkpointReadFlag) {
         fprintf(printParamsStream->fp, "// Started from checkpoint \"%s\"\n", checkpointReadDir);
      }
   }
   // Parent HyPerCol params
   status = ioParams(PARAMS_IO_WRITE);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", printParamsPath);
      exit(EXIT_FAILURE);
   }

   // HyPerLayer params
   for (int l=0; l<numLayers; l++) {
      HyPerLayer * layer = layers[l];
      status = layer->ioParams(PARAMS_IO_WRITE);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", printParamsPath);
         exit(EXIT_FAILURE);
      }
   }

   // HyPerConn params
   for (int c=0; c<numConnections; c++) {
      HyPerConn * connection = connections[c];
      status = connection->ioParams(PARAMS_IO_WRITE);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", printParamsPath);
         exit(EXIT_FAILURE);
      }
   }

   // Probe params

   // ColProbes
   for (int p=0; p<numProbes; p++) {
      probes[p]->ioParams(PARAMS_IO_WRITE);
   }

   // LayerProbes
   for (int l=0; l<numLayers; l++) {
      layers[l]->outputProbeParams();
   }

   // BaseConnectionProbes
   for (int c=0; c<numConnections; c++) {
      connections[c]->outputProbeParams();
   }

   if (printParamsStream) {
      PV_fclose(printParamsStream);
      printParamsStream = NULL;
   }

   return status;
}

int HyPerCol::exitRunLoop(bool exitOnFinish)
{
   int status = 0;

   // output final state of layers and connections
   //

   char cpDir[PV_PATH_MAX];
   if (checkpointWriteFlag || !suppressLastOutput) {
      int chars_printed;
      if (checkpointWriteFlag) {
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", checkpointWriteDir, currentStep);
      }
      else {
         assert(!suppressLastOutput);
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Last", outputPath);
      }
      if(chars_printed >= PV_PATH_MAX) {
         if (icComm->commRank()==0) {
            fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", checkpointWriteDir, currentStep);
            abort();
         }
      }
      checkpointWrite(cpDir);
   }

   if (exitOnFinish) {
      delete this;
      exit(0);
   }

   return status;
}

int HyPerCol::initializeThreads(int device)
{
   clDevice = new CLDevice(device);
   return 0;
}

#ifdef PV_USE_OPENCL
int HyPerCol::finalizeThreads()
{
   delete clDevice;
   return 0;
}
#endif // PV_USE_OPENCL

int HyPerCol::loadState()
{
   return 0;
}

int HyPerCol::insertProbe(ColProbe * p)
{
   ColProbe ** newprobes;
   newprobes = (ColProbe **) malloc( ((size_t) (numProbes + 1)) * sizeof(ColProbe *) );
   assert(newprobes != NULL);

   for (int i = 0; i < numProbes; i++) {
      newprobes[i] = probes[i];
   }
   delete probes;

   probes = newprobes;
   probes[numProbes] = p;

   return ++numProbes;
}

//int HyPerCol::addBaseConnectionProbe(BaseConnectionProbe * p) {
//   BaseConnectionProbe ** newprobes;
//   newprobes = (BaseConnectionProbe **) malloc( ((size_t) (numConnProbes + 1)) * sizeof(BaseConnectionProbe *) );
//   assert(newprobes != NULL);
//
//   for (int i=0; i<numConnProbes; i++) {
//      newprobes[i] = connProbes[i];
//   }
//   delete connProbes;
//   connProbes = newprobes;
//   connProbes[numConnProbes] = p;
//
//   return ++numConnProbes;
//}

int HyPerCol::addLayerProbe(LayerProbe * p) {
   LayerProbe ** newprobes;
   newprobes = (LayerProbe **) malloc( ((size_t) (numLayerProbes + 1)) * sizeof(LayerProbe *) );
   assert(newprobes != NULL);

   for (int i=0; i<numLayerProbes; i++) {
      newprobes[i] = layerProbes[i];
   }
   delete layerProbes;
   layerProbes = newprobes;
   layerProbes[numLayerProbes] = p;

   return ++numLayerProbes;
}

int HyPerCol::outputState(double time)
{
   for( int n = 0; n < numProbes; n++ ) {
       probes[n]->outputState(time, this);
   }
   return PV_SUCCESS;
}


HyPerLayer * HyPerCol::getLayerFromName(const char * layerName) {
   int n = numberOfLayers();
   for( int i=0; i<n; i++ ) {
      HyPerLayer * curLayer = getLayer(i);
      assert(curLayer);
      const char * curLayerName = curLayer->getName();
      assert(curLayerName);
      if( !strcmp( curLayer->getName(), layerName) ) return curLayer;
   }
   return NULL;
}

HyPerConn * HyPerCol::getConnFromName(const char * connName) {
   if( connName == NULL ) return NULL;
   int n = numberOfConnections();
   for( int i=0; i<n; i++ ) {
      HyPerConn * curConn = getConnection(i);
      assert(curConn);
      const char * curConnName = curConn->getName();
      assert(curConnName);
      if( !strcmp( curConn->getName(), connName) ) return curConn;
   }
   return NULL;
}

ColProbe * HyPerCol::getColProbeFromName(const char * probeName) {
   if (probeName == NULL) return NULL;
   ColProbe * p = NULL;
   int n = numberOfProbes();
   for (int i=0; i<n; i++) {
      ColProbe * curColProbe = getColProbe(i);
      const char * curName = curColProbe->getColProbeName();
      assert(curName);
      if (!strcmp(curName, probeName)) {
         p = curColProbe;
      }
      break;
   }
   return p;
}

unsigned int HyPerCol::getRandomSeed() {
   unsigned long t = 0UL;
   int rootproc = 0;
   if (columnId()==rootproc) {
       t = time((time_t *) NULL);
   }
#if PV_USE_MPI
   MPI_Bcast(&t, 1, MPI_UNSIGNED, rootproc, icComm->communicator());
#endif
   return t;
}

#ifdef OBSOLETE // Marked obsolete Aug 9, 2013.  Look, everybody, checkMarginWidths is obsolete!
int HyPerCol::checkMarginWidths() {
   // For each connection, make sure that the pre-synaptic margin width is
   // large enough for the patch size.

   int status = PV_SUCCESS;
   int status1, status2;
   for( int c=0; c < numConnections; c++ ) {
      HyPerConn * conn = connections[c];
      HyPerLayer * pre = conn->preSynapticLayer();
      HyPerLayer * post = conn->postSynapticLayer();

      int xScalePre = pre->getXScale();
      int xScalePost = post->getXScale();
      status1 = zCheckMarginWidth(conn, "x", conn->xPatchSize(), xScalePre, xScalePost, status);

      int yScalePre = pre->getYScale();
      int yScalePost = post->getYScale();
      status2 = zCheckMarginWidth(conn, "y", conn->yPatchSize(), yScalePre, yScalePost, status1);
      status = (status == PV_SUCCESS && status1 == PV_SUCCESS && status2 == PV_SUCCESS) ?
               PV_SUCCESS : PV_MARGINWIDTH_FAILURE;
   }
   for( int l=0; l < numLayers; l++ ) {
      HyPerLayer * layer = layers[l];
      status1 = lCheckMarginWidth(layer, "x", layer->getLayerLoc()->nx, layer->getLayerLoc()->nxGlobal, status);
      status2 = lCheckMarginWidth(layer, "y", layer->getLayerLoc()->ny, layer->getLayerLoc()->nyGlobal, status1);
      status = (status == PV_SUCCESS && status1 == PV_SUCCESS && status2 == PV_SUCCESS) ?
               PV_SUCCESS : PV_MARGINWIDTH_FAILURE;
   }
   return status;
}  // end HyPerCol::checkMarginWidths()

int HyPerCol::zCheckMarginWidth(HyPerConn * conn, const char * dim, int patchSize, int scalePre, int scalePost, int prevStatus) {
   int status;
   int scaleDiff = scalePre - scalePost;
   // if post has higher neuronal density than pre, scaleDiff < 0.
   HyPerLayer * pre = conn->preSynapticLayer();
   int padding = conn->preSynapticLayer()->getLayerLoc()->nb;
   int needed = scaleDiff > 0 ? ( patchSize/( (int) pow(2,scaleDiff) )/2 ) :
                                ( (patchSize/2) * ( (int) pow(2,-scaleDiff) ) );
   if( padding < needed ) {
      if( prevStatus == PV_SUCCESS ) {
         fprintf(stderr, "Margin width error.\n");
      }
      fprintf(stderr, "Connection \"%s\", dimension %s:\n", conn->getName(), dim);
      fprintf(stderr, "    Pre-synaptic margin width %d, patch size %d, presynaptic scale %d, postsynaptic scale %d\n",
              padding, patchSize, scalePre, scalePost);
      fprintf(stderr, "    Layer %s needs margin width of at least %d\n", pre->getName(), needed);
      if( numberOfColumns() > 1 || padding > 0 ) {
         status = PV_MARGINWIDTH_FAILURE;
      }
      else {
         fprintf(stderr, "Continuing, but there may be undesirable edge effects.\n");
         status = PV_SUCCESS;
      }
   }
   else status = PV_SUCCESS;
   return status;
}

int HyPerCol::lCheckMarginWidth(HyPerLayer * layer, const char * dim, int layerSize, int layerGlobalSize, int prevStatus) {
   int status;
   int nb = layer->getLayerLoc()->nb;
   if( layerSize < nb) {
      if( prevStatus == PV_SUCCESS ) {
         fprintf(stderr, "Margin width error.\n");
      }
      fprintf(stderr, "Layer \"%s\", dimension %s:\n", layer->getName(), dim);
      fprintf(stderr, "    Pre-synaptic margin width %d, overall layer size %d, layer size per process %d\n", nb, layerGlobalSize, layerSize);
      fprintf(stderr, "    Use either fewer processes in dimension %s, or a margin size <= %d.\n", dim, layerSize);
      status = PV_MARGINWIDTH_FAILURE;
   }
   else status = PV_SUCCESS;
   return status;
}
#endif // OBSOLETE

template <typename T>
int HyPerCol::writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val) {
   int status = PV_SUCCESS;
   if (columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "writeScalarToFile error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "w");
      if (pvstream==NULL) {
         fprintf(stderr, "writeScalarToFile error: unable to open path %s for writing.\n", filename);
         abort();
      }
      int num_written = PV_fwrite(&val, sizeof(val), 1, pvstream);
      if (num_written != 1) {
         fprintf(stderr, "writeScalarToFile error while writing to %s.\n", filename);
         abort();
      }
      PV_fclose(pvstream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, group_name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      std::ofstream fs;
      fs.open(filename);
      if (!fs) {
         fprintf(stderr, "writeScalarToFile error: unable to open path %s for writing.\n", filename);
         abort();
      }
      fs << val;
      fs << std::endl; // Can write as fs << val << std::endl, but eclipse flags that as an error 'Invalid overload of std::endl'
      fs.close();
   }
   return status;
}
// Declare the instantiations of writeScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeScalarToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int val);
template int HyPerCol::writeScalarToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long val);
template int HyPerCol::writeScalarToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float val);
template int HyPerCol::writeScalarToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double val);

template <typename T>
int HyPerCol::readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value) {
   int status = PV_SUCCESS;
   if( columnId() == 0 ) {
      char filename[PV_PATH_MAX];
      int chars_needed;
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if(chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerLayer::readScalarFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "r");
      *val = default_value;
      if (pvstream==NULL) {
         std::cerr << "readScalarFromFile warning: unable to open path \"" << filename << "\" for reading.  Value used will be " << *val;
         std::cerr << std::endl;
         // fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to open path %s for reading.  value used will be %f\n", filename, default_value);
      }
      else {
         int num_read = PV_fread(val, sizeof(T), 1, pvstream);
         if (num_read != 1) {
            std::cerr << "readScalarFromFile warning: unable to read from \"" << filename << "\".  Value used will be " << *val;
            std::cerr << std::endl;
            // fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to read from %s.  value used will be %f\n", filename, default_value);
         }
         PV_fclose(pvstream);
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(val, sizeof(T), MPI_CHAR, 0, icCommunicator()->communicator());
#endif // PV_USE_MPI

   return status;
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readScalarFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, int default_value);
template int HyPerCol::readScalarFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, long default_value);
template int HyPerCol::readScalarFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, float default_value);
template int HyPerCol::readScalarFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, double default_value);

} // PV namespace
