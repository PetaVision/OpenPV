/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#define TIMER_ON
#define TIMESTEP_OUTPUT

#include "HyPerCol.hpp"
#include <columns/InterColComm.hpp>
#include <normalizers/NormalizeBase.hpp>
#include <io/clock.h>
#include <io/io.h>

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
#include <fstream>
#include <time.h>
#include <csignal>
#include <limits>
#include <libgen.h>
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
#include <map>
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

namespace PV {

HyPerCol * castToHyPerCol(void * p) {
   HyPerCol * hc = static_cast<HyPerCol*>(p);
   hc = dynamic_cast<HyPerCol*>(hc);
   if (hc==NULL) {
      fprintf(stderr, "Pointer %p is not a valid HyPerCol pointer.\n", p);
   }
   return hc;
}

HyPerCol::HyPerCol(const char * name, PV_Init * initObj) {
   initialize_base();
   initialize(name, initObj);
}

HyPerCol::~HyPerCol()
{
   int n;

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   finalizeThreads();
#endif // PV_USE_OPENCL || PV_USE_CUDA

   //Print all timers
   writeTimers(stdout);

   if (image_file != NULL) free(image_file);

   for (n = 0; n < numConnections; n++) {
      delete connections[n];
   }
   free(connections);
   for (n = 0; n < numNormalizers; n++) {
      delete normalizers[n];
   }
   free(normalizers);

   int rank = globalRank(); // Need to save so that we know whether we're the process that does I/O, even after deleting icComm.

   if (phaseRecvTimers) {
      for (int phase=0; phase<numPhases; phase++) {
         if(phaseRecvTimers[phase]){
            delete phaseRecvTimers[phase];
         }
      }
      free(phaseRecvTimers);
   }

   for (n = 0; n < numLayers; n++) {
      if (layers[n] != NULL) {
         delete layers[n];
      }
   }
   free(layers);

   //if (ownsParams) delete params;
   //delete icComm;

   //if (ownsInterColComm) {
   //}
   //else {
   //}
   icComm->clearPublishers();

   delete runTimer;
   delete checkpointTimer;

   free(dtAdaptController);

   // colProbes[i] should not be deleted; it points to an entry in baseProbes and will
   // be deleted when baseProbes is deleted, below.
   free(colProbes);

   for (int k=0; k<numBaseProbes; k++) {
      delete baseProbes[k];
   }
   free(baseProbes);

   free(printParamsFilename);
   // free(outputNamesOfLayersAndConns);
   free(outputPath);
   if (srcPath) {
      free(srcPath);
   }
   free(initializeFromCheckpointDir);
   if (checkpointWriteFlag) {
      free(checkpointWriteDir); checkpointWriteDir = NULL;
      free(checkpointWriteTriggerModeString); checkpointWriteTriggerModeString = NULL;
   }
   if (checkpointReadFlag) {
      free(checkpointReadDir); checkpointReadDir = NULL;
      free(checkpointReadDirBase); checkpointReadDirBase = NULL;
   }
   if (dtAdaptFlag && writeTimescales){
      timeScaleStream.close();
   }

   if(pv_initObj->getArguments()->getLogFile()){
      //Flush all buffers
      fflush(stdout);
      fflush(stderr);
      std::cout.flush();
      std::cerr.flush();

      //Restore stdout
      dup2(origStdOut, fileno(stdout));
      dup2(origStdErr, fileno(stderr));

      //Close origStdOut file descriptors
      close(origStdOut);
      close(origStdErr);
   }
   if(timeScale){
      free(timeScale);
   }
   if(timeScaleMax){
      free(timeScaleMax);
   }
   if(timeScaleMax2){
      free(timeScaleMax2);
   }
   if(timeScaleTrue){
      free(timeScaleTrue);
   }
   if(oldTimeScale){
      free(oldTimeScale);
   }
   if(oldTimeScaleTrue){
      free(oldTimeScaleTrue);
   }
   if(deltaTimeAdapt){
      free(deltaTimeAdapt);
   }
}


#define DEFAULT_NUMSTEPS 1
int HyPerCol::initialize_base() {
   // Initialize all member variables to safe values.  They will be set to their actual values in initialize()
   warmStart = false;
   readyFlag = false;
   paramsProcessedFlag = false;
   currentStep = 0;
   layerArraySize = INITIAL_LAYER_ARRAY_SIZE;
   numLayers = 0;
   numPhases = 0;
   connectionArraySize = INITIAL_CONNECTION_ARRAY_SIZE;
   numConnections = 0;
   normalizerArraySize = INITIAL_CONNECTION_ARRAY_SIZE;
   numNormalizers = 0;
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
   cpWriteClockInterval = -1.0;
   deleteOlderCheckpoints = false;
   memset(lastCheckpointDir, 0, PV_PATH_MAX);
   defaultInitializeFromCheckpointFlag = false;
   suppressLastOutput = false;
   suppressNonplasticCheckpoints = false;
   checkpointIndexWidth = -1; // defaults to automatically determine index width
   simTime = 0.0;
   startTime = 0.0;
   stopTime = 0.0;
   deltaTime = DELTA_T;
   dtAdaptFlag = false;
   writeTimeScaleFieldnames = true;
   useAdaptMethodExp1stOrder = false;
   dtAdaptController = NULL;
   dtAdaptControlProbe = NULL;
   dtAdaptTriggerLayerName = NULL;
   dtAdaptTriggerLayer = NULL;
   dtAdaptTriggerOffset = 0.0;
   deltaTimeBase = DELTA_T;
   timeScale = NULL;
   timeScaleMax = NULL;
   timeScaleMax2 = NULL;
   timeScaleTrue = NULL;
   oldTimeScale = NULL;
   oldTimeScaleTrue = NULL;
   deltaTimeAdapt = NULL;
   timeScaleMaxBase  = 1.0;
   timeScaleMax2Base = 1.0;
   timeScaleMin = 1.0;
   changeTimeScaleMax = 1.0;
   changeTimeScaleMin = 0.0;
   dtMinToleratedTimeScale = 1.0e-4;
   // progressStep = 1L; // deprecated Dec 18, 2013
   progressInterval = 1.0;
   writeProgressToErr = false;
   origStdOut = -1;
   origStdErr = -1;

#ifdef PV_USE_OPENCL
   clDevice = NULL;
#endif
#ifdef PV_USE_CUDA
   cudaDevice = NULL;
   gpuGroupConns = NULL;
   numGpuGroup = 0;
#endif

   layers = NULL;
   connections = NULL;
   normalizers = NULL;
   layerStatus = NULL;
   connectionStatus = NULL;
   srcPath = NULL;
   outputPath = NULL;
   // outputNamesOfLayersAndConns = NULL;
   printParamsFilename = NULL;
   printParamsStream = NULL;
   luaPrintParamsStream = NULL;
   image_file = NULL;
   nxGlobal = 0;
   nyGlobal = 0;
   nbatch = 1;
   nbatchGlobal = 1;
   ownsParams = true;
   ownsInterColComm = true;
   params = NULL;
   icComm = NULL;
//   runDelegate = NULL;
   runTimer = NULL;
   checkpointTimer = NULL;
   phaseRecvTimers = NULL;
   numColProbes = 0;
   colProbes = NULL;
   numBaseProbes = 0;
   baseProbes = NULL;
   // numConnProbes = 0;
   // connProbes = NULL;
   filenamesContainLayerNames = 0;
   filenamesContainConnectionNames = 0;
   random_seed = 0U;
   random_seed_obj = 0U;
   writeTimescales = true; //Defaults to true
   errorOnNotANumber = false;
   numThreads = 1;
   recvLayerBuffer.clear();
   verifyWrites = true; // Default for reading back and verifying when calling PV_fwrite

   return PV_SUCCESS;
}

int HyPerCol::initialize(const char * name, PV_Init* initObj)
{
   if(!initObj->getInit()){
      initObj->initialize();
   }
   pv_initObj = initObj;
   this->icComm = pv_initObj->getComm();
   this->params = pv_initObj->getParams();
   if(this->params == NULL) {
      if (icComm->globalCommRank()==0) {
         std::cout << "HyPerCol::initialize error: params have not been set." << std::endl;
         MPI_Barrier(icComm->communicator());
      }
      exit(EXIT_FAILURE);
   }

   int rank = icComm->globalCommRank();

   char const * gpu_devices = pv_initObj->getArguments()->getGPUDevices();
   char const * working_dir = pv_initObj->getArguments()->getWorkingDir();
   warmStart = pv_initObj->getArguments()->getRestartFlag();
   char const * checkpoint_read_dir = pv_initObj->getArguments()->getCheckpointReadDir();
   if (checkpoint_read_dir) {
      checkpointReadDir = strdup(pv_initObj->getArguments()->getCheckpointReadDir());
   }

   //Set up log file if it exists
   char const * log_file = pv_initObj->getArguments()->getLogFile();
   if(log_file){
      //Flush buffers before duplicating stdout
      fflush(stdout);
      fflush(stderr);
      std::cout.flush();
      std::cerr.flush();
      //Save orig stdout and stderr
      origStdOut = dup(fileno(stdout));
      if (origStdOut == -1) {
         fprintf(stderr, "Unable to duplicate stdout: %s\n", strerror(errno));
         exit(errno);
      }
      origStdErr = dup(fileno(stderr));
      if (origStdErr == -1) {
         fprintf(stderr, "Unable to duplicate stderr: %s\n", strerror(errno));
         exit(errno);
      }

      //Open log file for stdout
      if(!freopen(log_file, "w", stdout)){
         std::cout << "Error opening log file " << log_file << " for writing: " << strerror(errno) << std::endl;
         exit(EXIT_FAILURE);
      }
      //Redirect stdout to stderr
      int dupstatus = dup2(fileno(stdout), fileno(stderr));
      if (dupstatus==-1) {
         fprintf(stderr, "Error duplicating stderr to stdout (log file \"%s\"): %s\n", log_file, strerror(errno));
         exit(errno);
      }
   }

#ifdef PVP_DEBUG
   if (pv_initObj->getArguments()->getRequireReturnFlag()) {
      if( rank == 0 ) {
         printf("Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while(charhit != '\n') {
            charhit = getc(stdin);
         }
      }
      MPI_Barrier(icComm->globalCommunicator());
   }
#endif // PVP_DEBUG

   this->name = strdup(name);
   this->runTimer = new Timer(name, "column", "run    ");
   this->checkpointTimer = new Timer(name, "column", "checkpoint ");

   layers = (HyPerLayer **) malloc(layerArraySize * sizeof(HyPerLayer *));
   connections = (BaseConnection **) malloc(connectionArraySize * sizeof(BaseConnection *));
   normalizers = (NormalizeBase **) malloc(normalizerArraySize * sizeof(NormalizeBase *));

//Either opencl or cuda, not both
#if defined(PV_USE_OPENCL) && defined(PV_USE_CUDA)
   std::cout << "HyPerCol error: Can use either OpenCL or CUDA, not both\n";
   exit(PV_FAILURE);
#endif

#ifdef PV_USE_OPENCL
   //Make sure the directive set in CMake is set here
#ifndef PV_DIR
#error PV_DIR macro must be set if using OpenCL
#endif
   srcPath = (char *) calloc(PV_PATH_MAX, sizeof(char));
   strcat(srcPath, PV_DIR);
   strcat(srcPath, "/src");
   printf("============================= srcPath is %s\n", srcPath);
#endif

   // numThreads will not be set, or used until HyPerCol::run.
   // This means that threading cannot happen in the initialization or communicateInitInfo stages,
   // but that should not be a problem.

   char const * programName = pv_initObj->getArguments()->getProgramName();

   if(working_dir && columnId()==0) {
      int status = chdir(working_dir);
      if(status) {
         fprintf(stderr, "Unable to switch directory to \"%s\"\n", working_dir);
         fprintf(stderr, "chdir error: %s\n", strerror(errno));
         exit(status);
      }
   }
   working_dir = NULL;


#ifdef PV_USE_MPI // Fail if there was a parsing error, but make sure nonroot processes don't kill the root process before the root process reaches the syntax error
   int parsedStatus;
   int rootproc = 0;
   if( globalRank() == rootproc ) {
      parsedStatus = this->params->getParseStatus();
   }
   MPI_Bcast(&parsedStatus, 1, MPI_INT, rootproc, icCommunicator()->globalCommunicator());
#else
   int parsedStatus = this->params->getParseStatus();
#endif
   if( parsedStatus != 0 ) {
      exit(parsedStatus);
   }

   if (pv_initObj->getArguments()->getOutputPath()) {
      outputPath = strdup(pv_initObj->getArguments()->getOutputPath());
      if (outputPath==NULL) {exit(13); /*Lucky 13 aka right now I'm too lazy to do error reporting right*/}
   }
   random_seed = pv_initObj->getArguments()->getRandomSeed();
   ioParams(PARAMS_IO_READ);

   checkpointSignal = 0;

#ifdef PV_USE_OPENCL
   ensureDirExists(srcPath);
#endif


   simTime = startTime;
   initialStep = (long int) nearbyint(startTime/deltaTimeBase);
   currentStep = initialStep;
   finalStep = (long int) nearbyint(stopTime/deltaTimeBase);
   nextProgressTime = startTime + progressInterval;

   if(checkpointWriteFlag) {
      switch (checkpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         nextCPWriteStep = initialStep;
         nextCPWriteTime = startTime; // Should be unnecessary
         cpWriteTimeInterval = -1;
         cpWriteClockInterval = -1.0;
         break;
      case CPWRITE_TRIGGER_TIME:
         nextCPWriteStep = initialStep; // Should be unnecessary
         nextCPWriteTime = startTime;
         cpWriteStepInterval = -1;
         cpWriteClockInterval = -1.0;
         break;
      case CPWRITE_TRIGGER_CLOCK:
         nextCPWriteClock = time(NULL);
         cpWriteTimeInterval = -1;
         cpWriteStepInterval = -1;
         break;
      default:
         assert(0); // All cases of checkpointWriteTriggerMode should have been covered above.
         break;
      }
   }

   //warmStart is set if command line sets the -r option.  PV_Arguments should prevent -r and -c from being both set.
   assert(!warmStart || !checkpointReadDir);
   if (warmStart) {
      // parse_options() and ioParams() must have both been called at this point, so that we have the correct outputPath and checkpointWriteFlag
      assert(checkpointReadDir==NULL);
      checkpointReadDir = (char *) calloc(PV_PATH_MAX, sizeof(char));
      if(checkpointReadDir==NULL) {
         fprintf(stderr, "%s error: unable to allocate memory for path to checkpoint read directory.\n", programName);
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
                  fprintf(stderr, "%s error: checkpoint read directory \"%s\" too long.\n", programName, cpDirString.c_str());
                  exit(EXIT_FAILURE);
               }
            }
            else {
               fprintf(stderr, "%s error: checkpoint read directory \"%s\" is not a directory.\n", programName, cpDirString.c_str());
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
                           programName, checkpointWriteDir);
                     exit(EXIT_FAILURE);
                  }
                  int pathlen=snprintf(checkpointReadDir, PV_PATH_MAX, "%sCheckpoint%ld", cpDirString.c_str(), cp_index);
                  if (pathlen>PV_PATH_MAX) {
                     fprintf(stderr, "%s error: checkpoint read directory \"%s\" too long.\n", programName, cpDirString.c_str());
                     exit(EXIT_FAILURE);
                  }

               }
               else {
                  fprintf(stderr, "%s error: checkpoint read directory \"%s\" is not a directory.\n", programName, checkpointWriteDir);
                  exit(EXIT_FAILURE);
               }
            }
            else if (errno == ENOENT) {
               fprintf(stderr, "%s error: restarting but neither Last nor checkpointWriteDir directory \"%s\" exists.\n", programName, checkpointWriteDir);
               exit(EXIT_FAILURE);
            }
         }
         else {
            fprintf(stderr, "%s error: restarting but Last directory does not exist and checkpointWriteDir is not defined (checkpointWrite=false)\n", programName);
         }

      }
      MPI_Bcast(checkpointReadDir, PV_PATH_MAX, MPI_CHAR, 0, icComm->communicator());
   }
   if (checkpointReadDir) {
      char* origChkPtr = checkpointReadDir;
      char** splitCheckpoint = (char**)calloc(icComm->numCommBatches(), sizeof(char*));
      assert(splitCheckpoint);
      size_t count = 0;
      char * tmp = NULL;
      tmp = strtok(checkpointReadDir, ":");
      while(tmp != NULL){
         splitCheckpoint[count] = strdup(tmp);
         count++;
         if(count > icComm->numCommBatches()){
            fprintf(stderr, "Checkpoint read dir parsing error: Specified too many colon seperated checkpoint read directories. Only specify %d checkpoint directories.\n", icComm->numCommBatches());
            exit(EXIT_FAILURE);
         }
         tmp = strtok(NULL, ":");
      }
      //Make sure number matches up
      if(count != icComm->numCommBatches()){
         fprintf(stderr, "Checkpoint read dir parsing error: Specified not enough colon seperated checkpoint read directories. Running with %d batch MPIs but only %zu colon seperated checkpoint directories.\n", icComm->numCommBatches(), count);
         exit(EXIT_FAILURE);
      }
      
      //Grab this rank's actual checkpointReadDir and replace with checkpointReadDir
      checkpointReadDir = strdup(splitCheckpoint[icComm->commBatch()]);
      //Free all tmp memories
      free(origChkPtr);
      for(int i = 0; i < icComm->numCommBatches(); i++){
         free(splitCheckpoint[i]);
      }
      free(splitCheckpoint);
      assert(checkpointReadDir);
      
      checkpointReadFlag = true;
      printf("Global Rank %d process setting checkpointReadDir to %s.\n", globalRank(), checkpointReadDir);
   }

   // run only on GPU for now
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   //Default to auto assign gpus
   initializeThreads(gpu_devices);
#endif
   gpu_devices = NULL;

   //Only print rank for comm rank 0
   if(globalRank() == 0){
#ifdef PV_USE_OPENCL
      clDevice->query_device_info();
#endif
#ifdef PV_USE_CUDA
      cudaDevice->query_device_info();
#endif
   }

   //Allocate timescales for batches
   timeScale = (double*) malloc(sizeof(double) * nbatch);
   if(timeScale ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for timeScale buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   timeScaleMax = (double*) malloc(sizeof(double) * nbatch);
   if(timeScaleMax ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for timeScaleMax buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   timeScaleMax2 = (double*) malloc(sizeof(double) * nbatch);
   if(timeScaleMax2 ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for timeScaleMax2 buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   timeScaleTrue = (double*) malloc(sizeof(double) * nbatch);
   if(timeScaleTrue ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for timeScaleTrue buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   oldTimeScale = (double*) malloc(sizeof(double) * nbatch);
   if(oldTimeScale ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for oldTimeScale buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   oldTimeScaleTrue = (double*) malloc(sizeof(double) * nbatch);
   if(oldTimeScaleTrue ==NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for oldTimeScaleTrue buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   deltaTimeAdapt = (double*) malloc(sizeof(double) * nbatch);
   if(deltaTimeAdapt == NULL) {
      fprintf(stderr, "%s error: unable to allocate memory for deltaTimeAdapt buffer.\n", programName);
      exit(EXIT_FAILURE);
   }
   //Initialize timeScales to 1
   for(int b = 0; b < nbatch; b++){
      timeScaleTrue[b]       = -1;
      oldTimeScaleTrue[b]    = -1;
      timeScale[b]           = timeScaleMin;
      timeScaleMax[b]        = timeScaleMaxBase;
      timeScaleMax2[b]       = timeScaleMax2Base;
      oldTimeScale[b]        = timeScaleMin;
      deltaTimeAdapt[b]      = deltaTimeBase;
   }

   ////Here, we decide if we thread over batches (ideal) or over neurons, depending on the number of threads and number of batches
   ////User can also overwrite this behavior with a specific flag
   //if(threadBatch == -1){
   //   if(getNBatch() >= getNumThreads()){
   //      threadBatch = 1;
   //      if (globalRank()==0) {
   //         printf("%s \"%s\" defaulting to threading over batches.\n",
   //               parameters()->groupKeywordFromName(name), name);
   //      }
   //   }
   //   else{
   //      threadBatch = 0;
   //      if (columnId()==0) {
   //         printf("%s \"%s\" defaulting to threading over neurons.\n",
   //               parameters()->groupKeywordFromName(name), name);
   //      }
   //   }
   //}
   ////At this point, threadBatch must be either true or false
   //assert(threadBatch == 0 || threadBatch == 1);


//   runDelegate = NULL;

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
      assert(luaPrintParamsStream);
      const char * keyword = params->groupKeywordFromName(group_name);
      fprintf(printParamsStream->fp, "\n");
      fprintf(printParamsStream->fp, "%s \"%s\" = {\n", keyword, group_name);

      fprintf(luaPrintParamsStream->fp, "%s = {\n", group_name);
      fprintf(luaPrintParamsStream->fp, "groupType = \"%s\";\n", keyword);
   }
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_startTime(ioFlag);
   ioParam_dt(ioFlag);
   ioParam_dtAdaptFlag(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
   ioParam_useAdaptMethodExp1stOrder(ioFlag);
   ioParam_dtAdaptController(ioFlag);
   ioParam_dtAdaptTriggerLayerName(ioFlag);
   ioParam_dtAdaptTriggerOffset(ioFlag);
   ioParam_dtScaleMax(ioFlag);
   ioParam_dtScaleMin(ioFlag);
   ioParam_dtChangeMax(ioFlag);
   ioParam_dtChangeMin(ioFlag);
   ioParam_dtMinToleratedTimeScale(ioFlag);
   ioParam_stopTime(ioFlag);
   ioParam_progressInterval(ioFlag);
   ioParam_writeProgressToErr(ioFlag);
   ioParam_verifyWrites(ioFlag);
   ioParam_outputPath(ioFlag);
   ioParam_printParamsFilename(ioFlag);
   ioParam_randomSeed(ioFlag);
   ioParam_nx(ioFlag);
   ioParam_ny(ioFlag);
   ioParam_nBatch(ioFlag);
   ioParam_filenamesContainLayerNames(ioFlag);
   ioParam_filenamesContainConnectionNames(ioFlag);
   ioParam_initializeFromCheckpointDir(ioFlag);
   ioParam_defaultInitializeFromCheckpointFlag(ioFlag);
   ioParam_checkpointRead(ioFlag);
   ioParam_checkpointWrite(ioFlag);
   ioParam_checkpointWriteDir(ioFlag);
   ioParam_checkpointWriteTriggerMode(ioFlag);
   ioParam_checkpointWriteStepInterval(ioFlag);
   ioParam_checkpointWriteTimeInterval(ioFlag);
   ioParam_checkpointWriteClockInterval(ioFlag);
   ioParam_checkpointWriteClockUnit(ioFlag);
   ioParam_deleteOlderCheckpoints(ioFlag);
   ioParam_suppressLastOutput(ioFlag);
   ioParam_suppressNonplasticCheckpoints(ioFlag);
   ioParam_checkpointIndexWidth(ioFlag);
   ioParam_writeTimescales(ioFlag);
   ioParam_errorOnNotANumber(ioFlag);
   return PV_SUCCESS;
}

int HyPerCol::ioParamsFinishGroup(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_WRITE && columnId()==0) {
      assert(printParamsStream);
      assert(luaPrintParamsStream);
      fprintf(printParamsStream->fp, "};\n");
      fprintf(luaPrintParamsStream->fp, "};\n\n");
   }
   return PV_SUCCESS;
}

template <typename T>
void HyPerCol::ioParamValueRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * value) {
   switch(ioFlag) {
   case PARAMS_IO_READ:
      if (typeid(T)==typeid(int)) {
         *value = params->valueInt(group_name, param_name);
      }
      else {
         *value = params->value(group_name, param_name);
      }
      break;
   case PARAMS_IO_WRITE:
      writeParam(param_name, *value);
      break;
   }
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template void HyPerCol::ioParamValueRequired<float>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, float * value);
template void HyPerCol::ioParamValueRequired<double>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, double * value);
template void HyPerCol::ioParamValueRequired<unsigned int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, unsigned int * value);
template void HyPerCol::ioParamValueRequired<bool>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, bool * value);
template void HyPerCol::ioParamValueRequired<int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, int * value);

template <typename T>
void HyPerCol::ioParamValue(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * value, T defaultValue, bool warnIfAbsent) {
   switch(ioFlag) {
   case PARAMS_IO_READ:
      if (typeid(T)==typeid(int)) {
         *value = params->valueInt(group_name, param_name, defaultValue, warnIfAbsent);
      }
      else {
         *value = (T) params->value(group_name, param_name, defaultValue, warnIfAbsent);
      }
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
      if ( params->stringPresent(group_name, param_name) ) {
         param_string = params->stringValue(group_name, param_name, warnIfAbsent);
      }
      else {
         // parameter was not set in params file; use the default.  But default might or might not be NULL.
         if (columnId()==0 && warnIfAbsent==true) {
            if (defaultValue != NULL) {
               fprintf(stderr, "Using default value \"%s\" for string parameter \"%s\" in group \"%s\"\n", defaultValue, param_name, group_name);
            }
            else {
               fprintf(stderr, "Using default value of NULL for string parameter \"%s\" in group \"%s\"\n", param_name, group_name);
            }
         }
         param_string = defaultValue;
      }
      if (param_string!=NULL) {
         *value = strdup(param_string);
         if (*value==NULL) {
            fprintf(stderr, "Global rank %d process unable to copy param %s in group \"%s\": %s\n", globalRank(), param_name, group_name, strerror(errno));
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
            fprintf(stderr, "Global Rank %d process unable to copy param %s in group \"%s\": %s\n", globalRank(), param_name, group_name, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
      else {
         if (globalRank()==0) {
            fprintf(stderr, "%s \"%s\" error: string parameter \"%s\" is required.\n",
                            params->groupKeywordFromName(group_name), group_name, param_name);
         }
         MPI_Barrier(icComm->globalCommunicator());
         exit(EXIT_FAILURE);
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
             fprintf(stderr, "%s \"%s\" error: global rank %d process unable to copy array parameter %s: %s\n",
                   parameters()->groupKeywordFromName(name), name, globalRank(), param_name, strerror(errno));
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
template void HyPerCol::ioParamArray<int>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, int ** value, int * arraysize);

void HyPerCol::ioParam_startTime(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "startTime", &startTime, startTime);
}

void HyPerCol::ioParam_dt(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "dt", &deltaTime, deltaTime);
   deltaTimeBase = deltaTime;  // use param value as base
}

void HyPerCol::ioParam_dtAdaptFlag(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "dtAdaptFlag", &dtAdaptFlag, dtAdaptFlag);
}

void HyPerCol::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "writeTimeScaleFieldnames", &writeTimeScaleFieldnames, writeTimeScaleFieldnames);
   }
}

void HyPerCol::ioParam_useAdaptMethodExp1stOrder(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "useAdaptMethodExp1stOrder", &useAdaptMethodExp1stOrder, useAdaptMethodExp1stOrder);
   }
}

void HyPerCol::ioParam_dtAdaptController(enum ParamsIOFlag ioFlag) {
   ioParamString(ioFlag, name, "dtAdaptController", &dtAdaptController, NULL);
}

void HyPerCol::ioParam_dtAdaptTriggerLayerName(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptController"));
   if (dtAdaptController && dtAdaptController[0]) {
      ioParamString(ioFlag, name, "dtAdaptTriggerLayerName", &dtAdaptTriggerLayerName, NULL);
   }
}

void HyPerCol::ioParam_dtAdaptTriggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptTriggerLayer"));
   if (dtAdaptTriggerLayerName && dtAdaptTriggerLayerName[0]) {
      ioParamValue(ioFlag, name, "dtAdaptTriggerOffset", &dtAdaptTriggerOffset, dtAdaptTriggerOffset);
   }
}

void HyPerCol::ioParam_dtScaleMax(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "dtScaleMax", &timeScaleMaxBase, timeScaleMaxBase);
   }
}

void HyPerCol::ioParam_dtScaleMax2(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "dtScaleMax2", &timeScaleMax2Base, timeScaleMax2Base);
   }
}

void HyPerCol::ioParam_dtScaleMin(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "dtScaleMin", &timeScaleMin, timeScaleMin);
   }
}

void HyPerCol::ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
      ioParamValue(ioFlag, name, "dtMinToleratedTimeScale", &dtMinToleratedTimeScale, dtMinToleratedTimeScale);
   }
}

void HyPerCol::ioParam_dtChangeMax(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "dtChangeMax", &changeTimeScaleMax, changeTimeScaleMax);
   }
}

void HyPerCol::ioParam_dtChangeMin(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
     ioParamValue(ioFlag, name, "dtChangeMin", &changeTimeScaleMin, changeTimeScaleMin);
   }
}

void HyPerCol::ioParam_stopTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !params->present(name, "stopTime") && params->present(name, "numSteps")) {
      assert(!params->presentAndNotBeenRead(name, "startTime"));
      assert(!params->presentAndNotBeenRead(name, "dt"));
      long int numSteps = params->value(name, "numSteps");
      stopTime = startTime + numSteps * deltaTimeBase;
      if (globalRank()==0) {
         fprintf(stderr, "Warning: numSteps is deprecated.  Use startTime, stopTime and dt instead.\n");
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
      if (globalRank()==0) {
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

void HyPerCol::ioParam_verifyWrites(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "verifyWrites", &verifyWrites, verifyWrites);
}

void HyPerCol::ioParam_outputPath(enum ParamsIOFlag ioFlag) {
   // outputPath can be set on the command line.
   switch(ioFlag) {
   case PARAMS_IO_READ:
      if (outputPath==NULL) {
         if( params->stringPresent(name, "outputPath") ) {
            const char* strval = params->stringValue(name, "outputPath");
            assert(strval);
            outputPath = strdup(strval);
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
   if (printParamsFilename==NULL || printParamsFilename[0]=='\0') {
      if (columnId()==0) {
         fprintf(stderr, "Error: printParamsFilename cannot be null or the empty string.\n");
      }
      MPI_Barrier(icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
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
         exit(EXIT_FAILURE);
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

void HyPerCol::ioParam_nBatch(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "nbatch", &nbatchGlobal, nbatchGlobal);
   //Make sure numCommBatches is a multiple of nbatch specified in the params file
   if(nbatchGlobal % icComm->numCommBatches() != 0){
      std::cout << "The total number of batches (" << nbatchGlobal << ") must be a multiple of the batch width (" << icComm->numCommBatches() << ")\n";
      exit(EXIT_FAILURE);
   }
   nbatch = nbatchGlobal/icComm->numCommBatches();
}

void HyPerCol::ioParam_filenamesContainLayerNames(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "filenamesContainLayerNames", &filenamesContainLayerNames, 0);
   if(filenamesContainLayerNames < 0 || filenamesContainLayerNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainLayerNames must have the value 0, 1, or 2.\n", name);
      exit(EXIT_FAILURE);
   }
}

void HyPerCol::ioParam_filenamesContainConnectionNames(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "filenamesContainConnectionNames", &filenamesContainConnectionNames, 0);
   if(filenamesContainConnectionNames < 0 || filenamesContainConnectionNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainConnectionNames must have the value 0, 1, or 2.\n", name);
      exit(EXIT_FAILURE);
   }
}

void HyPerCol::ioParam_initializeFromCheckpointDir(enum ParamsIOFlag ioFlag) {
   ioParamString(ioFlag, name, "initializeFromCheckpointDir", &initializeFromCheckpointDir, "", true/*warnIfAbsent*/);
}

void HyPerCol::ioParam_defaultInitializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "initializeFromCheckpointDir"));
   assert(initializeFromCheckpointDir); // Should never be null after ioParam_initializeFromCheckpoint is called: an empty string serves as turning the feature off
   if (initializeFromCheckpointDir[0] != '\0') {
      ioParamValue(ioFlag, name, "defaultInitializeFromCheckpointFlag", &defaultInitializeFromCheckpointFlag, defaultInitializeFromCheckpointFlag, true/*warn if absent*/);
   }

}

void HyPerCol::ioParam_checkpointRead(enum ParamsIOFlag ioFlag) {
   // checkpointRead, checkpointReadDir, and checkpointReadDirIndex parameters were deprecated on Mar 27, 2014.
   // Instead of setting checkpointRead=true; checkpointReadDir="foo"; checkpointReadDirIndex=100,
   // pass the option <-c foo/Checkpoint100> on the command line.
   // If "-c" was passed then checkpointReadDir will have been set by HyPerCol::initialize's call to parse_options.
   // If "-r" was passed then restartFromCheckpoint will  have been set.
   if (ioFlag==PARAMS_IO_READ && !checkpointReadDir && !warmStart) {
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
      if (ioFlag==PARAMS_IO_READ && globalRank()==0 && params->present(name, "checkpointRead")) {
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
         }
         else {
            if (globalRank()==0) {
               fprintf(stderr, "HyPerCol \"%s\": checkpointWriteTriggerMode \"%s\" is not recognized.\n", name, checkpointWriteTriggerModeString);
            }
            MPI_Barrier(icCommunicator()->globalCommunicator());
            exit(EXIT_FAILURE);
         }
      }
   }
}

void HyPerCol::ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
	   assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
	   if(checkpointWriteTriggerMode == CPWRITE_TRIGGER_STEP) {
	      ioParamValue(ioFlag, name, "checkpointWriteStepInterval", &cpWriteStepInterval, 1L);
	   }
   }
}

void HyPerCol::ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
	   assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
	   if(checkpointWriteTriggerMode == CPWRITE_TRIGGER_TIME) {
	      ioParamValue(ioFlag, name, "checkpointWriteTimeInterval", &cpWriteTimeInterval, deltaTimeBase);
	   }
   }
}

void HyPerCol::ioParam_checkpointWriteClockInterval(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
      if(checkpointWriteTriggerMode == CPWRITE_TRIGGER_CLOCK) {
         ioParamValueRequired(ioFlag, name, "checkpointWriteClockInterval", &cpWriteClockInterval);
      }
   }
}

void HyPerCol::ioParam_checkpointWriteClockUnit(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerMode"));
      if(checkpointWriteTriggerMode == CPWRITE_TRIGGER_CLOCK) {
         assert(!params->presentAndNotBeenRead(name, "checkpointWriteTriggerClockInterval"));
         ioParamString(ioFlag, name, "checkpointWriteClockUnit", &cpWriteClockUnitString, "seconds");
         if (ioFlag==PARAMS_IO_READ) {
            assert(cpWriteClockUnitString);
            for (size_t n=0; n<strlen(cpWriteClockUnitString); n++) {
               cpWriteClockUnitString[n] = tolower(cpWriteClockUnitString[n]);
            }
            if (!strcmp(cpWriteClockUnitString, "second") || !strcmp(cpWriteClockUnitString, "seconds") || !strcmp(cpWriteClockUnitString, "sec") || !strcmp(cpWriteClockUnitString, "s")) {
               free(cpWriteClockUnitString);
               cpWriteClockUnitString=strdup("seconds");
               cpWriteClockSeconds = (time_t) cpWriteClockInterval;
            }
            else if (!strcmp(cpWriteClockUnitString, "minute") || !strcmp(cpWriteClockUnitString, "minutes") || !strcmp(cpWriteClockUnitString, "min") || !strcmp(cpWriteClockUnitString, "m")) {
               free(cpWriteClockUnitString);
               cpWriteClockUnitString=strdup("minutes");
               cpWriteClockSeconds = (time_t) (60.0 * cpWriteTimeInterval);
            }
            else if (!strcmp(cpWriteClockUnitString, "hour") || !strcmp(cpWriteClockUnitString, "hours") || !strcmp(cpWriteClockUnitString, "hr") || !strcmp(cpWriteClockUnitString, "h")) {
               free(cpWriteClockUnitString);
               cpWriteClockUnitString=strdup("hours");
               cpWriteClockSeconds = (time_t) (3600.0 * cpWriteTimeInterval);
            }
            else if (!strcmp(cpWriteClockUnitString, "day") || !strcmp(cpWriteClockUnitString, "days")) {
               free(cpWriteClockUnitString);
               cpWriteClockUnitString=strdup("days");
               cpWriteClockSeconds = (time_t) (86400.0 * cpWriteTimeInterval);
            }
            else {
               if (globalRank()==0) {
                  fprintf(stderr, "checkpointWriteClockUnit \"%s\" is unrecognized.  Use \"seconds\", \"minutes\", \"hours\", or \"days\".\n", cpWriteClockUnitString);
               }
               MPI_Barrier(icCommunicator()->globalCommunicator());
               exit(EXIT_FAILURE);
            }
            if (cpWriteClockUnitString==NULL) {
               // would get executed if a strdup(cpWriteClockUnitString) statement fails.
               fprintf(stderr, "Error in global rank %d process converting checkpointWriteClockUnit: %s\n", globalRank(), strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
      }
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

void HyPerCol::ioParam_suppressNonplasticCheckpoints(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      ioParamValue(ioFlag, name, "suppressNonplasticCheckpoints", &suppressNonplasticCheckpoints, suppressNonplasticCheckpoints);
   }
}

void HyPerCol::ioParam_checkpointIndexWidth(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "checkpointWrite"));
   if (checkpointWriteFlag) {
      ioParamValue(ioFlag, name, "checkpointIndexWidth", &checkpointIndexWidth, checkpointIndexWidth);
   }
}

void HyPerCol::ioParam_writeTimescales(enum ParamsIOFlag ioFlag) {
   assert(!params->presentAndNotBeenRead(name, "dtAdaptFlag"));
   if (dtAdaptFlag) {
      ioParamValue(ioFlag, name, "writeTimescales", &writeTimescales, writeTimescales);
   }
}
void HyPerCol::ioParam_errorOnNotANumber(enum ParamsIOFlag ioFlag) {
   ioParamValue(ioFlag, name, "errorOnNotANumber", &errorOnNotANumber, errorOnNotANumber);
}


template <typename T>
void HyPerCol::writeParam(const char * param_name, T value) {
   if (columnId()==0) {
      assert(printParamsStream && printParamsStream->fp);
      assert(luaPrintParamsStream && luaPrintParamsStream->fp);
      std::stringstream vstr("");
      if (typeid(value)==typeid(false)) {
         vstr << (value ? "true" : "false");
      }
      else {
         if (std::numeric_limits<T>::has_infinity) {
             if (value<=-FLT_MAX) {
                 vstr << "-infinity";
             }
             else if (value>=FLT_MAX) {
                 vstr << "infinity";
             }
             else {
                 vstr << value;
             }
         }
         else {
             vstr << value;
         }
      }
      fprintf(printParamsStream->fp, "    %-35s = %s;\n", param_name, vstr.str().c_str());
      fprintf(luaPrintParamsStream->fp, "    %-35s = %s;\n", param_name, vstr.str().c_str());
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
      assert(luaPrintParamsStream && luaPrintParamsStream->fp);
      if (svalue!=NULL) {
         fprintf(printParamsStream->fp, "    %-35s = \"%s\";\n", param_name, svalue);
         fprintf(luaPrintParamsStream->fp, "    %-35s = \"%s\";\n", param_name, svalue);
      }
      else {
         fprintf(printParamsStream->fp, "    %-35s = NULL;\n", param_name);
         fprintf(luaPrintParamsStream->fp, "    %-35s = nil;\n", param_name);
      }
   }
}

template <typename T>
void HyPerCol::writeParamArray(const char * param_name, const T * array, int arraysize) {
   if (columnId()==0) {
      assert(printParamsStream!=NULL && printParamsStream->fp!=NULL && arraysize>=0);
      assert(luaPrintParamsStream!=NULL && luaPrintParamsStream->fp!=NULL);
      assert(arraysize>=0);
      if (arraysize>0) {
         fprintf(printParamsStream->fp, "    %-35s = [", param_name);
         fprintf(luaPrintParamsStream->fp, "    %-35s = {", param_name);
         for (int k=0; k<arraysize-1; k++) {
            fprintf(printParamsStream->fp, "%f,", (float) array[k]);
            fprintf(luaPrintParamsStream->fp, "%f,", (float) array[k]);
         }
         fprintf(printParamsStream->fp, "%f];\n", (float) array[arraysize-1]);
         fprintf(luaPrintParamsStream->fp, "%f};\n", (float) array[arraysize-1]);
      }
   }
}
// Declare the instantiations of writeParam that occur in other .cpp files; otherwise you'll get linker errors.
template void HyPerCol::writeParamArray<float>(const char * param_name, const float * array, int arraysize);
template void HyPerCol::writeParamArray<int>(const char * param_name, const int * array, int arraysize);


int HyPerCol::checkDirExists(const char * dirname, struct stat * pathstat) {
   // check if the given directory name exists for the rank zero process
   // the return value is zero if a successful stat(2) call and the error
   // if unsuccessful.  pathstat contains the result of the buffer from the stat call.
   // The rank zero process is the only one that calls stat(); it then Bcasts the
   // result to the rest of the processes.
   assert(pathstat);

   int rank = columnId();
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


static inline int _mkdir(const char *dir) {
   mode_t dirmode = S_IRWXU | S_IRWXG | S_IRWXO;
   char tmp[PV_PATH_MAX];
   char *p = NULL;
   int status = 0;

   int num_chars_needed = snprintf(tmp,sizeof(tmp),"%s",dir);
   if (num_chars_needed > PV_PATH_MAX) {
      fflush(stdout);
      fprintf(stderr,"Path \"%s\" is too long.",dir);
      exit(EXIT_FAILURE);
   }

   int len = strlen(tmp);
   if(tmp[len - 1] == '/')
      tmp[len - 1] = 0;

   for(p = tmp + 1; *p; p++)
      if(*p == '/') {
         *p = 0;
         status |= mkdir(tmp, dirmode);
         if(status != 0 && errno != EEXIST){
            return status;
         }
         *p = '/';
      }
   status |= mkdir(tmp, dirmode);
   if(errno == EEXIST){
      status = 0;
   }
   return status;
}

int HyPerCol::ensureDirExists(const char * dirname) {
   // see if path exists, and try to create it if it doesn't.
   // Since only rank 0 process should be reading and writing, only rank 0 does the mkdir call
   int rank = columnId();
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

         //Try up to 5 times until it works
         int const numAttempts = 5;
         for(int attemptNum = 0; attemptNum < numAttempts; attemptNum++){
            int mkdirstatus = _mkdir(dirname);
            if( mkdirstatus != 0 ) {
               if(attemptNum == numAttempts - 1){
                  fflush(stdout);
                  fprintf(stderr, "Directory \"%s\" could not be created: %s; Exiting\n", dirname, strerror(errno));
                  exit(EXIT_FAILURE);
               }
               else{
                  fflush(stdout);
                  fprintf(stderr, "Directory \"%s\" could not be created: %s; Retrying %d out of %d\n", dirname, strerror(errno), attemptNum + 1, numAttempts);
                  sleep(1);
               }
            }
            else{
               break;
            }
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

int HyPerCol::globalRank()
{
   return icComm->globalCommRank();
}

int HyPerCol::columnId()
{
   return icComm->commRank();
}

int HyPerCol::numberOfColumns()
{
   return icComm->commSize();
}

int HyPerCol::numberOfGlobalColumns()
{
   return icComm->globalCommSize();
}

int HyPerCol::commColumn()
{
   return icComm->commColumn();
}

int HyPerCol::commRow()
{
   return icComm->commRow();
}

int HyPerCol::commBatch()
{
   return icComm->commBatch();
}

int HyPerCol::numCommColumns()
{
   return icComm->numCommColumns();
}

int HyPerCol::numCommRows()
{
   return icComm->numCommRows();
}

int HyPerCol::numCommBatches()
{
   return icComm->numCommBatches();
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
      HyPerLayer ** newLayers = (HyPerLayer **) realloc(layers, layerArraySize * sizeof(HyPerLayer *));
      if (newLayers==NULL) {
         fprintf(stderr, "Global rank %d process unable to append layer %d (\"%s\") to list of layers: %s", globalRank(), numLayers, l->getName(), strerror(errno));
         exit(EXIT_FAILURE);
      }
      layers = newLayers;
      // HyPerLayer ** newLayers = (HyPerLayer **) malloc( layerArraySize * sizeof(HyPerLayer *) );
      // assert(newLayers);
      // for(int k=0; k<numLayers; k++) {
      //    newLayers[k] = layers[k];
      // }
      // free(layers);
      // layers = newLayers;
   }
   layers[numLayers++] = l;
   if (l->getPhase() >= numPhases) numPhases = l->getPhase()+1;
   return (numLayers - 1);
}

int HyPerCol::addConnection(BaseConnection * conn)
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
      BaseConnection ** newConnections = (BaseConnection **) malloc( connectionArraySize * sizeof(BaseConnection *) );
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

int HyPerCol::addNormalizer(NormalizeBase * normalizer) {
   assert((size_t) numNormalizers <= normalizerArraySize);
   if ((size_t) numNormalizers == normalizerArraySize) {
      normalizerArraySize += RESIZE_ARRAY_INCR;
      NormalizeBase ** newNormalizers = (NormalizeBase **) realloc(normalizers, normalizerArraySize*sizeof(NormalizeBase *));
      if(newNormalizers==NULL) {
         fprintf(stderr, "HyPerCol \"%s\" on global rank %d unable to resize normalizers array to size %zu\n", name, globalRank(), normalizerArraySize);
         exit(EXIT_FAILURE);
      }
      normalizers = newNormalizers;
   }
   normalizers[numNormalizers++] = normalizer;
   return PV_SUCCESS;
}

  // typically called by buildandrun via HyPerCol::run()
int HyPerCol::run(double start_time, double stop_time, double dt)
{
   startTime = start_time;
   stopTime = stop_time;
   deltaTime = dt;

   int status = PV_SUCCESS;
   if (!readyFlag) {
      ensureDirExists(outputPath);
      if (columnId()==0 && dtAdaptFlag && writeTimescales){
         size_t timeScaleFileNameLen = strlen(outputPath) + strlen("/HyPerCol_timescales.txt");
         char timeScaleFileName[timeScaleFileNameLen+1];
         int charsneeded = snprintf(timeScaleFileName, timeScaleFileNameLen+1, "%s/HyPerCol_timescales.txt", outputPath);
         assert(charsneeded<=timeScaleFileNameLen);
         timeScaleStream.open(timeScaleFileName);
         timeScaleStream.precision(17);
      }
      assert(printParamsFilename && printParamsFilename[0]);
      std::string printParamsFileString("");
      if (printParamsFilename[0] != '/') {
         printParamsFileString += outputPath;
         printParamsFileString += "/";
      }
      printParamsFileString += printParamsFilename;

      setNumThreads(false/*don't print messages*/);
      //When we call processParams, the communicateInitInfo stage will run, which can put out a lot of messages.
      //So if there's a problem with the -t option setting, the error message can be hard to find.
      //Instead of printing the error messages here, we will call setNumThreads a second time after
      //processParams(), and only then print messages.

      // processParams function does communicateInitInfo stage, sets up adaptive time step, and prints params
      status = processParams(printParamsFileString.c_str());
      MPI_Barrier(icCommunicator()->communicator());
      if (status != PV_SUCCESS) {
         fprintf(stderr, "HyPerCol \"%s\" failed to run.\n", name);
         exit(EXIT_FAILURE);
      }
      if (pv_initObj->getArguments()->getDryRunFlag()) { return PV_SUCCESS; }

      int thread_status = setNumThreads(true/*now, print messages related to setting number of threads*/);
      MPI_Barrier(icComm->globalCommunicator());
      if (thread_status !=PV_SUCCESS) {
         exit(EXIT_FAILURE);
      }

#ifdef PV_USE_OPENMP_THREADS
      assert(numThreads>0); // setNumThreads should fail if it sets numThreads less than or equal to zero
      omp_set_num_threads(numThreads);
#endif // PV_USE_OPENMP_THREADS

      int (HyPerCol::*layerInitializationStage)(int) = NULL;
      int (HyPerCol::*connInitializationStage)(int) = NULL;

      // allocateDataStructures stage
      layerInitializationStage = &HyPerCol::layerAllocateDataStructures;
      connInitializationStage = &HyPerCol::connAllocateDataStructures;
      doInitializationStage(layerInitializationStage, connInitializationStage, "allocateDataStructures");

      // do allocation stage for probes
      for (int i=0; i<numBaseProbes; i++) {
         BaseProbe * p = baseProbes[i];
         int pstatus = p->allocateDataStructures();
         if (pstatus==PV_SUCCESS) {
            if (globalRank()==0) printf("Probe \"%s\" allocateDataStructures completed.\n", p->getName());
         }
         else {
            assert(pstatus == PV_FAILURE); // PV_POSTPONE etc. hasn't been implemented for probes yet.
            exit(EXIT_FAILURE); // Any error message should be printed by probe's allocateDataStructures function
         }
      }

      //Allocate all phaseRecvTimers
      phaseRecvTimers = (Timer**) malloc(numPhases * sizeof(Timer*));
      for(int phase = 0; phase < numPhases; phase++){
         char tmpStr[10];
         sprintf(tmpStr, "phRecv%d", phase);
         phaseRecvTimers[phase] = new Timer(name, "column", tmpStr);
      }

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
         checkpointRead();
      }

      // setInitialValues stage sets the initial values of layers and connections, either from params or from checkpoint
      layerInitializationStage = &HyPerCol::layerSetInitialValues;
      connInitializationStage = &HyPerCol::connSetInitialValues;
      doInitializationStage(layerInitializationStage, connInitializationStage, "setInitialValues");
      free(layerStatus); layerStatus = NULL;
      free(connectionStatus); connectionStatus = NULL;

      // Initial normalization moved here to facilitate normalizations of groups of HyPeConns
      normalizeWeights();
      for (int c = 0; c < numConnections; c++) {
         connections[c]->finalizeUpdate(simTime, deltaTimeBase);
      }

      // publish initial conditions
      //
      for (int l = 0; l < numLayers; l++) {
         layers[l]->publish(icComm, simTime);
         //layers[l]->updateActiveIndices();
      }

      // wait for all published data to arrive and update active indices;
      //
      for (int l = 0; l < numLayers; l++) {
         icComm->wait(layers[l]->getLayerId());
         layers[l]->updateActiveIndices();
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

      readyFlag = true;
   }


//   if (runDelegate) {
//      // let delegate advance the time
//      //
//      runDelegate->run(simTime, stopTime);
//   }

#ifdef TIMER_ON
   start_clock();
#endif
   // time loop
   //
   long int step = 0;
   assert(status == PV_SUCCESS);
   while (simTime < stopTime - deltaTime/2.0 && status != PV_EXIT_NORMALLY) {
      // Should we move the if statement below into advanceTime()?
      // That way, the routine that polls for SIGUSR1 and sets checkpointSignal is the same
      // as the routine that acts on checkpointSignal and clears it, which seems clearer.  --pete July 7, 2015
      if( checkpointWriteFlag && (advanceCPWriteTime() || checkpointSignal) ) {
         // the order should be advanceCPWriteTime() || checkpointSignal so that advanceCPWriteTime() is called even if checkpointSignal is true.
         // that way advanceCPWriteTime's calculation of the next checkpoint time won't be thrown off.
         char cpDir[PV_PATH_MAX];
         int stepFieldWidth;
         if (checkpointIndexWidth >= 0) {
            stepFieldWidth = checkpointIndexWidth;
         }
         else {
            stepFieldWidth = (int) floor(log10((stopTime - startTime)/deltaTime))+1;
         }
         int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%0*ld", checkpointWriteDir, stepFieldWidth, currentStep);
         if(chars_printed >= PV_PATH_MAX) {
            if (globalRank()==0) {
               fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", checkpointWriteDir, currentStep);
               exit(EXIT_FAILURE);
            }
         }
         if ( !checkpointReadFlag || strcmp(checkpointReadDir, cpDir) ) {
            /* Note: the strcmp isn't perfect, since there are multiple ways to specify a path that points to the same directory */
            if (globalRank()==0) {
               printf("Checkpointing, simTime = %f\n", simulationTime());
            }

            checkpointWrite(cpDir);
         }
         else {
            if (globalRank()==0) {
               printf("Skipping checkpoint at time %f, since this would clobber the checkpointRead checkpoint.\n", simulationTime());
            }
         }
         if (checkpointSignal) {
            printf("Global rank %d: checkpointing in response to SIGUSR1.\n", globalRank());
            checkpointSignal = 0;
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

   const bool exitOnFinish = false;
   exitRunLoop(exitOnFinish);

#ifdef TIMER_ON
   stop_clock();
#endif

   return status;
}

// This routine sets the numThreads member variable.  It should only be called by the run() method,
// and only inside the !ready if-statement.
int HyPerCol::setNumThreads(bool printMessagesFlag) {
   bool printMsgs0 = printMessagesFlag && globalRank()==0;
   int thread_status = PV_SUCCESS;
   PV_Arguments const * arguments = pv_initObj->getArguments();
   int num_threads = 0;
#ifdef PV_USE_OPENMP_THREADS
   int max_threads = pv_initObj->getMaxThreads();
   int comm_size = icComm->globalCommSize();
   if (printMsgs0) {
      printf("Maximum number of OpenMP threads%s is %d\nNumber of MPI processes is %d.\n",
            comm_size==1 ? "" : " (over all processes)", max_threads, comm_size);
   }
   if (arguments->getUseDefaultNumThreads()) {
      num_threads = max_threads/comm_size; // integer arithmetic
      if (num_threads == 0) {
         num_threads = 1;
         if (printMsgs0) {
            fprintf(stderr, "Warning: more MPI processes than available threads.  Processors may be oversubscribed.\n");
         }
      }
   }
   else {
      num_threads = arguments->getNumThreads();
   }
   if (num_threads>0) {
      if (printMsgs0) {
         printf("Number of threads used is %d\n", num_threads);
      }
   }
   else if (num_threads==0) {
      thread_status = PV_FAILURE;
      if (printMsgs0) {
         fflush(stdout);
         fprintf(stderr, "%s error: number of threads must be positive (was set to zero)\n", arguments->getProgramName());
      }
   }
   else {
      assert(num_threads<0);
      thread_status = PV_FAILURE;
      if (printMsgs0) {
         fflush(stdout);
         fprintf(stderr, "%s was compiled with PV_USE_OPENMP_THREADS; therefore the \"-t\" argument is required.\n", arguments->getProgramName());
      }
   }
#else // PV_USE_OPENMP_THREADS
   if (arguments->getUseDefaultNumThreads()) {
      num_threads = 1;
      if (printMsgs0) {
         printf("Number of threads used is 1 (Compiled without OpenMP.\n");
      }
   }
   else {
      num_threads = arguments->getNumThreads();
      if (num_threads < 0) { num_threads = 1; }
      if (num_threads != 1) { thread_status = PV_FAILURE; }
   }
   if (printMsgs0) {
      if (thread_status!=PV_SUCCESS) {
         fflush(stdout);
         fprintf(stderr, "%s error: PetaVision must be compiled with OpenMP to run with threads.\n", arguments->getProgramName());
      }
   }
#endif // PV_USE_OPENMP_THREADS
   //set num_threads to member variable
   this->numThreads = num_threads;
   return thread_status;
}

int HyPerCol::processParams(char const * path) {
   if (!paramsProcessedFlag) {
      layerStatus = (int *) calloc((size_t) numLayers, sizeof(int));
      if (layerStatus==NULL) {
         fprintf(stderr, "Global rank %d process unable to allocate memory for status of %zu layers: %s\n", globalRank(), (size_t) numLayers, strerror(errno));
      }
      connectionStatus = (int *) calloc((size_t) numConnections, sizeof(int));
      if (connectionStatus==NULL) {
         fprintf(stderr, "Global rank %d process unable to allocate memory for status of %zu connections: %s\n", globalRank(), (size_t) numConnections, strerror(errno));
      }
   
      int (HyPerCol::*layerInitializationStage)(int) = NULL;
      int (HyPerCol::*connInitializationStage)(int) = NULL;
   
      // do communication step for layers and connections
      layerInitializationStage = &HyPerCol::layerCommunicateInitInfo;
      connInitializationStage = &HyPerCol::connCommunicateInitInfo;
      doInitializationStage(layerInitializationStage, connInitializationStage, "communicateInitInfo");
   
      // do communication step for probes
      // This is where probes are added to their respective target layers and connections
      for (int i=0; i<numBaseProbes; i++) {
         BaseProbe * p = baseProbes[i];
         int pstatus = p->communicateInitInfo();
         if (pstatus==PV_SUCCESS) {
            if (globalRank()==0) printf("Probe \"%s\" communicateInitInfo completed.\n", p->getName());
         }
         else {
            assert(pstatus == PV_FAILURE); // PV_POSTPONE etc. hasn't been implemented for probes yet.
            // A more detailed error message should be printed by probe's communicateInitInfo function.
            fprintf(stderr, "Probe \"%s\" communicateInitInfo failed.\n", p->getName());
            return PV_FAILURE;
         }
      }
   
      // add the dtAdaptController, if there is one.
      if (dtAdaptController && dtAdaptController[0]) {
         dtAdaptControlProbe = getColProbeFromName(dtAdaptController);
         if (dtAdaptControlProbe==NULL) {
            if (globalRank()==0) {
               fprintf(stderr, "HyPerCol \"%s\" error: dtAdaptController \"%s\" does not refer to a ColProbe in the HyPerCol.\n",
                     this->getName(), dtAdaptController);
            }
            return PV_FAILURE;
         }
   
         // add the dtAdaptTriggerLayer, if there is one.
         if (dtAdaptTriggerLayerName && dtAdaptTriggerLayerName[0]) {
            dtAdaptTriggerLayer = getLayerFromName(dtAdaptTriggerLayerName);
            if (dtAdaptTriggerLayer==NULL) {
               if (globalRank()==0) {
                  fprintf(stderr, "HyPerCol \"%s\" error: dtAdaptTriggerLayerName \"%s\" does not refer to layer in the column.\n", this->getName(), dtAdaptTriggerLayerName);
               }
               return PV_FAILURE;
            }
         }
      }
   }

   // Print a cleaned up version of params to the file given by printParamsFilename
   parameters()->warnUnread();
   std::string printParamsPath = "";
   if (path!=NULL && path[0] != '\0') {
      outputParams(path);
   }
   else {
      if (globalRank()==0) {
         printf("HyPerCol \"%s\": path for printing parameters file was empty or null.\n", name);
      }
   }
   paramsProcessedFlag = true;
   return PV_SUCCESS;
}

int HyPerCol::doInitializationStage(int (HyPerCol::*layerInitializationStage)(int), int (HyPerCol::*connInitializationStage)(int), const char * stageName) {
   int status = PV_SUCCESS;
   for (int l=0; l<numLayers; l++) {
      layerStatus[l]=PV_POSTPONE;
   }
   for (int c=0; c<numConnections; c++) {
      connectionStatus[c]=PV_POSTPONE;
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
               if (globalRank()==0) printf("Layer \"%s\" %s completed.\n", layers[l]->getName(), stageName);
               break;
            case PV_POSTPONE:
               if (globalRank()==0) printf("Layer \"%s\": %s postponed.\n", layers[l]->getName(), stageName);
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
         if (connectionStatus[c]==PV_POSTPONE) {
            int status = (this->*connInitializationStage)(c);
            switch (status) {
            case PV_SUCCESS:
               connectionStatus[c] = PV_SUCCESS;
               numPostponedConns--;
               assert(numPostponedConns>=0);
               if (globalRank()==0) printf("Connection \"%s\" %s completed.\n", connections[c]->getName(), stageName);
               break;
            case PV_POSTPONE:
               if (globalRank()==0) printf("Connection \"%s\" %s postponed.\n", connections[c]->getName(), stageName);
               break;
            case PV_FAILURE:
               exit(EXIT_FAILURE); // Error message should be printed in connection's communicateInitInfo().
               break;
            default:
               assert(0); // This shouldn't be possible
            }
         }
      }
   }
   while (numPostponedLayers < prevNumPostponedLayers || numPostponedConns < prevNumPostponedConns);

   if (numPostponedLayers != 0 || numPostponedConns != 0) {
      printf("%s loop has hung on global rank %d process.\n", stageName, globalRank());
      for (int l=0; l<numLayers; l++) {
         if (layerStatus[l]==PV_POSTPONE) {
            printf("Layer \"%s\" on global rank %d is still postponed.\n", layers[l]->getName(), globalRank());
         }
      }
      for (int c=0; c<numConnections; c++) {
         if (layerStatus[c]==PV_POSTPONE) {
            printf("Connection \"%s\" on global rank %d is still postponed.\n", connections[c]->getName(), globalRank());
         }
      }
      exit(EXIT_FAILURE);
   }
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
   BaseConnection * conn = connections[c];
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
   assert(c>=0 && c<numConnections);
   BaseConnection * conn = connections[c];
   assert(conn->getDataStructuresAllocatedFlag()==false);
   int status = conn->allocateDataStructures();
   if (status==PV_SUCCESS) conn->setDataStructuresAllocatedFlag();
   return status;
}

int HyPerCol::layerSetInitialValues(int l) {
   HyPerLayer * layer = layers[l];
   assert(l>=0 && l<numLayers && layer->getInitialValuesSetFlag()==false);
   int status = layer->initializeState();
   if (status==PV_SUCCESS) layer->setInitialValuesSetFlag();
   return status;
}

int HyPerCol::connSetInitialValues(int c) {
   BaseConnection * conn = connections[c];
   assert(c>=0 && c<numConnections && conn->getInitialValuesSetFlag()==false);
   int status = conn->initializeState();
   if (status==PV_SUCCESS) conn->setInitialValuesSetFlag();
   return status;
}

int HyPerCol::normalizeWeights() {
   int status = PV_SUCCESS;
   //   for (int c=0; c < numConnections; c++) {
   //      NormalizeBase * normalizer = connections[c]->getNormalizer();
   //      if (normalizer) { status = normalizer->normalizeWeightsWrapper(); }
   //      if (status != PV_SUCCESS) {
   //         fprintf(stderr, "Normalizer failed for connection \"%s\".\n", connections[c]->getName());
   //         exit(EXIT_FAILURE);
   //      }
   //   }
   for (int n = 0; n < numNormalizers; n++) {
      NormalizeBase * normalizer = normalizers[n];
      if (normalizer) { status = normalizer->normalizeWeightsWrapper(); }
      if (status != PV_SUCCESS) {
         fprintf(stderr, "Normalizer \"%s\" failed.\n", normalizers[n]->getName());
      }
   }
   return status;
}

int HyPerCol::initPublishers() {
   for( int l=0; l<numLayers; l++ ) {
      // PVLayer * clayer = layers[l]->getCLayer();
      icComm->addPublisher(layers[l]);
   }
   for( int c=0; c<numConnections; c++ ) {
      icComm->subscribe(connections[c]);
   }

   return PV_SUCCESS;
}

double * HyPerCol::adaptTimeScale(){
   for (int b=0; b<nbatch; b++) {
      oldTimeScaleTrue[b] = timeScaleTrue[b];
      oldTimeScale[b] = timeScale[b];
   }
   calcTimeScaleTrue();
   for(int b = 0; b < nbatch; b++){
      // forces timeScale to remain constant if Error is changing too rapidly
      // if change in timeScaleTrue is negative, revert to minimum timeScale
      // TODO?? add ability to revert all dynamical variables to previous values if Error increases?

      //Set timeScaleTrue to new minTimeScale
      double minTimeScaleTmp = timeScaleTrue[b];

      // force the minTimeScaleTmp to be <= timeScaleMaxBase
      minTimeScaleTmp = minTimeScaleTmp < timeScaleMaxBase ? minTimeScaleTmp : timeScaleMaxBase;

      // set the timeScale to minTimeScaleTmp iff minTimeScaleTmp > 0, otherwise default to timeScaleMin
      timeScale[b] = minTimeScaleTmp > 0.0 ? minTimeScaleTmp : timeScaleMin;

      // only let the timeScale change by a maximum percentage of oldTimescale of changeTimeScaleMax on any given time step
      double changeTimeScale = (timeScale[b] - oldTimeScale[b])/oldTimeScale[b];
      timeScale[b] = changeTimeScale < changeTimeScaleMax ? timeScale[b] : oldTimeScale[b] * (1 + changeTimeScaleMax);

      //Positive if timescale increased, error decreased
      //Negative if timescale decreased, error increased
      double changeTimeScaleTrue = timeScaleTrue[b] - oldTimeScaleTrue[b];
      // keep the timeScale constant if the error is decreasing too rapidly
      if (changeTimeScaleTrue > changeTimeScaleMax){
         timeScale[b] = oldTimeScale[b];
      }

      // if error is increasing,
      if (changeTimeScaleTrue < changeTimeScaleMin){
         //retreat back to the MIN(timeScaleMin, minTimeScaleTmp)
         if (minTimeScaleTmp > 0.0){
            double setTimeScale = oldTimeScale[b] < timeScaleMin ? oldTimeScale[b] : timeScaleMin;
            timeScale[b] = setTimeScale < minTimeScaleTmp ? setTimeScale : minTimeScaleTmp;
            //timeScale =  minTimeScaleTmp < timeScaleMin ? minTimeScaleTmp : setTimeScale;
         }
         else{
            timeScale[b] = timeScaleMin;
         }
      }

      if(timeScale[b] > 0 && timeScaleTrue[b] > 0 && timeScale[b] > timeScaleTrue[b]){
         std::cout << "timeScale is bigger than timeScaleTrue\n";
         std::cout << "timeScale: " << timeScale[b] << "\n";
         std::cout << "timeScaleTrue: " << timeScaleTrue[b] << "\n";
         std::cout << "minTimeScaleTmp: " << minTimeScaleTmp << "\n";
         std::cout << "oldTimeScaleTrue " << oldTimeScaleTrue[b] << "\n";
         exit(EXIT_FAILURE);
      }

      // deltaTimeAdapt is only used internally to set scale of each update step
      deltaTimeAdapt[b] = timeScale[b] * deltaTimeBase;
   }
   return deltaTimeAdapt;
}

  // time scale adaptation using model of E(dt) ~= E_0 * exp(-dt/tau_eff) + E_inf
  // to first order:
  //   E(0)  = E_0 + E_inf
  //   E(dt) = E_0 * (1 - dt/tau_eff) + E_inf
  // solving for tau_eff yields
  //   tau_eff = dt * E_0 / |dE| <= dt * E(0) / |dE|
  // where
  //   dE = E(0) - E(dt)
  // to 2nd order in a Taylor series expansion:  optim_dt ~= tau_eff -> argmin E'(optim_dt)
  // where E' is the Tayler series expansion of E(dt) to 2nd order in dt
double * HyPerCol::adaptTimeScaleExp1stOrder(){
   for (int b=0; b<nbatch; b++) {
     oldTimeScaleTrue[b]    = timeScaleTrue[b];
     oldTimeScale[b]        = timeScale[b];
   }
   calcTimeScaleTrue(); // sets timeScaleTrue[b] to sqrt(Energy(t+dt)/|I|^2))^-1
   for(int b = 0; b < nbatch; b++){
     
     // if ((timeScale[b] == timeScaleMin) && (oldTimeScale[b] == timeScaleMax2[b])) {
     //   timeScaleMax2[b] = (1 + changeTimeScaleMin) * timeScaleMax2[b];
     // }
       
     double E_dt  =  timeScaleTrue[b]; 
     double E_0   =  oldTimeScaleTrue[b];
     double dE_dt = (E_0 - E_dt)  /  deltaTimeAdapt[b];

     if ( (dE_dt <= 0.0) || (E_0 <= 0) || (E_dt <= 0) ) {
       timeScale[b]      = timeScaleMin;
       deltaTimeAdapt[b] = timeScale[b] * deltaTimeBase;
       timeScaleMax[b]   = timeScaleMaxBase;
       //timeScaleMax2[b]  = oldTimeScale[b]; // set Max2 to value of time scale at which instability appeared
     }
     else {
       double tau_eff = E_0 / dE_dt;

       // dt := timeScaleMaxBase * tau_eff
       timeScale[b] = changeTimeScaleMax * tau_eff / deltaTimeBase;
       //timeScale[b] = (timeScale[b] <= timeScaleMax2[b]) ? timeScale[b] : timeScaleMax2[b];
       timeScale[b] = (timeScale[b] <= timeScaleMax[b]) ? timeScale[b] : timeScaleMax[b];
       timeScale[b] = (timeScale[b] <  timeScaleMin) ? timeScaleMin : timeScale[b];

       if (timeScale[b] == timeScaleMax[b]) {
	 timeScaleMax[b] = (1 + changeTimeScaleMin) * timeScaleMax[b];
       }
       
       // deltaTimeAdapt is only used internally to set scale of each update step
       deltaTimeAdapt[b] = timeScale[b] * deltaTimeBase;

       //std::cout << "simTime: " << simTime << "\n";
       //std::cout << "oldTimeScaleTrue: " << oldTimeScaleTrue[b] << "\n";
       //std::cout << "oldTimeScale: " << oldTimeScale[b] << "\n";
       //std::cout << "E_dt: " << E_dt << "\n";
       //std::cout << "E_0: " << E_0 << "\n";
       //std::cout << "dE_dt: " << dE_dt << "\n";
       //std::cout << "tau_eff: " << tau_eff << "\n";
       //std::cout << "timeScale: " << timeScale[b] << "\n";
       //std::cout << "timeScaleMax: " << timeScaleMax[b] << "\n";
       //std::cout << "timeScaleMax2: " << timeScaleMax2[b] << "\n";
       //std::cout << "deltaTimeAdapt: " << deltaTimeAdapt[b] << "\n";
       //std::cout <<  "\n";
       
     }
   }
   return deltaTimeAdapt;
}

int HyPerCol::calcTimeScaleTrue() {
   if (!dtAdaptControlProbe) {
      // Using dtAdaptFlag w/o dtAdaptControlProbe was deprecated Feb 1, 2016.
      if (columnId()==0) {
         fflush(stdout);
         fprintf(stderr, "\n\nWARNING: Setting dtAdaptFlag without defining a dtAdaptControlProbe is deprecated.\n\n\n");
      }
      // If there is no probe controlling the adaptive timestep,
      // query all layers to check for barriers on how big the time scale can be.
      // By default, HyPerLayer::getTimeScale returns -1
      // (that is, the layer doesn't care how big the time scale is).
      // Movie and MoviePvp return minTimeScale when expecting to load a new frame
      // on next time step based on current value of deltaTime.
      // ANNNormalizeErrorLayer (deprecated) is the only other layer in pv-core
      // that overrides getTimeScale.
      for (int b=0; b<nbatch; b++) {
         // copying of timeScale and timeScaleTrue was moved to adaptTimeScale, just before the call to calcTimeScaleTrue -- Oct. 8, 2015
         // set the true timeScale to the minimum timeScale returned by each layer, stored in minTimeScaleTmp
         double minTimeScaleTmp = -1;
         for(int l = 0; l < numLayers; l++) {
            //Grab timescale
            double timeScaleTmp = layers[l]->calcTimeScale(b);
            if (timeScaleTmp > 0.0){
               //Error if smaller than tolerated
               if (timeScaleTmp < dtMinToleratedTimeScale) {
                  if (globalRank()==0) {
                     if (nbatch==1) {
                        fprintf(stderr, "Error: Layer \"%s\" returned time scale %g, less than dtMinToleratedTimeScale=%g.\n", layers[l]->getName(), timeScaleTmp, dtMinToleratedTimeScale);
                     }
                     else {
                        fprintf(stderr, "Error: Layer \"%s\", batch element %d, returned time scale %g, less than dtMinToleratedTimeScale=%g.\n", layers[l]->getName(), b, timeScaleTmp, dtMinToleratedTimeScale);
                     }
                  }
                  MPI_Barrier(icComm->globalCommunicator());
                  exit(EXIT_FAILURE);
               }
               //Grabbing lowest timeScaleTmp
               if (minTimeScaleTmp > 0.0){
                  minTimeScaleTmp = timeScaleTmp < minTimeScaleTmp ? timeScaleTmp : minTimeScaleTmp;
               }
               //Initial set
               else{
                  minTimeScaleTmp = timeScaleTmp;
               }
            }
         }
         timeScaleTrue[b] = minTimeScaleTmp;
      }
   }
   else {
      // If there is a probe controlling the adaptive timestep, use its value for timeScaleTrue.
      std::vector<double> colProbeValues;
      bool triggersNow = false;
      if (dtAdaptTriggerLayer) {
         double triggerTime = dtAdaptTriggerLayer->getNextUpdateTime() - dtAdaptTriggerOffset;
         triggersNow = fabs(simTime - triggerTime) < (deltaTimeBase/2);
      }
      if (triggersNow) {
         colProbeValues.assign(nbatch, -1.0);
      }
      else {
         dtAdaptControlProbe->getValues(simTime, &colProbeValues);
      }
      assert(colProbeValues.size()==nbatch); // getValues sets dtAdaptControlProbe->vectorSize to be equal to nbatch
      for (int b=0; b<nbatch; b++) {
         double timeScaleProbe = colProbeValues.at(b);
         if (timeScaleProbe > 0 && timeScaleProbe < dtMinToleratedTimeScale) {
            if (globalRank()==0) {
               if (nbatch==1) {
                  fprintf(stderr, "Error: Probe \"%s\" has time scale %g, less than dtMinToleratedTimeScale=%g.\n", dtAdaptControlProbe->getName(), timeScaleProbe, dtMinToleratedTimeScale);
               }
               else {
                  fprintf(stderr, "Error: Layer \"%s\", batch element %d, has time scale %g, less than dtMinToleratedTimeScale=%g.\n", dtAdaptControlProbe->getName(), b, timeScaleProbe, dtMinToleratedTimeScale);
               }
            }
            MPI_Barrier(icComm->globalCommunicator());
            exit(EXIT_FAILURE);
         }
         timeScaleTrue[b] =  timeScaleProbe;
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::advanceTime(double sim_time)
{
   if (simTime >= nextProgressTime) {
      nextProgressTime += progressInterval;
      if (columnId() == 0) {
         FILE * progressStream = writeProgressToErr ? stderr : stdout;
         time_t current_time;
         time(&current_time);
         fprintf(progressStream, "   time==%f  %s", sim_time, ctime(&current_time));
      }
   }

   runTimer->start();

   // make sure simTime is updated even if HyPerCol isn't running time loop
   // triggerOffset might fail if simTime does not advance uniformly because
   // simTime could skip over tigger event
   // !!!TODO: fix trigger layer to compute timeScale so as not to allow bypassing trigger event
   simTime = sim_time + deltaTimeBase;
   currentStep++;

   deltaTime = deltaTimeBase;
   if (dtAdaptFlag){ // adapt deltaTime
     // hack code to test new adapt time scale method using exponential approx to energy
     //bool useAdaptMethodExp1stOrder = false; //true;
     if (useAdaptMethodExp1stOrder){
       adaptTimeScaleExp1stOrder();}
     else{
       adaptTimeScale();
     }
     if(writeTimescales && columnId() == 0) {
       if (writeTimeScaleFieldnames) {
         timeScaleStream << "sim_time = " << sim_time << "\n";
       }
       else {
         timeScaleStream << sim_time << ", ";
       }
         for(int b = 0; b < nbatch; b++){
	   if (writeTimeScaleFieldnames) {
	     timeScaleStream << "\tbatch = " << b << ", timeScale = " << timeScale[b] << ", " << "timeScaleTrue = " << timeScaleTrue[b];
	   }
	   else {
	     timeScaleStream << b << ", " << timeScale[b] << ", " << timeScaleTrue[b];
	   }
	   if (useAdaptMethodExp1stOrder) {
	     if (writeTimeScaleFieldnames) {
	       timeScaleStream <<  ", " << "timeScaleMax = " << timeScaleMax[b] << std::endl;
	       // timeScaleStream <<  ", " << "timeScaleMax = " << timeScaleMax[b] <<  ", " << "timeScaleMax2 = " << timeScaleMax2[b] << std::endl;
	     }
	     else {
	       // timeScaleStream <<  ", " << timeScaleMax[b] <<  ", " << timeScaleMax2[b] << std::endl;
	       timeScaleStream <<  ", " << timeScaleMax[b] << std::endl;
	     }
	   }
	   else {
	     timeScaleStream << std::endl;
	   }
         }
         timeScaleStream.flush();
     }
   } // dtAdaptFlag

   // At this point all activity from the previous time step has
   // been delivered to the data store.
   //

   int status = PV_SUCCESS;
   bool exitAfterUpdate = false;

   // update the connections (weights)
   //
   for (int c = 0; c < numConnections; c++) {
      status = connections[c]->updateState(simTime, deltaTimeBase);
      if (!exitAfterUpdate) {
         exitAfterUpdate = status == PV_EXIT_NORMALLY;
      }
   }
   normalizeWeights();
   for (int c = 0; c < numConnections; c++) {
      connections[c]->finalizeUpdate(simTime, deltaTimeBase);
      connections[c]->outputState(simTime);
   }


   if (globalRank()==0) {
      int sigstatus = PV_SUCCESS;
      sigset_t pollusr1;

      sigstatus = sigpending(&pollusr1); assert(sigstatus==0);
      checkpointSignal = sigismember(&pollusr1, SIGUSR1); assert(checkpointSignal==0 || checkpointSignal==1);
      if (checkpointSignal) {
         sigstatus = sigemptyset(&pollusr1); assert(sigstatus==0);
         sigstatus = sigaddset(&pollusr1, SIGUSR1); assert(sigstatus==0);
         int result=0;
         sigwait(&pollusr1, &result);
         assert(result==SIGUSR1);
      }
   }
   // Balancing MPI_Recv is after the for-loop over phases.  Is this better than MPI_Bcast?  Should it be MPI_Isend?
   if (globalRank()==0) {
      for (int k=1; k<numberOfGlobalColumns(); k++) {
         MPI_Send(&checkpointSignal, 1/*count*/, MPI_INT, k/*destination*/, 99/*tag*/, icComm->globalCommunicator());
      }
   }

   // Each layer's phase establishes a priority for updating
   for (int phase=0; phase<numPhases; phase++) {
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      //Clear layer buffer
      recvLayerBuffer.clear();
      updateLayerBuffer.clear();
      updateLayerBufferGpu.clear();
#endif

      //Ordering needs to go recvGpu, if(recvGpu and upGpu)update, recvNoGpu, update rest 

      //Time recv for each phase
      // clear GSyn buffers
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         //Save non gpu layer recv for later
         if(!layers[l]->getRecvGpu()){
            recvLayerBuffer.push_back(layers[l]);
            continue;
         }
#endif
         //Recv GPU
         layers[l]->resetGSynBuffers(simTime, deltaTimeBase);  // deltaTimeAdapt is not used 
         phaseRecvTimers[phase]->start();
         layers[l]->recvAllSynapticInput();
         phaseRecvTimers[phase]->stop();
         //if(recvGpu and upGpu)

         //Update for gpu recv and gpu update
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         if(layers[l]->getUpdateGpu()){
#endif
            status = layers[l]->updateStateWrapper(simTime, deltaTimeBase);
            if (!exitAfterUpdate) {
               exitAfterUpdate = status == PV_EXIT_NORMALLY;
            }
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         }
         //If not updating on gpu, save for later
         else{
            updateLayerBuffer.push_back(layers[l]);
         }
#endif
      }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      //Run non gpu layers
      for(std::vector<HyPerLayer*>::iterator it = recvLayerBuffer.begin(); it < recvLayerBuffer.end(); it++){
         HyPerLayer * layer = *it;
         layer->resetGSynBuffers(simTime, deltaTimeBase);  // deltaTimeAdapt is not used 
         phaseRecvTimers[phase]->start();
         layer->recvAllSynapticInput();
         phaseRecvTimers[phase]->stop();
         if(layer->getUpdateGpu()){
            updateLayerBufferGpu.push_back(layer);
         }
         //Update for non gpu recv and non gpu update
         else{
            status = layer->updateStateWrapper(simTime, deltaTimeBase);
            if (!exitAfterUpdate) {
               exitAfterUpdate = status == PV_EXIT_NORMALLY;
            }
         }
      }
      getDevice()->syncDevice();

      //Update for non gpu recv and gpu update
      for(std::vector<HyPerLayer*>::iterator it = updateLayerBufferGpu.begin(); it < updateLayerBufferGpu.end(); it++){
         HyPerLayer * layer = *it;
         status = layer->updateStateWrapper(simTime, deltaTimeBase);
         if (!exitAfterUpdate) {
            exitAfterUpdate = status == PV_EXIT_NORMALLY;
         }
      }

      getDevice()->syncDevice();
      //Barriers for all gpus, and copy back all data structures
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         phaseRecvTimers[phase]->start();
         layers[l]->copyAllActivityFromDevice();
         layers[l]->copyAllVFromDevice();
         layers[l]->copyAllGSynFromDevice();
         layers[l]->addGpuTimers();
         phaseRecvTimers[phase]->stop();
      }

      //Update for gpu recv and non gpu update
      for(std::vector<HyPerLayer*>::iterator it = updateLayerBuffer.begin(); it < updateLayerBuffer.end(); it++){
         HyPerLayer * layer = *it;
         status = layer->updateStateWrapper(simTime, deltaTimeBase);
         if (!exitAfterUpdate) {
            exitAfterUpdate = status == PV_EXIT_NORMALLY;
         }
      }

      //Clear all buffers
      recvLayerBuffer.clear();
      updateLayerBuffer.clear();
      updateLayerBufferGpu.clear();
#endif


      //    for (int l = 0; l < numLayers; l++) {
      //       // deliver new synaptic activity to any
      //       // postsynaptic layers for which this
      //       // layer is presynaptic.
      //       layers[l]->triggerReceive(icComm);
      //    }

      //// Update the layers (activity)
      //// We don't put updateState in the same loop over layers as recvAllSynapticInput
      //// because we plan to have updateState update the datastore directly, and
      //// recvSynapticInput uses the datastore to compute GSyn.
      //for(int l = 0; l < numLayers; l++) {
      //   if (layers[l]->getPhase() != phase) continue;
      //   status = layers[l]->updateStateWrapper(simTime, deltaTimeAdapt);
      //   if (!exitAfterUpdate) {
      //      exitAfterUpdate = status == PV_EXIT_NORMALLY;
      //   }
      //}

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
      }


      // wait for all published data to arrive
      //
      char brokenlayers[numLayers];
      memset(brokenlayers, 0, (size_t) numLayers);
      for (int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         layers[l]->waitOnPublish(icComm);
         //Update active indices
         layers[l]->updateActiveIndices();

         //
         //    // also calls layer probes
         layers[l]->outputState(simTime); // also calls layer probes' outputState
         if (errorOnNotANumber) {
            for (int n=0; n<layers[l]->getNumExtended(); n++) {
               pvadata_t a = layers[l]->getLayerData()[n];
               if (a!=a) {
                  status = PV_FAILURE;
                  brokenlayers[l] = 1;
               }
            }
         }
      }
      if (status==PV_FAILURE) {
         for (int l=0; l<numLayers;l++) {
            if (brokenlayers[l]) {
               fprintf(stderr, "Layer \"%s\" has not-a-number values in the activity buffer.  Exiting.\n", layers[l]->getName());
            }
         }
         exit(EXIT_FAILURE);
      }

   }

   // Balancing MPI_Send is before the for-loop over phases.  Is this better than MPI_Bcast?
   if (globalRank()!=0) {
      MPI_Recv(&checkpointSignal, 1/*count*/, MPI_INT, 0/*source*/, 99/*tag*/, icCommunicator()->globalCommunicator(), MPI_STATUS_IGNORE);
   }

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
   time_t now; // needed only by CPWRITE_TRIGGER_CLOCK, but can't declare variables inside a case
   switch (this->checkpointWriteTriggerMode) {
   case CPWRITE_TRIGGER_STEP:
      assert(cpWriteStepInterval>0 && cpWriteTimeInterval<0 && cpWriteClockInterval<0.0);
      advanceCPTime = currentStep >= nextCPWriteStep;
      if( advanceCPTime ) {
         nextCPWriteStep += cpWriteStepInterval;
      }
      break;
   case CPWRITE_TRIGGER_TIME:
      assert(cpWriteStepInterval<0 && cpWriteTimeInterval>0 && cpWriteClockInterval<0.0);
      advanceCPTime = simTime >= nextCPWriteTime;
      if( advanceCPTime ) {
         nextCPWriteTime += cpWriteTimeInterval;
      }
      break;
   case CPWRITE_TRIGGER_CLOCK:
      assert(cpWriteStepInterval<0 && cpWriteTimeInterval<0 && cpWriteClockInterval>0.0);
      now = time(NULL);
      advanceCPTime = now >= nextCPWriteClock;
      if (advanceCPTime) {
         if (globalRank()==0) {
            printf("Checkpoint triggered at %s", ctime(&now));
         }
         nextCPWriteClock += cpWriteClockSeconds;
         if (globalRank()==0) {
            printf("Next checkpoint trigger will be at %s", ctime(&nextCPWriteClock));
         }
      }
      break;
   default:
      assert(0); // all possible cases are considered above.
      break;
   }
   return advanceCPTime;
}

int HyPerCol::checkpointRead() {
   struct timestamp_struct {
      double time; // time measured in units of dt
      long int step; // step number, usually time/dt
   };
   struct timestamp_struct timestamp;
   size_t timestamp_size = sizeof(struct timestamp_struct);
   assert(sizeof(struct timestamp_struct) == sizeof(long int) + sizeof(double));
   if( columnId()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", checkpointReadDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", checkpointReadDir);
         exit(EXIT_FAILURE);
      }
      PV_Stream * timestampfile = PV_fopen(timestamppath,"r",false/*verifyWrites*/);
      if (timestampfile == NULL) {
         fprintf(stderr, "HyPerCol::checkpointRead error: unable to open \"%s\" for reading.\n", timestamppath);
         exit(EXIT_FAILURE);
      }
      long int startpos = getPV_StreamFilepos(timestampfile);
      PV_fread(&timestamp,1,timestamp_size,timestampfile);
      long int endpos = getPV_StreamFilepos(timestampfile);
      assert(endpos-startpos==(int)timestamp_size);
      PV_fclose(timestampfile);
   }
   MPI_Bcast(&timestamp,(int) timestamp_size,MPI_CHAR,0,icCommunicator()->communicator());
   simTime = timestamp.time;
   currentStep = timestamp.step;

   double t = startTime;
   for (long int k=initialStep; k<currentStep; k++) {
      if (t >= nextProgressTime) {
         nextProgressTime += progressInterval;
      }
      t += deltaTimeBase;
   }

   if (dtAdaptFlag == true) {
      struct timescalemax_struct {
         double timeScale; // timeScale factor for increasing/decreasing dt
         double timeScaleTrue; // true timeScale as returned by HyPerLayer::getTimeScaleTrue() typically computed by an adaptTimeScaleController (ColProbe)
         double timeScaleMax; //  current maximum allowed value of timeScale as returned by HyPerLayer::getTimeScaleMaxPtr()
         double timeScaleMax2; //  current maximum allowed value of timeScale as returned by HyPerLayer::getTimeScaleMax2Ptr()
         double deltaTimeAdapt;
      };
      struct timescale_struct {
         double timeScale; // timeScale factor for increasing/decreasing dt
         double timeScaleTrue; // true timeScale as returned by HyPerLayer::getTimeScaleTrue() typically computed by an adaptTimeScaleController (ColProbe)
         double deltaTimeAdapt;
      };
      struct timescalemax_struct timescalemax[nbatch];
      struct timescale_struct timescale[nbatch];
      if (useAdaptMethodExp1stOrder) {
         for(int b = 0; b < nbatch; b++){
            timescalemax[b].timeScale = 1;
            timescalemax[b].timeScaleTrue = 1;
            timescalemax[b].timeScaleMax = 1;
            timescalemax[b].timeScaleMax2 = 1;
            timescalemax[b].deltaTimeAdapt = 1;
         }
      }
      else {
         //Default values
         for(int b = 0; b < nbatch; b++){
            timescale[b].timeScale = 1;
            timescale[b].timeScaleTrue = 1;
            timescale[b].deltaTimeAdapt = 1;
         }
      }
      size_t timescale_size = sizeof(struct timescale_struct);
      size_t timescalemax_size = sizeof(struct timescalemax_struct);
      if (useAdaptMethodExp1stOrder) {
         assert(sizeof(struct timescalemax_struct) == sizeof(double) + sizeof(double) + sizeof(double) + sizeof(double) + sizeof(double));
      }
      else {
         assert(sizeof(struct timescale_struct) == sizeof(double) + sizeof(double) + sizeof(double));
      }
      // read timeScale info
      if(columnId()==0 ) {
         char timescalepath[PV_PATH_MAX];
         int chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/timescaleinfo.bin", checkpointReadDir);
         if (chars_needed >= PV_PATH_MAX) {
            fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timescaleinfo.bin\" is too long.\n", checkpointReadDir);
            exit(EXIT_FAILURE);
         }
         PV_Stream * timescalefile = PV_fopen(timescalepath,"r",false/*verifyWrites*/);
         if (timescalefile == NULL) {
            fprintf(stderr, "HyPerCol::checkpointRead error: unable to open \"%s\" for reading: %s.\n", timescalepath, strerror(errno));
            if (useAdaptMethodExp1stOrder) {
               fprintf(stderr, "    will use default value of timeScale=%f, timeScaleTrue=%f, timeScaleMax=%f, timeScaleMax2=%f\n", 1.0, 1.0, 1.0, 1.0);
            }
            else {
               fprintf(stderr, "    will use default value of timeScale=%f, timeScaleTrue=%f\n", 1.0, 1.0);
            }
         }
         else {
            for(int b = 0; b < nbatch; b++){
               long int startpos = getPV_StreamFilepos(timescalefile);
               if (useAdaptMethodExp1stOrder) {
                  PV_fread(&timescalemax[b],1,timescalemax_size,timescalefile);
               }
               else {
                  PV_fread(&timescale[b],1,timescale_size,timescalefile);
               }
               long int endpos = getPV_StreamFilepos(timescalefile);
               if (useAdaptMethodExp1stOrder) {
                  assert(endpos-startpos==(int)sizeof(struct timescalemax_struct));
               }
               else {
                  assert(endpos-startpos==(int)sizeof(struct timescale_struct));
               }
            }
            PV_fclose(timescalefile);
         }
      }
      //Grab only the necessary part based on comm batch id
      if (useAdaptMethodExp1stOrder) {
         MPI_Bcast(&timescalemax,(int) timescalemax_size*nbatch,MPI_CHAR,0,icCommunicator()->communicator());
         for (int b = 0; b < nbatch; b++){
            timeScale[b] = timescalemax[b].timeScale;
            timeScaleTrue[b] = timescalemax[b].timeScaleTrue;
            timeScaleMax[b] = timescalemax[b].timeScaleMax;
            timeScaleMax2[b] = timescalemax[b].timeScaleMax2;
            deltaTimeAdapt[b] = timescalemax[b].deltaTimeAdapt;
         }
      }
      else {
         MPI_Bcast(&timescale,(int) timescale_size*nbatch,MPI_CHAR,0,icCommunicator()->communicator());
         for (int b = 0; b < nbatch; b++){
            timeScale[b] = timescale[b].timeScale;
            timeScaleTrue[b] = timescale[b].timeScaleTrue;
            deltaTimeAdapt[b] = timescale[b].deltaTimeAdapt;
         }
      }
   }

   if(checkpointWriteFlag) {
      char nextCheckpointPath[PV_PATH_MAX];
      int chars_needed;
      PV_Stream * nextCheckpointFile = NULL;
      switch(checkpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         readScalarFromFile(checkpointReadDir, name, "nextCheckpointStep", &nextCPWriteStep, currentStep+cpWriteStepInterval);
         break;
      case CPWRITE_TRIGGER_TIME:
         readScalarFromFile(checkpointReadDir, name, "nextCheckpointTime", &nextCPWriteTime, simTime+cpWriteTimeInterval);
         break;
      case CPWRITE_TRIGGER_CLOCK:
         // Nothing to do in this case
         break;
      default:
         // All cases of checkpointWriteTriggerMode are handled above
         assert(0);
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::writeTimers(FILE* stream){
   int rank=columnId();
   if (rank==0) {
      runTimer->fprint_time(stream);
      checkpointTimer->fprint_time(stream);
      icCommunicator()->fprintTime(stream);
      for (int c=0; c<numConnections; c++) {
         connections[c]->writeTimers(stream);
      }
      for (int phase=0; phase<numPhases; phase++) {
         if (phaseRecvTimers && phaseRecvTimers[phase]) { phaseRecvTimers[phase]->fprint_time(stream); }
         for (int n = 0; n < numLayers; n++) {
            if (layers[n] != NULL) {
               if(layers[n]->getPhase() != phase) continue;
               layers[n]->writeTimers(stream);
            }
         }
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::checkpointWrite(const char * cpDir) {
   checkpointTimer->start();
   if (columnId()==0) {
      printf("Checkpointing to directory \"%s\" at simTime = %f\n", cpDir, simTime);
      struct stat timeinfostat;
      char timeinfofilename[PV_PATH_MAX];
      int chars_needed = snprintf(timeinfofilename, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointWrite error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         exit(EXIT_FAILURE);
      }
      int statstatus = stat(timeinfofilename, &timeinfostat);
      if (statstatus == 0) {
         fprintf(stderr, "Warning: Checkpoint directory \"%s\" has existing timeinfo.bin, which is now being deleted.\n", cpDir);
         int unlinkstatus = unlink(timeinfofilename);
         if (unlinkstatus != 0) {
            fprintf(stderr, "Error deleting \"%s\": %s\n", timeinfofilename, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }

   ensureDirExists(cpDir);
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointWrite(cpDir);
   }
   for( int c=0; c<numConnections; c++ ) {
      if (connections[c]->getPlasticityFlag() || !suppressNonplasticCheckpoints) { connections[c]->checkpointWrite(cpDir); }
   }

   // Timers
   if (columnId()==0) {
      std::string timerpathstring = cpDir;
      timerpathstring += "/";
      timerpathstring += "timers.txt";

      //std::string timercsvstring = cpDir;
      //timercsvstring += "/";
      //timercsvstring += "timers.csv";

      const char * timerpath = timerpathstring.c_str();
      PV_Stream * timerstream = PV_fopen(timerpath, "w", getVerifyWrites());
      if (timerstream==NULL) {
         fprintf(stderr, "Unable to open \"%s\" for checkpointing timer information: %s\n", timerpath, strerror(errno));
         exit(EXIT_FAILURE);
      }
      writeTimers(timerstream->fp);

      //const char * timercsvpath = timercsvstring.c_str();
      //PV_Stream * timercsvstream = PV_fopen(timercsvpath, "w", getVerifyWrites());
      //if (timercsvstream==NULL) {
      //   fprintf(stderr, "Unable to open \"%s\" for checkpointing timer information: %s\n", timercsvpath, strerror(errno));
      //   exit(EXIT_FAILURE);
      //}
      //writeCSV(timercsvstream->fp);

      PV_fclose(timerstream); timerstream = NULL;
      //PV_fclose(timercsvstream); timercsvstream = NULL;
   }

   // write adaptive time step info if dtAdaptFlag == true
   if( columnId()==0 && dtAdaptFlag == true) {
      char timescalepath[PV_PATH_MAX];
      int chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/timescaleinfo.bin", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timescalefile = PV_fopen(timescalepath,"w", getVerifyWrites());
      assert(timescalefile);
      for(int b = 0; b < nbatch; b++){
         if (PV_fwrite(&timeScale[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            fprintf(stderr, "HyPerCol::checkpointWrite error writing timeScale to %s\n", timescalefile->name);
            exit(EXIT_FAILURE);
         }
         if (PV_fwrite(&timeScaleTrue[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            fprintf(stderr, "HyPerCol::checkpointWrite error writing timeScaleTrue to %s\n", timescalefile->name);
            exit(EXIT_FAILURE);
         }
         if (useAdaptMethodExp1stOrder) {
            if (PV_fwrite(&timeScaleMax[b],1,sizeof(double),timescalefile) != sizeof(double)) {
               fprintf(stderr, "HyPerCol::checkpointWrite error writing timeScaleMax to %s\n", timescalefile->name);
               exit(EXIT_FAILURE);
            }
         }
         if (useAdaptMethodExp1stOrder) {
            if (PV_fwrite(&timeScaleMax2[b],1,sizeof(double),timescalefile) != sizeof(double)) {
               fprintf(stderr, "HyPerCol::checkpointWrite error writing timeScaleMax2 to %s\n", timescalefile->name);
               exit(EXIT_FAILURE);
            }
         }
         if (PV_fwrite(&deltaTimeAdapt[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            fprintf(stderr, "HyPerCol::checkpointWrite error writing deltaTimeAdapt to %s\n", timescalefile->name);
            exit(EXIT_FAILURE);
         }
      }
      PV_fclose(timescalefile);
      chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/timescaleinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timescalefile = PV_fopen(timescalepath,"w", getVerifyWrites());
      assert(timescalefile);
      int kb0 = commBatch() * nbatch;
      for(int b = 0; b < nbatch; b++){
         fprintf(timescalefile->fp,"batch = %d\n", b+kb0);
         fprintf(timescalefile->fp,"time = %g\n", timeScale[b]);
         fprintf(timescalefile->fp,"timeScaleTrue = %g\n", timeScaleTrue[b]);
      }
      PV_fclose(timescalefile);
   }

   std::string checkpointedParamsFile = cpDir;
   checkpointedParamsFile += "/";
   checkpointedParamsFile += "pv.params";
   this->outputParams(checkpointedParamsFile.c_str());

   if (checkpointWriteFlag) {
      char nextCheckpointPath[PV_PATH_MAX];
      int chars_needed;
      PV_Stream * nextCheckpointFile = NULL;
      switch(checkpointWriteTriggerMode) {
      case CPWRITE_TRIGGER_STEP:
         writeScalarToFile(cpDir, name, "nextCheckpointStep", nextCPWriteStep);
         break;
      case CPWRITE_TRIGGER_TIME:
         writeScalarToFile(cpDir, name, "nextCheckpointTime", nextCPWriteTime);
         break;
      case CPWRITE_TRIGGER_CLOCK:
         // Nothing to do in this case
         break;
      default:
         // All cases of checkpointWriteTriggerMode are handled above
         assert(0);
      }
   }


   // Note: timeinfo should be done at the end of the checkpointing, so that its presence serves as a flag that the checkpoint has completed.
   if( columnId()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timestampfile = PV_fopen(timestamppath,"w", getVerifyWrites());
      assert(timestampfile);
      PV_fwrite(&simTime,1,sizeof(double),timestampfile);
      PV_fwrite(&currentStep,1,sizeof(long int),timestampfile);
      PV_fclose(timestampfile);
      chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timestampfile = PV_fopen(timestamppath,"w", getVerifyWrites());
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
            int rmrf_result = system(rmrf_string);
            if (rmrf_result != 0) {
               fprintf(stderr, "Warning: unable to delete older checkpoint \"%s\": rm command returned %d\n",
                     lastCheckpointDir, rmrf_result);
            }
         }
      }
      int chars_needed = snprintf(lastCheckpointDir, PV_PATH_MAX, "%s", cpDir);
      assert(chars_needed < PV_PATH_MAX);
   }


   if (icComm->commRank()==0) {
      fprintf(stderr, "checkpointWrite complete. simTime = %f\n", simTime);
   }
   checkpointTimer->stop();
   return PV_SUCCESS;
}

int HyPerCol::outputParams(char const * path) {
   assert(path!=NULL && path[0]!='\0');
   int status = PV_SUCCESS;
   int rank=icComm->commRank();
   assert(printParamsStream==NULL);
   char printParamsPath[PV_PATH_MAX];
   char * tmp = strdup(path); // duplicate string since dirname() is allowed to modify its argument
   if (tmp==NULL) {
      fflush(stdout);
      fprintf(stderr, "HyPerCol::outputParams unable to allocate memory: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   char * containingdir = dirname(tmp);
   status = ensureDirExists(containingdir); // must be called by all processes, even though only rank 0 creates the directory
   if (status != PV_SUCCESS) {
      fflush(stdout);
      fprintf(stderr, "HyPerCol::outputParams unable to create directory \"%s\"\n", containingdir);
   }
   free(tmp);
   if(rank == 0){
      if( strlen(path)+4/*allow room for .lua at end, and string terminator*/ > (size_t) PV_PATH_MAX ) {
         fprintf(stderr, "outputParams called with too long a filename.  Parameters will not be printed.\n");
         status = ENAMETOOLONG;
      }
      else {
         printParamsStream = PV_fopen(path, "w", getVerifyWrites());
         if( printParamsStream == NULL ) {
            status = errno;
            fprintf(stderr, "outputParams error opening \"%s\" for writing: %s\n", path, strerror(errno));
         }
         //Get new lua path
         char luapath [PV_PATH_MAX];
         strcpy(luapath, path);
         strcat(luapath, ".lua");
         luaPrintParamsStream = PV_fopen(luapath, "w", getVerifyWrites());
         if( luaPrintParamsStream == NULL ) {
            status = errno;
            fprintf(stderr, "outputParams error opening \"%s\" for writing: %s\n", luapath, strerror(errno));
         }
      }
      assert(printParamsStream != NULL);
      assert(luaPrintParamsStream != NULL);

      //Params file output
      outputParamsHeadComments(printParamsStream->fp, "//");

      //Lua file output
      outputParamsHeadComments(luaPrintParamsStream->fp, "--");
      //Load util module based on PVPath
      fprintf(luaPrintParamsStream->fp, "package.path = package.path .. \";\" .. \"" PV_DIR "/parameterWrapper/?.lua\"\n");
      fprintf(luaPrintParamsStream->fp, "local pv = require \"PVModule\"\n\n");
      fprintf(luaPrintParamsStream->fp, "-- Base table variable to store\n"); 
      fprintf(luaPrintParamsStream->fp, "local pvParameters = {\n"); 
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

   // BaseConnection params
   for (int c=0; c<numConnections; c++) {
      BaseConnection * connection = connections[c];
      status = connection->ioParams(PARAMS_IO_WRITE);
      if( status != PV_SUCCESS ) {
         fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", printParamsPath);
         exit(EXIT_FAILURE);
      }
   }

   // Probe params

   // ColProbes
   for (int p=0; p<numColProbes; p++) {
      colProbes[p]->ioParams(PARAMS_IO_WRITE);
   }

   // LayerProbes
   for (int l=0; l<numLayers; l++) {
      layers[l]->outputProbeParams();
   }

   // BaseConnectionProbes
   for (int c=0; c<numConnections; c++) {
      connections[c]->outputProbeParams();
   }

   if(rank == 0){
      fprintf(luaPrintParamsStream->fp, "} --End of pvParameters\n"); 
      fprintf(luaPrintParamsStream->fp, "\n-- Print out PetaVision approved parameter file to the console\n"); 
      fprintf(luaPrintParamsStream->fp, "pv.printConsole(pvParameters)\n"); 
   }

   if (printParamsStream) {
      PV_fclose(printParamsStream);
      printParamsStream = NULL;
   }
   if (luaPrintParamsStream) {
      PV_fclose(luaPrintParamsStream);
      luaPrintParamsStream = NULL;
   }
   return status;
}

int HyPerCol::outputParamsHeadComments(FILE* fp, char const * commentToken) {
   time_t t = time(NULL);
   fprintf(fp, "%s PetaVision, " PV_REVISION "\n", commentToken);
   fprintf(fp, "%s Run time %s", commentToken, ctime(&t)); // newline is included in output of ctime
#ifdef PV_USE_MPI
   fprintf(fp, "%s Compiled with MPI and run using %d rows and %d columns.\n", commentToken, icComm->numCommRows(), icComm->numCommColumns());
#else // PV_USE_MPI
   fprintf(fp, "%s Compiled without MPI.\n", commentToken);
#endif // PV_USE_MPI
#ifdef PV_USE_OPENCL
   fprintf(fp, "%s Compiled with OpenCL.\n", commentToken);
#else
   fprintf(fp, "%s Compiled without OpenCL.\n", commentToken);
#endif // PV_USE_OPENCL
#ifdef PV_USE_CUDA
   fprintf(fp, "%s Compiled with CUDA.\n", commentToken);
#else
   fprintf(fp, "%s Compiled without CUDA.\n", commentToken);
#endif
#ifdef PV_USE_OPENMP_THREADS
   fprintf(fp, "%s Compiled with OpenMP parallel code", commentToken);
   if (numThreads>0) { fprintf(fp, " and run using %d threads.\n", numThreads); }
   else if (numThreads==0) { fprintf(fp, " but number of threads was set to zero (error).\n"); }
   else { fprintf(fp, " but the -t option was not specified.\n"); }
#else
   fprintf(fp, "%s Compiled without OpenMP parallel code", commentToken);
   if (numThreads==1) { fprintf(fp, ".\n"); }
   else if (numThreads==0) { fprintf(fp, " but number of threads was set to zero (error).\n"); }
   else { fprintf(fp, " but number of threads specified was %d instead of 1. (error).\n", numThreads); }
#endif // PV_USE_OPENMP_THREADS
   if (checkpointReadFlag) {
      fprintf(fp, "%s Started from checkpoint \"%s\"\n", commentToken, checkpointReadDir);
   }
   return PV_SUCCESS;
}

// Uses the arguments cpDir, objectName, and suffix to create a path of the form
// [cpDir]/[objectName][suffix]
// (the brackets are not in the created path, but the slash is)
// The string returned is allocated with malloc, and the calling routine is responsible for freeing the string.
char * HyPerCol::pathInCheckpoint(const char * cpDir, const char * objectName, const char * suffix) {
   assert(cpDir!=NULL && suffix!=NULL);
   size_t n = strlen(cpDir)+strlen("/")+strlen(objectName)+strlen(suffix)+(size_t) 1; // the +1 leaves room for the terminating null
   char * filename = (char *) malloc(n);
   if (filename==NULL) {
      fprintf(stderr, "Error: rank %d process unable to allocate filename \"%s/%s%s\": %s\n", columnId(), cpDir, objectName, suffix, strerror(errno));
      exit(EXIT_FAILURE);
   }
   int chars_needed = snprintf(filename, n, "%s/%s%s", cpDir, objectName, suffix);
   assert(chars_needed < n);
   return filename;
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
            exit(EXIT_FAILURE);
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

int HyPerCol::getAutoGPUDevice(){
   int returnGpuIdx = -1;
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   int mpiRank = icComm->globalCommRank();
   int numMpi = icComm->globalCommSize();
   char hostNameStr[PV_PATH_MAX];
   gethostname(hostNameStr, PV_PATH_MAX);
   size_t hostNameLen = strlen(hostNameStr) + 1; //+1 for null terminator

   //Each rank communicates which host it is on
   //Root process
   if(mpiRank == 0){
      //Allocate data structure for rank to host
      char rankToHost[numMpi][PV_PATH_MAX];
      assert(rankToHost);
      //Allocate data structure for rank to maxGpu
      int rankToMaxGpu[numMpi];
      //Allocate final data structure for rank to GPU index
      int rankToGpu[numMpi];
      assert(rankToGpu);

      for(int rank = 0; rank < numMpi; rank++){
         if(rank == 0){
            strcpy(rankToHost[rank], hostNameStr);
            rankToMaxGpu[rank] = PVCuda::CudaDevice::getNumDevices();
         }
         else{
            MPI_Recv(rankToHost[rank], PV_PATH_MAX, MPI_CHAR, rank, 0, icComm->globalCommunicator(), MPI_STATUS_IGNORE);
            MPI_Recv(&(rankToMaxGpu[rank]), 1, MPI_INT, rank, 0, icComm->globalCommunicator(), MPI_STATUS_IGNORE);
         }
      }

      //rankToHost now is an array such that the index is the rank, and the value is the host
      //Convert to a map of vectors, such that the key is the host name and the value
      //is a vector of mpi ranks that is running on that host
      std::map<std::string, std::vector<int> > hostMap;
      for(int rank = 0; rank < numMpi; rank++){
         hostMap[std::string(rankToHost[rank])].push_back(rank);
      }

      //Determine what gpus to use per mpi
      for(std::map<std::string, std::vector<int> >::const_iterator m_it = hostMap.begin();
            m_it != hostMap.end(); ++m_it){
         std::vector<int> rankVec = m_it->second;
         int numRanksPerHost = rankVec.size();
         assert(numRanksPerHost > 0);
         //Grab maxGpus of current host
         int maxGpus = rankToMaxGpu[rankVec[0]];
         //Warnings for overloading/underloading gpus
         if(numRanksPerHost != maxGpus){
            fprintf(stderr, "HyPerCol:: Warning! Host \"%s\" (rank[s] ", m_it->first.c_str());
            for(int v_i = 0; v_i < numRanksPerHost; v_i++){
               if(v_i != numRanksPerHost-1){
                  fprintf(stderr, "%d, ", rankVec[v_i]);
               }
               else{
                  fprintf(stderr, "%d", rankVec[v_i]);
               }
            }
            fprintf(stderr, ") is being %s, with %d mpi processes mapped to %d total GPU[s]\n",
                  numRanksPerHost < maxGpus ? "underloaded":"overloaded",
                  numRanksPerHost, maxGpus);
         }

         //Match a rank to a gpu
         for(int v_i = 0; v_i < numRanksPerHost; v_i++){
            rankToGpu[rankVec[v_i]] = v_i % maxGpus;
         }
      }

      //MPI sends to each process to specify which gpu the rank should use
      for(int rank = 0; rank < numMpi; rank++){
         std::cout << "Rank " << rank << " on host \"" << rankToHost[rank] << "\" (" << rankToMaxGpu[rank] << " GPU[s]) using GPU index " <<
            rankToGpu[rank] << "\n";
         if(rank == 0){
            returnGpuIdx = rankToGpu[rank];
         }
         else{
            MPI_Send(&(rankToGpu[rank]), 1, MPI_INT, rank, 0, icComm->globalCommunicator());
         }
      }
   }
   //Non root process
   else{
      //Send host name
      MPI_Send(hostNameStr, hostNameLen, MPI_CHAR, 0, 0, icComm->globalCommunicator());
      //Send max gpus for that host
      int maxGpu = PVCuda::CudaDevice::getNumDevices();
      MPI_Send(&maxGpu, 1, MPI_INT, 0, 0, icComm->globalCommunicator());
      //Recv gpu idx
      MPI_Recv(&(returnGpuIdx), 1, MPI_INT, 0, 0, icComm->globalCommunicator(), MPI_STATUS_IGNORE);
   }
   assert(returnGpuIdx >= 0 && returnGpuIdx < PVCuda::CudaDevice::getNumDevices());
#else
   //This function should never be called when not running with GPUs
   assert(false);
#endif
   return returnGpuIdx;
}

int HyPerCol::initializeThreads(char const * in_device)
{
   int numMpi = icComm->globalCommSize();
   int device;

   //default value
   if(in_device == NULL){
      std::cout << "Auto assigning GPUs\n";
      device = getAutoGPUDevice();
   }
   else{
      std::vector <int> deviceVec;
      std::stringstream ss(in_device);
      std::string stoken;
      //Grabs strings from ss into item, seperated by commas
      while(std::getline(ss, stoken, ',')){
         //Convert stoken to integer
         for(std::string::const_iterator k = stoken.begin(); k != stoken.end(); ++k){
            if(!isdigit(*k)){
               fprintf(stderr, "Device specification error: %s contains unrecognized characters. Must be comma seperated integers greater or equal to 0 with no other characters allowed (including spaces).\n", in_device);
               exit(EXIT_FAILURE);
            }
         }
         deviceVec.push_back(atoi(stoken.c_str()));
      }
      //Check length of deviceVec
      //Allowed cases are 1 device specified or greater than or equal to number of mpi processes devices specified
      if(deviceVec.size() == 1){
         device = deviceVec[0];
      }
      else if(deviceVec.size() >= numMpi){
         device = deviceVec[icComm->globalCommRank()];
      }
      else{
         fprintf(stderr, "Device specification error: Number of devices specified (%zu) must be either 1 or >= than number of mpi processes (%d).\n", deviceVec.size(), numMpi);
         exit(EXIT_FAILURE);
      }
      std::cout << "Global MPI Process " << icComm->globalCommRank() << " using device " << device << "\n";
   }

#ifdef PV_USE_OPENCL
   clDevice = new CLDevice(device);
#endif
#ifdef PV_USE_CUDA
   cudaDevice = new PVCuda::CudaDevice(device);
#endif
   return 0;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
int HyPerCol::finalizeThreads()
{
#ifdef PV_USE_OPENCL
   delete clDevice;
#endif
#ifdef PV_USE_CUDA
   delete cudaDevice;
   if(gpuGroupConns){
      free(gpuGroupConns);
   }
#endif
   return 0;
}
#endif // PV_USE_OPENCL


#ifdef PV_USE_CUDA
void HyPerCol::addGpuGroup(BaseConnection* conn, int gpuGroupIdx){
   //default gpuGroupIdx is -1, so do nothing if this is the case
   if(gpuGroupIdx < 0){
      return;
   }
   //Resize buffer if not big enough
   if(gpuGroupIdx >= numGpuGroup){
      int oldNumGpuGroup = numGpuGroup;
      numGpuGroup = gpuGroupIdx + 1;
      gpuGroupConns = (BaseConnection**) realloc(gpuGroupConns, numGpuGroup * sizeof(BaseConnection*)); 
      //Initialize newly allocated part to NULL
      for(int i = oldNumGpuGroup; i < numGpuGroup; i++){
         gpuGroupConns[i] = NULL;
      }
   }
   //If empty, fill
   if(gpuGroupConns[gpuGroupIdx] == NULL){
      gpuGroupConns[gpuGroupIdx] = conn;
   }
   //Otherwise, do nothing
   
   ////Base connection does not have patch size info. This check will be done in HyPerConn
   //else{
   //   //Check connection sizes to make sure they're the same
   //   BaseConnection* origConn = gpuGroupConns[gpuGroupIdx];
   //   //Different num arbors is okay, since GPU mem holds only one arbor at a time
   //   //nxp, nyp, nfp, numKernels all have to be the same
   //   if(origConn->xPatchSize() != conn->xPatchSize() ||
   //      origConn->yPatchSize() != conn->yPatchSize() ||
   //      origConn->fPatchSize() != conn->fPatchSize() ||
   //      origConn->getNumDataPatches() != conn->getNumDataPatches()){
   //         std::cout << "Connection " << conn->getName() << " of size (" <<
   //         conn->getNumDataPatches() << ", " << 
   //         conn->xPatchSize() << ", " <<
   //         conn->yPatchSize() << ", " <<
   //         conn->fPatchSize() << 
   //         ") does not match the gpuGroupConnection " << 
   //         origConn->getName() << " of size (" <<
   //         origConn->getNumDataPatches() << ", " << 
   //         origConn->xPatchSize() << ", " <<
   //         origConn->yPatchSize() << ", " <<
   //         origConn->fPatchSize() << ").\n";
   //         exit(-1);
   //   }
   //}
   return;
}

BaseConnection* HyPerCol::getGpuGroupConn(int gpuGroupIdx){
   return gpuGroupConns[gpuGroupIdx];
}
#endif



int HyPerCol::loadState()
{
   return 0;
}

int HyPerCol::insertProbe(ColProbe * p)
{
   ColProbe ** newprobes;
   newprobes = (ColProbe **) malloc( ((size_t) (numColProbes + 1)) * sizeof(ColProbe *) );
   assert(newprobes != NULL);

   for (int i = 0; i < numColProbes; i++) {
      newprobes[i] = colProbes[i];
   }
   free(colProbes);

   colProbes = newprobes;
   colProbes[numColProbes] = p;
   return ++numColProbes;
}

// BaseProbes include layer probes, connection probes, and column probes.
int HyPerCol::addBaseProbe(BaseProbe * p) {
   BaseProbe ** newprobes;
   // Instead of mallocing a new buffer and freeing the old buffer, this could be a realloc.
   newprobes = (BaseProbe **) malloc( ((size_t) (numBaseProbes + 1)) * sizeof(BaseProbe *) );
   assert(newprobes != NULL);

   for (int i=0; i<numBaseProbes; i++) {
      newprobes[i] = baseProbes[i];
   }
   free(baseProbes);
   baseProbes = newprobes;
   baseProbes[numBaseProbes] = p;

   return ++numBaseProbes;
}

int HyPerCol::outputState(double time)
{
   for( int n = 0; n < numColProbes; n++ ) {
       colProbes[n]->outputStateWrapper(time, deltaTimeBase);
   }
   return PV_SUCCESS;
}


HyPerLayer * HyPerCol::getLayerFromName(const char * layerName) {
   if (layerName==NULL) { return NULL; }
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

BaseConnection * HyPerCol::getConnFromName(const char * connName) {
   if( connName == NULL ) return NULL;
   int n = numberOfConnections();
   for( int i=0; i<n; i++ ) {
      BaseConnection * curConn = getConnection(i);
      assert(curConn);
      const char * curConnName = curConn->getName();
      assert(curConnName);
      if( !strcmp( curConn->getName(), connName) ) return curConn;
   }
   return NULL;
}

NormalizeBase * HyPerCol::getNormalizerFromName(const char * normalizerName) {
   if( normalizerName == NULL ) return NULL;
   int n = numberOfNormalizers();
   for( int i=0; i<n; i++ ) {
      NormalizeBase * curNormalizer = getNormalizer(i);
      assert(curNormalizer);
      const char * curNormalizerName = curNormalizer->getName();
      assert(curNormalizerName);
      if( !strcmp(curNormalizer->getName(), normalizerName) ) return curNormalizer;
   }
   return NULL;
}

ColProbe * HyPerCol::getColProbeFromName(const char * probeName) {
   if (probeName == NULL) return NULL;
   ColProbe * p = NULL;
   int n = numberOfProbes();
   for (int i=0; i<n; i++) {
      ColProbe * curColProbe = getColProbe(i);
      const char * curName = curColProbe->getName();
      assert(curName);
      if (!strcmp(curName, probeName)) {
         p = curColProbe;
         break;
      }
   }
   return p;
}

BaseProbe * HyPerCol::getBaseProbeFromName(const char * probeName) {
   if (probeName == NULL) { return NULL; }
   BaseProbe * p = NULL;
   int n = numberOfBaseProbes();
   for (int i=0; i<n; i++) {
      BaseProbe * curBaseProbe = getBaseProbe(i);
      const char * curName = curBaseProbe->getName();
      assert(curName);
      if (!strcmp(curName, probeName)) {
         p = curBaseProbe;
         break;
      }
   }
   return p;
}

unsigned int HyPerCol::getRandomSeed() {
   unsigned long t = 0UL;
   int rootproc = 0;
   if (columnId()==rootproc) {
       t = time((time_t *) NULL);
   }
   MPI_Bcast(&t, 1, MPI_UNSIGNED, rootproc, icComm->communicator());
   return t;
}

template <typename T>
int HyPerCol::writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val) {
   return writeArrayToFile(cp_dir, group_name, val_name, &val, 1);
}

// Declare the instantiations of writeScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeScalarToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int val);
template int HyPerCol::writeScalarToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long val);
template int HyPerCol::writeScalarToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float val);
template int HyPerCol::writeScalarToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double val);

template <typename T>
int HyPerCol::writeArrayToFile(const char * cp_dir, const char * group_name, const char * val_name, T* val, size_t count) {
   int status = PV_SUCCESS;
   if (columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "writeArrayToFile error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         exit(EXIT_FAILURE);
      }
      PV_Stream * pvstream = PV_fopen(filename, "w", getVerifyWrites());
      if (pvstream==NULL) {
         fprintf(stderr, "writeArrayToFile error: unable to open path %s for writing.\n", filename);
         exit(EXIT_FAILURE);
      }
      int num_written = PV_fwrite(val, sizeof(T), count, pvstream);
      if (num_written != count) {
         fprintf(stderr, "writeArrayToFile error while writing to %s.\n", filename);
         exit(EXIT_FAILURE);
      }
      PV_fclose(pvstream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, group_name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      std::ofstream fs;
      fs.open(filename);
      if (!fs) {
         fprintf(stderr, "writeArrayToFile error: unable to open path %s for writing.\n", filename);
         exit(EXIT_FAILURE);
      }
      for(int i = 0; i < count; i++){
         fs << val[i];
         fs << std::endl; // Can write as fs << val << std::endl, but eclipse flags that as an error 'Invalid overload of std::endl'
      }
      fs.close();
   }
   return status;
}
// Declare the instantiations of writeArrayToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeArrayToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int* val, size_t count);
template int HyPerCol::writeArrayToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long* val, size_t count);
template int HyPerCol::writeArrayToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float* val, size_t count);
template int HyPerCol::writeArrayToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double* val, size_t count);

template <typename T>
int HyPerCol::readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value) {
   return readArrayFromFile(cp_dir, group_name, val_name, val, 1, default_value);
}

// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readScalarFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, int default_value);
template int HyPerCol::readScalarFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, long default_value);
template int HyPerCol::readScalarFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, float default_value);
template int HyPerCol::readScalarFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, double default_value);

template <typename T>
int HyPerCol::readArrayFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, size_t count, T default_value) {
   int status = PV_SUCCESS;
   if( columnId() == 0 ) {
      char filename[PV_PATH_MAX];
      int chars_needed;
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name); // Could use pathInCheckpoint if not for the .bin
      if(chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerLayer::readArrayFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         exit(EXIT_FAILURE);
      }
      PV_Stream * pvstream = PV_fopen(filename, "r", getVerifyWrites());
      for(int i = 0; i < count; i++){
         val[i] = default_value;
      }
      if (pvstream==NULL) {
         std::cerr << "readArrayFromFile warning: unable to open path \"" << filename << "\" for reading.  Value used will be " << *val;
         std::cerr << std::endl;
         // fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to open path %s for reading.  value used will be %f\n", filename, default_value);
      }
      else {
         int num_read = PV_fread(val, sizeof(T), count, pvstream);
         if (num_read != count) {
            std::cerr << "readArrayFromFile warning: unable to read from \"" << filename << "\".  Value used will be " << *val;
            std::cerr << std::endl;
            // fprintf(stderr, "HyPerLayer::readArrayFloat warning: unable to read from %s.  value used will be %f\n", filename, default_value);
         }
         PV_fclose(pvstream);
      }
   }
   MPI_Bcast(val, sizeof(T)*count, MPI_CHAR, 0, icCommunicator()->communicator());

   return status;
}
// Declare the instantiations of readArrayToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readArrayFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, size_t count, int default_value);
template int HyPerCol::readArrayFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, size_t count, long default_value);
template int HyPerCol::readArrayFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, size_t count, float default_value);
template int HyPerCol::readArrayFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, size_t count, double default_value);

} // PV namespace

