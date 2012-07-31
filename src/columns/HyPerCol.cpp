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
#include "../utils/pv_random.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>

#define PV_MAX_NUMSTEPS (pow(2,FLT_MANT_DIG))
#define HYPERCOL_DIRINDEX_MAX 99999999

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[])
         : warmStart(false), isInitialized(false)
{
   initialize(name, argc, argv);
}

#ifdef OBSOLETE // Marked obsolete Jul 29, 2011.  Use -w option in argc and argv instead of pv_path.
HyPerCol::HyPerCol(const char * name, int argc, char * argv[], const char * pv_path)
         : warmStart(false), isInitialized(false)
{
   initialize(name, argc, argv);
   assert(strlen(path) + strlen(pv_path) < PV_PATH_MAX);
   sprintf(path, "%s/%s", path, pv_path);
}
#endif // OBSOLETE

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
      // TODO: check to see if finalize called
      if (layers[n] != NULL) {
         delete layers[n]; // will call *_finalize
      }
      else {
         // TODO move finalize
         // PVLayer_finalize(getCLayer(n));
      }
   }

   delete params;

   delete icComm;

   printf("%32s: total time in %6s %10s: ", name, "column", "run    ");
   runTimer->elapsed_time();
   fflush(stdout);
   delete runTimer;

   free(connections);
   free(layers);
   free(name);
   free(path);
   free(outputPath);
   free(outputNamesOfLayersAndConns);
}

int HyPerCol::initFinish(void)
{
   int status = 0;

   for (int i = 0; i < this->numLayers; i++) {
      status = layers[i]->initFinish();
      if (status != 0) {
         fprintf(stderr, "[%d]: HyPerCol::initFinish: ERROR condition, exiting...\n", this->columnId());
         exit(status);
      }
   }

#ifdef OBSOLETE
   // TODO - fix this to modern version?
   log_parameters(numSteps, image_file);
#endif

   isInitialized = true;

   return status;
}

#define NUMSTEPS 1
int HyPerCol::initialize(const char * name, int argc, char ** argv)
{
   icComm = new InterColComm(&argc, &argv);
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
      MPI_Barrier(icComm->communicator());
   }
#endif
;
   layerArraySize = INITIAL_LAYER_ARRAY_SIZE;
   connectionArraySize = INITIAL_CONNECTION_ARRAY_SIZE;

   this->name = strdup(name);
   this->runTimer = new Timer();

   char * param_file;
   char * working_dir;
   simTime = 0;
   currentStep = 0;
   numLayers = 0;
   numConnections = 0;
   layers = (HyPerLayer **) malloc(layerArraySize * sizeof(HyPerLayer *));
   connections = (HyPerConn **) malloc(connectionArraySize * sizeof(HyPerConn *));

   int opencl_device = 0;  // default to GPU for now
   numSteps = 0; // numSteps = 2;
   outputPath = NULL;
   image_file = NULL;
   param_file = NULL;
   working_dir = NULL;
   unsigned long random_seed = 0;
   deleteOlderCheckpoints = false;
   parse_options(argc, argv, &outputPath, &param_file,
                 &numSteps, &opencl_device, &random_seed, &working_dir);

   if(working_dir) {
      int status = chdir(working_dir);
      if(status) {
         fprintf(stderr, "Unable to switch directory to \"%s\"\n", working_dir);
         fprintf(stderr, "chdir returned error %d\n", errno);
         exit(status);
      }
   }

   path = (char *) malloc(1+PV_PATH_MAX);
   assert(path != NULL);
   path = getcwd(path, PV_PATH_MAX);

   int groupArraySize = 2*(layerArraySize + connectionArraySize);
   params = new PVParams(param_file, groupArraySize, this);  // PVParams::addGroup can resize if initialGroups is exceeded
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

   // set number of steps from params file if it wasn't set on the command line
   if( !numSteps ) {
      if( params->present(name, "numSteps") ) {
         numSteps = params->value(name, "numSteps");
      }
      else {
         numSteps = NUMSTEPS;
         printf("Number of steps specified neither in the command line nor the params file.\n"
                "Number of steps set to default %d\n",NUMSTEPS);
      }
   }
   if( (double) numSteps > PV_MAX_NUMSTEPS ) {
      fprintf(stderr, "The number of time steps %d is greater than %ld, the maximum allowed by floating point precision\n", numSteps, (long int) PV_MAX_NUMSTEPS);
      exit(EXIT_FAILURE);
   }

   // set how often advanceTime() prints a message indicating progress
   progressStep = params->value(name, "progressStep", 2000, true);

   // set output path from params file if it wasn't set on the command line
   if (outputPath == NULL ) {
      if( params->stringPresent(name, "outputPath") ) {
         outputPath = strdup(params->stringValue(name, "outputPath"));
         assert(outputPath != NULL);
      }
      else {
         outputPath = strdup(OUTPUT_PATH);
         assert(outputPath != NULL);
         printf("Output path specified neither in command line nor in params file.\n"
                "Output path set to default \"%s\"\n",OUTPUT_PATH);
      }
   }
   ensureDirExists(outputPath);

   const char * printParamsFilename = params->stringValue(name, "printParamsFilename", false);
   if( printParamsFilename != NULL ) {
      outputParams(printParamsFilename);
   }

   // run only on GPU for now
#ifdef PV_USE_OPENCL
   initializeThreads(opencl_device);
   clDevice->query_device_info();
#endif

   // set random seed if it wasn't set in the command line
   if( !random_seed ) {
      if( params->present(name, "randomSeed") ) {
         random_seed = params->value(name, "randomSeed");
      }
      else {
         random_seed = getRandomSeed();
         printf("Using time to get random seed. Seed set to %lu\n", random_seed);
      }
   }
   pv_srandom(random_seed); // initialize random seed

   nxGlobal = (int) params->value(name, "nx");
   nyGlobal = (int) params->value(name, "ny");

   deltaTime = DELTA_T;
   deltaTime = params->value(name, "dt", deltaTime, true);

   runDelegate = NULL;

   numProbes = 0;
   probes = NULL;

   filenamesContainLayerNames = params->value(name, "filenamesContainLayerNames", 0);
   if(filenamesContainLayerNames < 0 || filenamesContainLayerNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainLayerNames must have the value 0, 1, or 2.\n", name);
      abort();
   }

   const char * lcfilename = params->stringValue(name, "outputNamesOfLayersAndConns", false);
   if( lcfilename != NULL && lcfilename[0] != 0 && rank==0 ) {
      outputNamesOfLayersAndConns = (char *) malloc( (strlen(outputPath)+strlen(lcfilename)+2)*sizeof(char) );
      if( !outputNamesOfLayersAndConns ) {
         fprintf(stderr, "HyPerCol \"%s\": Unable to allocate memory for outputNamesOfLayersAndConns.  Exiting.\n", name);
         exit(EXIT_FAILURE);
      }
      sprintf(outputNamesOfLayersAndConns, "%s/%s", outputPath, lcfilename);
   }
   else {
      outputNamesOfLayersAndConns = NULL;
   }

   checkpointReadFlag = params->value(name, "checkpointRead", false) != 0;
   if(checkpointReadFlag) {
      const char * cpreaddir = params->stringValue(name, "checkpointReadDir", true);
      if( cpreaddir != NULL ) {
         checkpointReadDir = strdup(cpreaddir);
      }
      else {
         if( rank == 0 ) {
            fprintf(stderr, "Column \"%s\": if checkpointRead is set, the string checkpointReadDir must be defined.  Exiting.\n", name);
         }
         exit(EXIT_FAILURE);
      }
      struct stat checkpointReadDirStat;
      int dirExistStatus = checkDirExists(checkpointReadDir, &checkpointReadDirStat);
      if( dirExistStatus != 0 ) {
         if( rank == 0 ) {
            fprintf(stderr, "Column \"%s\": unable to read checkpointReadDir \"%s\".  Error %d\n", name, checkpointReadDir, dirExistStatus);
         }
         exit(EXIT_FAILURE);
      }
      cpReadDirIndex = (int) params->value(name, "checkpointReadDirIndex", -1, true);
      if( cpReadDirIndex < 0 || cpReadDirIndex > HYPERCOL_DIRINDEX_MAX ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr, "Column \"%s\": checkpointReadDirIndex must be between 0 and %d, inclusive.  Exiting.\n", name, HYPERCOL_DIRINDEX_MAX);
         }
         exit(EXIT_FAILURE);
      }
   }

   checkpointWriteFlag = params->value(name, "checkpointWrite", false) != 0;
   if(checkpointWriteFlag) {
      const char * cpwritedir = params->stringValue(name, "checkpointWriteDir", true);
      if( cpwritedir != NULL ) {
         checkpointWriteDir = strdup(cpwritedir);
      }
      else {
         if( rank == 0 ) {
            fprintf(stderr, "Column \"%s\": if checkpointWrite is set, the string checkpointWriteDir must be defined.  Exiting.\n", name);
         }
         exit(EXIT_FAILURE);
      }
      ensureDirExists(checkpointWriteDir);
      bool usingWriteStep = params->present(name, "checkpointWriteStepInterval") && params->value(name, "checkpointWriteStepInterval")>0;
      bool usingWriteTime = params->present(name, "checkpointWriteTimeInterval") && params->value(name, "checkpointWriteTimeInterval")>0;
      if( !usingWriteStep && !usingWriteTime ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr,"If checkpointWrite is set, one of checkpointWriteStepInterval or checkpointWriteTimeInterval must be positive.\n");
         }
         exit(EXIT_FAILURE);
      }
      if( usingWriteStep && usingWriteTime ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr,"If checkpointWrite is set, only one of checkpointWriteStepInterval or checkpointWriteTimeInterval can be positive.\n");
         }
         exit(EXIT_FAILURE);
      }
      if( usingWriteStep ) {
         cpWriteStepInterval = (int) params->value(name, "checkpointWriteStepInterval");
         cpWriteTimeInterval = -1;
      }
      if( usingWriteTime ) {
         cpWriteTimeInterval = params->value(name, "checkpointWriteTimeInterval");
         cpWriteStepInterval = -1;
      }
      nextCPWriteStep = 0;
      nextCPWriteTime = 0;

      deleteOlderCheckpoints = params->value(name, "deleteOlderCheckpoints", false) != 0;
      if (deleteOlderCheckpoints) {
         memset(lastCheckpointDir, 0, PV_PATH_MAX);
      }
   }
   else {
      suppressLastOutput = params->value(name, "suppressLastOutput", false) != 0;
   }

   return PV_SUCCESS;
}

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
         int mkdirstatus = mkdir(dirname, 0700);
         if( mkdirstatus ) {
            fflush(stdout);
            fprintf(stderr, "Directory \"%s\" could not be created: error %d\n", dirname, errno);
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      if( rank == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Checking status of directory \"%s\" gave error %d\n", dirname, resultcode);
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
   assert(numLayers <= layerArraySize);

   // Check for duplicate layer names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numLayers; k++) {
   //    if( !strcmp(l->getName(), layers[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, l->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }

   if( numLayers ==  layerArraySize ) {
      layerArraySize += RESIZE_ARRAY_INCR;
      HyPerLayer ** newLayers = (HyPerLayer **) malloc( layerArraySize * sizeof(HyPerLayer *) );
      assert(newLayers);
      for(int k=0; k<numLayers; k++) {
         newLayers[k] = layers[k];
      }
      free(layers);
      layers = newLayers;
   }
   l->columnWillAddLayer(icComm, numLayers);
   layers[numLayers++] = l;
   return (numLayers - 1);
}

int HyPerCol::addConnection(HyPerConn * conn)
{
   int connId = numConnections;

   assert(numConnections <= connectionArraySize);
   // Check for duplicate connection names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numConnections; k++) {
   //    if( !strcmp(conn->getName(), connections[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, conn->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }
   if( numConnections == connectionArraySize ) {
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

int HyPerCol::run(int nTimeSteps)
{
   if( checkMarginWidths() != PV_SUCCESS ) {
      fprintf(stderr, "Margin width failure; unable to continue.\n");
      return PV_MARGINWIDTH_FAILURE;
   }

   if( outputNamesOfLayersAndConns ) {
      assert( icComm->commRank() == 0 );
      printf("Dumping layer and connection names to \"%s\"\n", outputNamesOfLayersAndConns);
      FILE * fpOutputNames = fopen(outputNamesOfLayersAndConns,"w");
      if( fpOutputNames == NULL ) {
         fprintf(stderr, "HyPerCol \"%s\" unable to open \"%s\" for writing: error %d.  Exiting.\n", name, outputNamesOfLayersAndConns, errno);
         exit(errno);
      }
      fprintf(fpOutputNames, "Layers and Connections in HyPerCol \"%s\"\n\n", name);
      for( int k=0; k<numLayers; k++ ) {
         fprintf(fpOutputNames, "    Layer % 4d: %s\n", k, layers[k]->getName());
      }
      fprintf(fpOutputNames, "\n");
      for( int k=0; k<numConnections; k++ ) {
         fprintf(fpOutputNames, "    Conn. % 4d: %s\n", k, connections[k]->getName());
      }
      int fcloseStatus = fclose(fpOutputNames);
      if( fcloseStatus != 0 ) {
         fprintf(stderr, "Warning: Attempting to close output file \"%s\" generated an error.\n", outputNamesOfLayersAndConns);
      }
      fpOutputNames = NULL;
   }

   stopTime = simTime + nTimeSteps * deltaTime;
   const bool exitOnFinish = false;

   if (!isInitialized) {
      initFinish();
   }

   initPublishers(); // create the publishers and their data stores

   numSteps = nTimeSteps;

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // Initialize either by loading from checkpoint, or calling initializeState
   // This needs to happen after initPublishers so that we can initialize the values in the data stores,
   // and before the layers' publish calls so that the data in border regions gets copied correctly.
   if ( checkpointReadFlag ) {
      int str_len = snprintf(NULL, 0, "%s/Checkpoint%d", checkpointReadDir, cpReadDirIndex);
      char * cpDir = (char *) malloc( (str_len+1)*sizeof(char) );
      snprintf(cpDir, str_len+1, "%s/Checkpoint%d", checkpointReadDir, cpReadDirIndex);
      checkpointRead(cpDir);
      if (checkpointWriteFlag && deleteOlderCheckpoints) {
         int chars_needed = snprintf(lastCheckpointDir, PV_PATH_MAX, "%s", cpDir);
         if (chars_needed >= PV_PATH_MAX) {
            if (icComm->commRank()==0) {
               fprintf(stderr, "checkpointRead error: path \"%s\" is too long.\n", cpDir);
            }
            abort();
         }
      }
   }
   else {
      for ( int l=0; l<numLayers; l++ ) {
         layers[l]->initializeState();
      }
   }

   parameters()->warnUnread();

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
   int step = 0;
   while (simTime < stopTime) {
      if( checkpointWriteFlag && advanceCPWriteTime() ) {
         if (icComm->commRank()==0) {
            fprintf(stderr, "Checkpointing, simTime = %f\n", simulationTime());
         }
         if( currentStep >= HYPERCOL_DIRINDEX_MAX+1 ) {
            if( icComm->commRank() == 0 ) {
               fflush(stdout);
               fprintf(stderr, "Column \"%s\": step number exceeds maximum value %d.  Exiting\n", name, HYPERCOL_DIRINDEX_MAX);
            }
            exit(EXIT_FAILURE);
         }
         char cpDir[PV_PATH_MAX];
         int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%d", checkpointWriteDir, currentStep);
         if(chars_printed >= PV_PATH_MAX) {
            if (icComm->commRank()==0) {
               fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%d\" is too long.\n", checkpointWriteDir, currentStep);
               abort();
            }
         }
         checkpointWrite(cpDir);
      }
      simTime = advanceTime(simTime);

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

int HyPerCol::initPublishers() {
   for( int l=0; l<numLayers; l++ ) {
      PVLayer * clayer = layers[l]->getCLayer();
      icComm->addPublisher(layers[l], clayer->activity->numItems, clayer->numDelayLevels);
   }
   for( int c=0; c<numConnections; c++ ) {
      icComm->subscribe(connections[c]);
   }

   return PV_SUCCESS;
}

float HyPerCol::advanceTime(float sim_time)
{
#ifdef TIMESTEP_OUTPUT
   if (currentStep%progressStep == 0 && columnId() == 0) {
      printf("   [%d]: time==%f\n", columnId(), sim_time);
   }
#endif

   runTimer->start();

   // At this point all activity from the previous time step has
   // been delivered to the data store.
   //

   // update the connections (weights)
   //
   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(sim_time);
   }
   for (int c = 0; c < numConnections; c++) {
      connections[c]->updateState(sim_time, deltaTime);
   }

   for (int l = 0; l < numLayers; l++) {
      // deliver new synaptic activity to any
      // postsynaptic layers for which this
      // layer is presynaptic.
      layers[l]->triggerReceive(icComm);
   }

   // Update the layers (activity)
   // In order for probing the GSyn channels to work,
   // this needs to be after all the triggerReceive
   // calls and before any of the updateState calls.
   // This is because triggerReceive updates the GSyn
   // buffers but updateState clears them.
   // However, this means that probes of V and A are
   // one step behind probes of GSyn.
   for (int l = 0; l < numLayers; l++) {
      layers[l]->outputState(sim_time);
   }
   
   for(int l = 0; l < numLayers; l++) {
      layers[l]->updateState(sim_time, deltaTime);
   }

   // This loop separate from the update layer loop above
   // to provide time for layer data to be copied from
   // the OpenCL device.
   //
   for (int l = 0; l < numLayers; l++) {
      // after updateBorder completes all necessary data has been
      // copied from the device (GPU) to the host (CPU)
      layers[l]->updateBorder(sim_time, deltaTime);

      // TODO - move this to layer
      // Advance time level so we have a new place in data store
      // to copy the data.  This should be done immediately before
      // publish so there is a place to publish and deliver the data to.
      // No one can access the data store (except to publish) until
      // wait has been called.  This should be fixed so that publish goes
      // to last time level and level is advanced only after wait.
      icComm->increaseTimeLevel(layers[l]->getLayerId());

      layers[l]->publish(icComm, sim_time);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->waitOnPublish(icComm);
   }

   // make sure simTime is updated even if HyPerCol isn't running time loop

   float outputTime = simTime; // so that outputState is called with the correct time
                               // but doesn't effect runTimer

   simTime = sim_time + deltaTime;
   currentStep++;

   runTimer->stop();

   outputState(outputTime);

   return simTime;
}

bool HyPerCol::advanceCPWriteTime() {
   // returns true if nextCPWrite{Step,Time} has been advanced
   bool advanceCPTime;
   if( cpWriteStepInterval>0 ) {
      assert(cpWriteTimeInterval<0.0f);
      advanceCPTime = currentStep >= nextCPWriteStep;
      if( advanceCPTime ) {
         nextCPWriteStep += cpWriteStepInterval;
      }
   }
   else if( cpWriteTimeInterval>0.0f ) {
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
   size_t bufsize = sizeof(int) + sizeof(float);
   unsigned char * buf = (unsigned char *) malloc(bufsize);
   assert(buf);
   if( icCommunicator()->commRank()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      FILE * timestampfile = fopen(timestamppath,"r");
      if (timestampfile == NULL) {
         fprintf(stderr, "HyPerCol::checkpointRead error: unable to open \"%s\" for reading.\n", timestamppath);
         abort();
      }
      fread(buf,1,bufsize,timestampfile);
      fclose(timestampfile);
   }
#ifdef PV_USE_MPI
   MPI_Bcast(buf,bufsize,MPI_CHAR,0,icCommunicator()->communicator());
#endif // PV_USE_MPI
#ifdef OBSOLETE // Marked obsolete Feb 6, 2012.  nextCPWrite{Time,Step} is retrieved from params file so it doesn't have to be saved in timeinfo
   float * fbuf = (float *) (buf);
   int * ibuf = (int *) (buf+2*sizeof(float));
   simTime = fbuf[0];
   nextCPWriteTime = fbuf[1];
   currentStep = ibuf[0];
   nextCPWriteStep = ibuf[1];
#endif // OBSOLETE
   simTime = *((float *) buf);
   currentStep = *((int *) (buf+sizeof(float)));
   float checkTime;
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointRead(cpDir, &checkTime);
      assert(checkTime==simTime);
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointRead(cpDir, &checkTime);
      assert(checkTime==simTime);
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
   fprintf(stderr, "Rank %d in checkpointWrite. simTime = %f\n", icComm->commRank(), simTime);
   if( currentStep >= HYPERCOL_DIRINDEX_MAX+1 ) {
      if( icComm->commRank() == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Column \"%s\": step number exceeds maximum value %d.  Exiting\n", name, HYPERCOL_DIRINDEX_MAX);
      }
      exit(EXIT_FAILURE);
   }
   ensureDirExists(cpDir);
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointWrite(cpDir);
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointWrite(cpDir);
   }
   if( icCommunicator()->commRank()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      FILE * timestampfile = fopen(timestamppath,"w");
      assert(timestampfile);
      fwrite(&simTime,1,sizeof(float),timestampfile);
      fwrite(&currentStep,1,sizeof(int),timestampfile);
      fclose(timestampfile);
      chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timestampfile = fopen(timestamppath,"w");
      assert(timestampfile);
      fprintf(timestampfile,"time = %g\n", simTime);
      fprintf(timestampfile,"timestep = %d\n", currentStep);
      fclose(timestampfile);
   }

   if (deleteOlderCheckpoints) {
      assert(checkpointWriteFlag); // checkpointWrite is called by exitRunLoop when checkpointWriteFlag is false; in this case deleteOlderCheckpoints should be false as well.
      if (lastCheckpointDir[0]) {
         if (icComm->commRank()==0) {
            struct stat lcp_stat;
            int statstatus = stat(lastCheckpointDir, &lcp_stat);
            if ( statstatus!=0 || !(lcp_stat.st_mode & S_IFDIR) ) {
               if (statstatus==0) {
                  fprintf(stderr, "Error deleting older checkpoint: failed to stat \"%s\": error %d.\n", lastCheckpointDir, errno);
               }
               else {
                  fprintf(stderr, "Deleting older checkpoint: \"%s\" exists but is not a directory.\n", lastCheckpointDir);
               }
            }
// Delete old checkpoint.  Calling system('rm -r ...') rings alarm bells, and should.  So I masked the rm -r with an echo command.
// As it appears on the repository, setting deleteOlderCheckpoints to true doesn't actually delete the checkpoint, but instead
// sends the string 'rm -r ...' to standard output.  To really activate the deleteOlderCheckpoint feature,
// delete the echo from the snprintf format string below.
#define RMRFSIZE (PV_PATH_MAX + 13)
            char rmrf_string[RMRFSIZE];
            int chars_needed = snprintf(rmrf_string, RMRFSIZE, "rm -r '%s'", lastCheckpointDir);  // deleted "echo"  disabling
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

int HyPerCol::outputParams(const char * filename) {
   int status = PV_SUCCESS;
#ifdef PV_USE_MPI
   int rank=icComm->commRank();
#else
   int rank=0;
#endif
   if( rank==0 && filename != NULL && filename[0] != '\0' ) {
      char printParamsPath[PV_PATH_MAX];
      int len = snprintf(printParamsPath, PV_PATH_MAX, "%s/%s", outputPath, filename);
      if( len < PV_PATH_MAX ) {
         FILE * fp = fopen(printParamsPath, "w");
         if( fp != NULL ) {
            status = params->outputParams(fp);
            if( status != PV_SUCCESS ) {
               fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", filename);
            }
         }
         else {
            status = errno;
            fprintf(stderr, "outputParams: Unable to open \"%s\" for writing.  Error %d\n", filename, errno);
         }
      }
      else {
         fprintf(stderr, "outputParams: outputPath + printParamsFilename gives too long a filename.  Parameters will not be printed.\n");
      }
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
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%d", checkpointWriteDir, currentStep);
      }
      else {
         assert(!suppressLastOutput);
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Last", outputPath);
      }
      if(chars_printed >= PV_PATH_MAX) {
         if (icComm->commRank()==0) {
            fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%d\" is too long.\n", checkpointWriteDir, currentStep);
            abort();
         }
      }
      checkpointWrite(cpDir);
   }

#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Final output is written to {outputPath}/Last, above, using CheckpointWrite
   bool last = true;
   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(simTime, last);
   }

   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(simTime, last);
   }

   if (exitOnFinish) {
      delete this;
      exit(0);
   }
#endif // OBSOLETE

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

#ifdef OBSOLETE // Marked obsolete Nov 1, 2011.  Nobody calls this routine and it will be supplanted by checkpointWrite()
int HyPerCol::writeState()
{
   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(simTime);
   }
   return 0;
}
#endif // OBSOLETE


int HyPerCol::insertProbe(ColProbe * p)
{
   ColProbe ** newprobes;
   newprobes = (ColProbe **) malloc((numProbes + 1) * sizeof(ColProbe *));
   assert(newprobes != NULL);

   for (int i = 0; i < numProbes; i++) {
      newprobes[i] = probes[i];
   }
   delete probes;

   probes = newprobes;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerCol::outputState(float time)
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


int HyPerCol::checkMarginWidths() {
   // For each connection, make sure that the pre-synaptic margin width is
   // large enough for the patch size.

   // TODO instead of having marginWidth supplied to HyPerLayers in the
   // params.pv file, calculate them based on the patch sizes here.
   // Hard part:  numExtended-sized quantities (e.g. clayer->activity) can't
   // be allocated and initialized until after nPad is determined.

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


} // PV namespace
