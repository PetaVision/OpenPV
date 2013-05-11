/*
 * HyPerCol.hpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#undef UNDERCONSTRUCTION

#include "HyPerColRunDelegate.hpp"
#include "InterColComm.hpp"
#include "../layers/PVLayer.h"
#include "../layers/HyPerLayer.hpp"
#include "../connections/HyPerConn.hpp"
#include "../io/PVParams.hpp"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"
#include "../io/ColProbe.hpp"
#include <time.h>
#include <sys/stat.h>
#include <fstream>

#include "../arch/opencl/CLDevice.hpp"

namespace PV {

//class HyPerLayer;
//class InterColComm;
//class HyPerConn;
class ColProbe;
class PVParams;

class HyPerCol {

public:

   HyPerCol(const char * name, int argc, char * argv[], PV::PVParams * params=NULL);
   // HyPerCol(const char* name, int argc, char* argv[], const char * path); // Not defined in .cpp file
   virtual ~HyPerCol();

   int initFinish(void); // call after all layers/connections have been added
   int initializeThreads(int device);
#ifdef PV_USE_OPENCL
   int finalizeThreads();
#endif //PV_USE_OPENCL

   int run()  {return run(numSteps);}
   int run(long int nTimeSteps);

   int advanceTime(double time);
   int exitRunLoop(bool exitOnFinish);

   int loadState();
   int columnId();

//   int deliver(PVConnection* conn, PVRect preRegion, int count, float* buf);

   int addLayer(HyPerLayer * l);
   int addConnection(HyPerConn * conn);

   HyPerLayer * getLayerFromName(const char * layerName);
   HyPerConn * getConnFromName(const char * connectionName);

   HyPerLayer * getLayer(int which)       {return layers[which];}
   HyPerConn  * getConnection(int which)  {return connections[which];}
   ColProbe * getColProbe(int which)      {return probes[which];}

   char * getName()                       {return name;}
   // char * getPath()                       {return path;}
   char * getOutputPath()                 {return outputPath;}
   int getNxGlobal()                      {return nxGlobal;}
   int getNyGlobal()                      {return nyGlobal;}

   CLDevice * getCLDevice()               {return clDevice;}

   InterColComm * icCommunicator()        {return icComm;}

   PVParams * parameters()                {return params;}

   bool  warmStartup()                    {return warmStart;}

   double getDeltaTime()                  {return deltaTime;}
   double simulationTime()                {return simTime;}
   double getStopTime()                   {return stopTime;}
   long int getCurrentStep()              {return currentStep;}
   bool getCheckpointReadFlag()           {return checkpointReadFlag;}
   bool getCheckpointWriteFlag()          {return checkpointWriteFlag;}
   bool getSuppresLastOutputFlag()        {return suppressLastOutput;}

   int includeLayerName()                 {return filenamesContainLayerNames;}
   int includeConnectionName()            {return filenamesContainConnectionNames;}

   const char * inputFile()               {return image_file;}

   long int numberOfTimeSteps()           {return numSteps;}

   int numberOfColumns();

   int numberOfLayers()                   {return numLayers;}
   int numberOfConnections()              {return numConnections;}
   int numberOfProbes()                   {return numProbes;}

   /** returns the number of border regions, either an actual image border or a neighbor **/
   int numberOfBorderRegions()            {return MAX_NEIGHBORS;}

   int commColumn(int colId);
   int commRow(int colId);
   int numCommColumns()                   {return icComm->numCommColumns();}
   int numCommRows()                      {return icComm->numCommRows();}

   // a random seed based on column id
   unsigned long getSeed() { return random_seed; }
   unsigned long getObjectSeed(int count) { unsigned long seed = random_seed_obj; random_seed_obj += count; return seed;}

   unsigned long getRandomSeed();
      // Nov. 28, 2012.  All MPI processes get the same base seed, and should use global information to seed individual neurons.
      // {return (unsigned long) time((time_t *) NULL); } // Aug 21, 2012: Division by 1+columnId() moved to calling routine}

   void setDelegate(HyPerColRunDelegate * delegate)  {runDelegate = delegate;}

   int insertProbe(ColProbe * p);
   int outputState(double time);
   int ensureDirExists(const char * dirname);

   template <typename T>
   int writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val);
   template <typename T>
   int readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value=(T) 0);

#ifdef UNDERCONSTRUCTION
   static int outputParamGroup(PV_Stream * pvstream, const char * classname, const char * groupname, int indentation);
   static int outputParamCloseGroup(PV_Stream * pvstream, const char * classname, int indentation);
   static int outputParamInt(PV_Stream * pvstream, const char * paramName, int value, int indentation);
   static int outputParamLongInt(PV_Stream * pvstream, const char * paramName, long int value, int indentation);
   static int outputParamUnsignedLongInt(PV_Stream * pvstream, const char * paramName, unsigned long int value, int indentation);
   static int outputParamFloat(PV_Stream * pvstream, const char * paramName, float value, int indentation);
   static int outputParamDouble(PV_Stream * pvstream, const char * paramName, double value, int indentation);
   static int outputParamBoolean(PV_Stream * pvstream, const char * paramName, bool value, int indentation);
   static int outputParamFilename(PV_Stream * pvstream, const char * paramName, const char * value, int indentation);
   static int outputParamString(PV_Stream * pvstream, const char * paramName, const char * value, int indentation);
   static int indent(PV_Stream * pvstream, int indentation);
#endif // UNDERCONSTRUCTION

private:
   int initialize_base();
   int initialize(const char * name, int argc, char ** argv, PVParams * params);
   int checkDirExists(const char * dirname, struct stat * pathstat);
   int initPublishers();
   bool advanceCPWriteTime();
   int checkpointRead(const char * cpDir);
   int checkpointWrite(const char * cpDir);
   int outputParams(const char * filename);
#ifdef UNDERCONSTRUCTION // Plans to output the params, including those set to default values, as an XML file.
   int outputParamsXML(const char * filename);
   int outputParamsXML(PV_Stream * pvstream);
   template <typename T> static int hexdump(PV_Stream * pvstream, T value);
#endif // UNDERCONSTRUCTION
   int checkMarginWidths();
   int zCheckMarginWidth(HyPerConn * conn, const char * dim, int patchSize, int scalePre, int scalePost, int prevStatus);
   int lCheckMarginWidth(HyPerLayer * layer, const char * dim, int layerSize, int layerGlobalSize, int prevStatus);

   long int numSteps;
   long int currentStep;
   size_t layerArraySize;
   int numLayers;
   int numPhases;
   size_t connectionArraySize;
   int numConnections;

   bool warmStart;
   bool isInitialized;     // true when all initialization has been completed
   bool checkpointReadFlag;    // whether to load from a checkpoint directory
   bool checkpointWriteFlag;   // whether to write from a checkpoint directory
   char * checkpointReadDir;   // name of the directory to read an initializing checkpoint from
   long int cpReadDirIndex;  // checkpoint number within checkpointReadDir to read
   char * checkpointWriteDir; // name of the directory to write checkpoints to
   long int cpWriteStepInterval;
   long int nextCPWriteStep;
   double cpWriteTimeInterval;
   double nextCPWriteTime;
   bool deleteOlderCheckpoints; // If true, whenever a checkpoint other than the first is written, the preceding checkpoint is deleted. Default is false.
   char lastCheckpointDir[PV_PATH_MAX]; // Holds the last checkpoint directory written; used if deleteOlderCheckpoints is true.

   bool suppressLastOutput; // If checkpointWriteFlag is false and this flag is false, on exit a checkpoint is sent to the {outputPath}/Last directory.
                            // If checkpointWriteFlag is false and this flag is true, no checkpoint is done on exit.
                            // The flag has no effect if checkpointWriteFlag is true (in which case a checkpoint is written on exit to the next directory in checkpointWriteDir

   double simTime;          // current time in milliseconds
   double stopTime;         // time to stop time
   double deltaTime;        // time step interval
   long int progressStep;       // How many timesteps between outputting progress
   bool writeProgressToErr;// Whether to write progress step to standard error (True) or out (False) (default is out)

   CLDevice * clDevice;    // object for running kernels on OpenCL device

   HyPerLayer ** layers;
   HyPerConn  ** connections;

   char * name;
   // char * path;
   char * outputPath;     // path to output file directory
   char * outputNamesOfLayersAndConns;  // path to file for writing list of layer names and connection names
   char * image_file;
   int nxGlobal;
   int nyGlobal;

   bool           ownsParams; // True if params was created from params file by initialize, false if params was passed in the constructor
   bool           ownsInterColComm; // True if icComm was created by initialize, false if passed in the constructor
   PVParams     * params; // manages input parameters
   InterColComm * icComm; // manages communication between HyPerColumns};

   HyPerColRunDelegate * runDelegate; // runs time loop

   Timer * runTimer;

   int numProbes;
   ColProbe ** probes;

   int filenamesContainLayerNames; // Controls the form of layers' clayer->activeFP
                                   // Value 0: layers have form a5.pvp
                                   // Value 1: layers have form a5_NameOfLayer.pvp
                                   // Value 2: layers have form NameOfLayer.pvp
   int filenamesContainConnectionNames; // Similar to filenamesContainLayerNames, but for connections

   unsigned long random_seed;
   unsigned long random_seed_obj;  // Objects that need to generate random numbers should request a seed from
                                   // the HyPerCol, saying how many they need (across all processes in an MPI run).
                                   // random_seed_obj is incremented by the number requested, so that everything
                                   // that needs a random seed gets a unique seed, and things are reproducible.

}; // class HyPerCol

} // namespace PV

#endif /* HYPERCOL_HPP_ */
