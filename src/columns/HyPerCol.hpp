/*
 * HyPerCol.hpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

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
#include <typeinfo>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <sstream>

#include "../arch/opencl/CLDevice.hpp"

enum CheckpointWriteTriggerMode { CPWRITE_TRIGGER_STEP, CPWRITE_TRIGGER_TIME, CPWRITE_TRIGGER_CLOCK };

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

   int initializeThreads(int device);
#ifdef PV_USE_OPENCL
   int finalizeThreads();
#endif //PV_USE_OPENCL

   int run()  {return run(startTime, stopTime, deltaTimeBase);}
   int run(double startTime, double stopTime, double dt);

   int advanceTime(double time);
   int exitRunLoop(bool exitOnFinish);

   int loadState();
   int columnId();

//   int deliver(PVConnection* conn, PVRect preRegion, int count, float* buf);

   int addLayer(HyPerLayer * l);
   int addConnection(HyPerConn * conn);

   HyPerLayer * getLayerFromName(const char * layerName);
   HyPerConn * getConnFromName(const char * connectionName);
   ColProbe * getColProbeFromName(const char * probeName);

   HyPerLayer * getLayer(int which)       {return layers[which];}
   HyPerConn  * getConnection(int which)  {return connections[which];}
   ColProbe * getColProbe(int which)      {return probes[which];}

   char * getName()                       {return name;}
   char * getSrcPath()                    {return srcPath;}
   char * getOutputPath()                 {return outputPath;}
   int getNxGlobal()                      {return nxGlobal;}
   int getNyGlobal()                      {return nyGlobal;}

   CLDevice * getCLDevice()               {return clDevice;}

   InterColComm * icCommunicator()        {return icComm;}

   PV_Stream * getPrintParamsStream()      {return printParamsStream;}

   PVParams * parameters()                {return params;}

   bool  warmStartup()                    {return warmStart;}

   double getDeltaTime()                  {return deltaTime;}
   bool  getDtAdaptFlag()                 {return dtAdaptFlag;}
   double getDeltaTimeBase()              {return deltaTimeBase;}
   double getTimeScale()                  {return timeScale;}
   double getTimeScaleMax()               {return timeScaleMax;}
   double getTimeScaleMin()               {return timeScaleMin;}
   double getChangeTimeScaleMax()         {return changeTimeScaleMax;}
   double getChangeTimeScaleMin()         {return changeTimeScaleMin;}
   double simulationTime()                {return simTime;}
   double getStartTime()                  {return startTime;}
   double getStopTime()                   {return stopTime;}
   long int getInitialStep()              {return initialStep;}
   long int getFinalStep()                {return finalStep;}
   long int getCurrentStep()              {return currentStep;}
   bool getCheckpointReadFlag()           {return checkpointReadFlag;}
   bool getCheckpointWriteFlag()          {return checkpointWriteFlag;}
   bool getSuppresLastOutputFlag()        {return suppressLastOutput;}
   const char * getPrintParamsFilename()  {return printParamsFilename;}

   int includeLayerName()                 {return filenamesContainLayerNames;}
   int includeConnectionName()            {return filenamesContainConnectionNames;}

   const char * inputFile()               {return image_file;}

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
   unsigned int getSeed() { return random_seed; }
   unsigned int getObjectSeed(int count) { unsigned long seed = random_seed_obj; random_seed_obj += count; return seed;}

   unsigned int getRandomSeed();
      // Nov. 28, 2012.  All MPI processes get the same base seed, and should use global information to seed individual neurons.
      // {return (unsigned long) time((time_t *) NULL); } // Aug 21, 2012: Division by 1+columnId() moved to calling routine}

   void setDelegate(HyPerColRunDelegate * delegate)  {runDelegate = delegate;}

   int insertProbe(ColProbe * p);
   int addLayerProbe(LayerProbe * p);
   // int addBaseConnectionProbe(BaseConnectionProbe * p);
   int outputState(double time);
   int ensureDirExists(const char * dirname);

   template <typename T>
   int writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val);
   template <typename T>
   int readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value=(T) 0);

   int ioParamsStartGroup(enum ParamsIOFlag ioFlag, const char * group_name);
   int ioParamsFinishGroup(enum ParamsIOFlag ioFlag);
   template <typename T>
   void ioParamValueRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * val);
   template <typename T>
   void ioParamValue(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T * val, T defaultValue, bool warnIfAbsent=true);
   void ioParamString(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value, const char * defaultValue, bool warnIfAbsent=true);
   void ioParamStringRequired(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, char ** value);
   template <typename T>
   void ioParamArray(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, T ** value, int * arraysize);
   template <typename T>
   void writeParam(const char * param_name, T value);
   void writeParamString(const char * param_name, const char * svalue);
   template <typename T>
   void writeParamArray(const char * param_name, const T * array, int arraysize);

private:
   int initialize_base();
   int initialize(const char * name, int argc, char ** argv, PVParams * params);
   int ioParams(enum ParamsIOFlag ioFlag);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_startTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dt(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dtAdaptFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dtScaleMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dtScaleMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dtChangeMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dtChangeMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_stopTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_progressInterval(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeProgressToErr(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_printParamsFilename(enum ParamsIOFlag ioFlag);
   virtual void ioParam_randomSeed(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nx(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ny(enum ParamsIOFlag ioFlag);
   virtual void ioParam_filenamesContainLayerNames(enum ParamsIOFlag ioFlag);
   virtual void ioParam_filenamesContainConnectionNames(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointRead(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointWrite(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointWriteDir(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointWriteTriggerMode(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointWriteStepInterval(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkpointWriteTimeInterval(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deleteOlderCheckpoints(enum ParamsIOFlag ioFlag);
   virtual void ioParam_suppressLastOutput(enum ParamsIOFlag ioFlag);
   virtual void ioParam_printTimescales(enum ParamsIOFlag ioFlag);

   int checkDirExists(const char * dirname, struct stat * pathstat);

   int doInitializationStage(int (HyPerCol::*layerInitializationStage)(int), int (HyPerCol::*connInitializationStage)(int), const char * stageName);
   int layerCommunicateInitInfo(int l);
   int connCommunicateInitInfo(int c);
   int layerAllocateDataStructures(int l);
   int connAllocateDataStructures(int c);

   int initPublishers();
   bool advanceCPWriteTime();
   int checkpointRead(const char * cpDir);
   int checkpointWrite(const char * cpDir);
   int outputParams();

   virtual double adaptTimeScale();

#ifdef OBSOLETE // Marked obsolete Aug 9, 2013.  Look, everybody, checkMarginWidths is obsolete!
   int checkMarginWidths();
   int zCheckMarginWidth(HyPerConn * conn, const char * dim, int patchSize, int scalePre, int scalePost, int prevStatus);
   int lCheckMarginWidth(HyPerLayer * layer, const char * dim, int layerSize, int layerGlobalSize, int prevStatus);
#endif // OBSOLETE

   long int currentStep;
   long int initialStep;
   long int finalStep;
   size_t layerArraySize;
   int numLayers;
   int numPhases;
   size_t connectionArraySize;
   int numConnections;

   bool warmStart;             // whether to start from a checkpoint
   bool checkpointReadFlag;    // whether to load from a checkpoint directory
   bool checkpointWriteFlag;   // whether to write from a checkpoint directory
   char * checkpointReadDir;   // name of the directory to read an initializing checkpoint from
   char * checkpointReadDirBase;   // name of the directory containing che checkpoint read from (used by deprecated params-based method for loading from checkpoint)
   long int cpReadDirIndex;  // checkpoint number within checkpointReadDir to read
   char * checkpointWriteDir; // name of the directory to write checkpoints to
   enum CheckpointWriteTriggerMode checkpointWriteTriggerMode;
   char * checkpointWriteTriggerModeString;
   long int cpWriteStepInterval;
   long int nextCPWriteStep;
   double cpWriteTimeInterval;
   double nextCPWriteTime;
   bool deleteOlderCheckpoints; // If true, whenever a checkpoint other than the first is written, the preceding checkpoint is deleted. Default is false.
   char lastCheckpointDir[PV_PATH_MAX]; // Holds the last checkpoint directory written; used if deleteOlderCheckpoints is true.

   bool suppressLastOutput; // If checkpointWriteFlag is false and this flag is false, on exit a checkpoint is sent to the {outputPath}/Last directory.
                            // If checkpointWriteFlag is false and this flag is true, no checkpoint is done on exit.
                            // The flag has no effect if checkpointWriteFlag is true (in which case a checkpoint is written on exit to the next directory in checkpointWriteDir

   double startTime;
   double simTime;          // current time in milliseconds
   double stopTime;         // time to stop time
   double deltaTime;        // time step interval
   bool   dtAdaptFlag;      // turns adaptive time step on/off
   double deltaTimeBase;    // base time step interval if dtAdaptFlag == true, timeScale is applied to this value
   double timeScale;        // scale factor for deltaTimeBase, deltaTime = timeScale*deltaTimeBase
   double timeScaleTrue;    // true timeScale returned by min(HyPerLayer::getTimeScale) before MIN/MAX/CHANGE constraints applied
   double timeScaleMax;     // maximum value of timeScale (prevents deltaTime from growing too large)
   double timeScaleMin;     // minimum value of timeScale (not really a minimum, actually sets starting/iniital value of deltaTime)
   double changeTimeScaleMax;     // maximum change in value of timeScale (prevents deltaTime from growing too quickly)
   double changeTimeScaleMin;     // typically 0 or negative, maximum DECREASE in timeScale allowed before resetting timeScale -> timeScaleMin
   double progressInterval; // Output progress after simTime increases by this amount.
   double nextProgressTime; // Next time to output a progress message
   bool writeProgressToErr;// Whether to write progress step to standard error (True) or standard output (False) (default is output)

   CLDevice * clDevice;    // object for running kernels on OpenCL device

   HyPerLayer ** layers;
   HyPerConn  ** connections;

   char * name;
   char * srcPath;        // path to PetaVision src directory (used to compile OpenCL kernels)
   char * outputPath;     // path to output file directory
   // char * outputNamesOfLayersAndConns;  // path to file for writing list of layer names and connection names
   char * printParamsFilename; // filename for outputting the params, including defaults and excluding unread params
   PV_Stream * printParamsStream; // file pointer associated with printParamsFilename
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

   int numLayerProbes;
   LayerProbe ** layerProbes;

   // int numConnProbes;
   // BaseConnectionProbe ** connProbes;

   int filenamesContainLayerNames; // Controls the form of layers' clayer->activeFP
                                   // Value 0: layers have form a5.pvp
                                   // Value 1: layers have form a5_NameOfLayer.pvp
                                   // Value 2: layers have form NameOfLayer.pvp
   int filenamesContainConnectionNames; // Similar to filenamesContainLayerNames, but for connections

   unsigned int random_seed;
   unsigned int random_seed_obj;  // Objects that need to generate random numbers should request a seed from
                                  // the HyPerCol, saying how many they need (across all processes in an MPI run).
                                  // random_seed_obj is incremented by the number requested, so that everything
                                  // that needs a random seed gets a unique seed, and things are reproducible.
                                  //
   bool printTimescales;

}; // class HyPerCol

} // namespace PV

#endif /* HYPERCOL_HPP_ */
