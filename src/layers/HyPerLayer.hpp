/*
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 *  To make it easy to subclass from classes in the HyPerLayer hierarchy,
 *  please follow the guidelines below when adding subclasses to the HyPerLayer hierarchy:
 *
 *  For a class named DerivedLayer that is derived from a class named BaseLayer,
 *  the .hpp file should have
namespace PV {
class DerivedLayer : public BaseLayer {
public:
  DerivedLayer(arguments); // The constructor called by
  // other methods
protected:
  DerivedLayer();
  int initialize(arguments);
  // other methods and member variables
private:
  int initialize_base();
  // other methods and member variables
};
}
 *
 * The .cpp file should have
namespace PV {
DerivedLayer::DerivedLayer() {
  initialize_base();
  // initialize(arguments) should *not* be called by the protected constructor.
}
DerivedLayer::DerivedLayer(arguments, generally includes the layer's name and the parent HyPerCol) {
  initialize_base();
  initialize(arguments);
}
DerivedLayer::initialize_base() {
  // the most basic initializations.  Don't call any virtual methods,
  // or methods that call virtual methods, etc. from initialize_base();
}
DerivedLayer::initialize(arguments) {
  // DerivedLayer-specific initializations that need to precede BaseClass initialization, if any
  BaseClass::initialize(BaseClass initialization arguments);
  // DerivedLayer-specific initializations
}

  // other DerivedLayer methods
}
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include "../layers/PVLayer.h"
#include "../layers/LayerDataInterface.hpp"
#include "../columns/DataStore.hpp"
#include "../columns/HyPerCol.hpp"
#include "../columns/InterColComm.hpp"
#include "../io/LayerProbe.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"

#ifndef PV_USE_OPENCL
#  include "../layers/updateStateFunctions.h"
#else
#  undef PV_USE_OPENCL
#  include "../layers/updateStateFunctions.h"
#  define PV_USE_OPENCL
#endif


#ifdef PV_USE_OPENCL
#define PV_CL_COPY_BUFFERS 0
#define PV_CL_EVENTS 1
#include "../arch/opencl/CLKernel.hpp"
#define EV_GSYN 0
#define EV_ACTIVITY 1
#define EV_HPL_PHI_E 0
#define EV_HPL_PHI_I 1
#endif

namespace PV {

class InitV;

class HyPerLayer : public LayerDataInterface {

protected:

   // only subclasses can be constructed directly
   HyPerLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);

   virtual int initializeLayerId(int layerId);
   int setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int margin, int nf);
   virtual int allocateBuffers();
   int readDataStoreFromFile(const char * filename, InterColComm * comm, double * timed);
   static int readHeader(const char * filename, InterColComm * comm, double * timed, int * params, const PVLayerLoc * loc);
   static int writeBuffer(FILE * fp, InterColComm * comm, double dtime, pvdata_t * buffer, int numbands, bool extended, bool contiguous, const PVLayerLoc * loc);
   int incrementNBands(int * numCalls);
   int writeDataStoreToFile(const char * filename, InterColComm * comm, double dtime);
   virtual int calcActiveIndices();
   int readScalarFloat(const char * cp_dir, const char * val_name, double * val_ptr, double default_value=0.0f);
   int writeScalarFloat(const char * cp_dir, const char * val_name, double value);

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);
#endif

private:
   int initialize_base();

public:

   virtual ~HyPerLayer() = 0;
   virtual int initializeState();

   static int copyToBuffer(pvdata_t * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);
   static int copyToBuffer(unsigned char * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);

   template <typename T>
   static int copyFromBuffer(const T * buf, T * data,
                             const PVLayerLoc * loc, bool extended, T scale);

   static int copyFromBuffer(const unsigned char * buf, pvdata_t * data,
                             const PVLayerLoc * loc, bool extended, float scale);

   // TODO - make protected
   PVLayer  * clayer;

   virtual int triggerReceive(InterColComm * comm);
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int arborID);
   virtual int updateState (double time, double dt);
   virtual int updateBorder(double time, double dt);
   virtual int publish(InterColComm * comm, double time);
   virtual int waitOnPublish(InterColComm * comm);

   virtual int updateActiveIndices();
   int resetBuffer(pvdata_t * buf, int numItems);

   virtual int reconstruct(HyPerConn * conn, PVLayerCube * cube);

   int initFinish();

   int mirrorInteriorToBorder(int whichBorder, PVLayerCube * cube, PVLayerCube * borderCube);

   virtual int columnWillAddLayer(InterColComm * comm, int id);

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);
   static int readBufferFile(const char * filename, InterColComm * comm, double * timed, pvdata_t * buffer, int numbands, bool extended, bool contiguous, const PVLayerLoc * loc);
   static int writeBufferFile(const char * filename, InterColComm * comm, double dtime, pvdata_t * buffer, int numbands, bool extended, bool contiguous, const PVLayerLoc * loc);

   virtual int readState (double * timef);
#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Dumping the state is now done by checkpointWrite.
   virtual int writeState(double timef, bool last=false);
#endif // OBSOLETE
   virtual int outputState(double timef, bool last=false);
   virtual int writeActivity(double timef);
   virtual int writeActivitySparse(double timef);

   virtual int insertProbe(LayerProbe * probe);

   /** returns the number of neurons in layer (for borderId=0) or a border region **/
   virtual int numberOfNeurons(int borderId);

   virtual int mirrorToNorthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToNorth    (PVLayerCube * dest, PVLayerCube* src);
   virtual int mirrorToNorthEast(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToWest     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToEast     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouth    (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthEast(PVLayerCube * dest, PVLayerCube * src);

   const char * getOutputFilename(char * buf, const char * dataName, const char * term);

   // Public access functions:

   const char * getName()            {return name;}

   int getNumNeurons()               {return clayer->numNeurons;}
   int getNumExtended()              {return clayer->numExtended;}
   int getNumGlobalNeurons()         {const PVLayerLoc * loc = getLayerLoc(); return loc->nxGlobal*loc->nyGlobal*loc->nf;}
   int getNumGlobalExtended()        {const PVLayerLoc * loc = getLayerLoc(); return (loc->nxGlobal+2*loc->nb)*(loc->nyGlobal+2*loc->nb)*loc->nf;}

   int  getLayerId()                 {return clayer->layerId;}
   PVLayerType getLayerType()        {return clayer->layerType;}
   void setLayerId(int id)           {clayer->layerId = id;}
   int increaseDelayLevels(int neededDelay);

   PVLayer*  getCLayer()             {return clayer;}
   pvdata_t * getV()                 {return clayer->V;}           // name query
   pvdata_t * getActivity()          {return clayer->activity->data;}
   int getNumChannels()              {return numChannels;}
   pvdata_t * getChannel(ChannelType ch) {                         // name query
      return ch < this->numChannels ? GSyn[ch] : NULL;
   }
   int getXScale()                   {return clayer->xScale;}
   int getYScale()                   {return clayer->yScale;}

   HyPerCol* getParent()             {return parent;}
   void setParent(HyPerCol* parent)  {this->parent = parent;}

   bool useMirrorBCs()               {return this->mirrorBCflag;}
   bool getSpikingFlag()             {return this->spikingFlag;}

   // implementation of LayerDataInterface interface
   //
   const pvdata_t   * getLayerData(int delay=0);
   const PVLayerLoc * getLayerLoc()  { return &(clayer->loc); }
   bool isExtended()                 { return true; }

   virtual int gatherToInteriorBuffer(unsigned char * buf);

   virtual int label(int k);

   virtual int * getMarginIndices();
   virtual int getNumMargin();

protected:

   /* static */ int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   virtual int setActivity();
   void freeChannels();

   HyPerCol * parent;

   char * name;                 // well known name of layer

   int numChannels;             // number of channels
   pvdata_t ** GSyn;            // of dynamic length numChannels

   int numProbes;
   LayerProbe ** probes;

   int * labels;                // label for the feature a neuron is tuned to

   bool mirrorBCflag;           // true when mirror BC are to be applied

   int ioAppend;                // controls opening of binary files
   double writeTime;             // time of next output
   float writeStep;             // output time interval

   bool spikingFlag;
   bool writeNonspikingActivity;
   int writeActivityCalls;      // Number of calls to writeActivity (written to nbands in the header of the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the header of the a%d.pvp file)

   int * marginIndices;   // indices of neurons in margin
   int numMargin;         // number of neurons in margin

   // OpenCL variables
   //
#ifdef PV_USE_OPENCL
public:
   int initializeGPU(); //this method setups up GPU stuff...
   //virtual int getNumCLEvents() {return 0;}
   virtual const char * getKernelName() {return NULL;}
   virtual int getNumKernelArgs() {return numKernelArgs;}
   virtual int getNumCLEvents()   {return numEvents;}

   CLBuffer * getChannelCLBuffer() {
      return clGSyn;
   }
//   CLBuffer * getChannelCLBuffer(ChannelType ch) {
//      return ch < this->numChannels ? clGSyn[ch] : NULL;
//   }
   //#define EV_PHI_E 0
   //#define EV_PHI_I 1
   virtual int getEVGSyn() {return EV_GSYN;}
   virtual int getEVGSynE() {return EV_HPL_PHI_E;}
   virtual int getEVGSynI() {return EV_HPL_PHI_I;}
   virtual int getEVActivity() {return EV_ACTIVITY;}
   CLBuffer * getLayerDataStoreCLBuffer();
   size_t     getLayerDataStoreOffset(int delay=0);
   void initUseGPUFlag();
   inline bool getUseGPUFlag() {return gpuAccelerateFlag;}
   //int initializeDataStoreThreadBuffers();
   inline bool getCopyDataStoreFlag() {return copyDataStoreFlag;}
   int waitForDataStoreCopy();
   int copyDataStoreCLBuffer();
   void tellLayerToCopyDataStoreCLBuffer(/*cl_event * evCpDataStore*/) {copyDataStoreFlag=true; /*evCopyDataStore=evCpDataStore;*/}

   //temporary method for debuging recievesynapticinput
   virtual inline int getGSynEvent(ChannelType ch) {
      switch (ch) {
      case CHANNEL_EXC: return getEVGSynE();
      case CHANNEL_INH: return getEVGSynI();
      default: return -1;
      }
   }
   virtual void copyGSynFromDevice() {
      int gsynEvent = getEVGSyn();
//      if(numWait>0) {
//         clWaitForEvents(numWait, &evList[gsynEvent]);
//         for (int i = 0; i < numWait; i++) {
//            clReleaseEvent(evList[i]);
//         }
//         numWait = 0;
//      }
      if(gsynEvent>=0){
         clGSyn->copyFromDevice(&evList[gsynEvent]);
         clWaitForEvents(1, &evList[gsynEvent]);
         clReleaseEvent(evList[gsynEvent]);
      }
   }
//   virtual void copyChannelFromDevice(ChannelType ch) {
//      int gsynEvent = getGSynEvent(ch);
//      if(gsynEvent>=0){
//         getChannelCLBuffer(ch)->copyFromDevice(&evList[gsynEvent]);
//         clWaitForEvents(1, &evList[gsynEvent]);
//         clReleaseEvent(evList[gsynEvent]);
//      }
//   }
   virtual void copyGSynToDevice() {
      int gsynEvent = getEVGSyn();
      if(gsynEvent>=0){
         clGSyn->copyToDevice(&evList[gsynEvent]);
         clWaitForEvents(1, &evList[gsynEvent]);
         clReleaseEvent(evList[gsynEvent]);
      }
      //copyToDevice=true;
   }
   void startTimer() {recvsyn_timer->start();}
   void stopTimer() {recvsyn_timer->stop();}

protected:

   CLKernel * krUpdate;        // CL kernel for update state call

   // OpenCL buffers
   //
   CLBuffer * clV;
   //CLBuffer **clGSyn;         // of dynamic length numChannels
   CLBuffer * clGSyn;         // of dynamic length numChannels
   CLBuffer * clActivity;
   CLBuffer * clPrevTime;
   CLBuffer * clParams;       // for transferring params to kernel

   int numKernelArgs;         // number of arguments in kernel call
   int numEvents;             // number of events in event list
   int numWait;               // number of events to wait for
   cl_event * evList;         // event list
   cl_event   evUpdate;
   //cl_event * evCopyDataStore;

   int nxl;  // local OpenCL grid size in x
   int nyl;  // local OpenCL grid size in y

   bool gpuAccelerateFlag;
   //bool copyToDevice;
   bool copyDataStoreFlag;
   //bool buffersInitialized;

#endif

   Timer * update_timer;
   Timer * recvsyn_timer;
};

} // namespace PV

// Template functions
//
template <typename T>
int PV::HyPerLayer::copyFromBuffer(const T * buf, T * data,
                                   const PVLayerLoc * loc, bool extended, T scale)
{
   size_t sf, sx, sy;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nb;
      nyBorder = loc->nb;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
   }
   else {
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + nyBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + nxBorder;
         for (int f = 0; f < nf; f++) {
            data[iex*sx + jex*sy + f*sf] = scale * buf[ii++];
         }
      }
   }
   return 0;
}

#endif /* HYPERLAYER_HPP_ */
