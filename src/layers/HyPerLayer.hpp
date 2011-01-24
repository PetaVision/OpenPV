/*
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include "../layers/PVLayer.h"
#include "../layers/LayerDataInterface.hpp"
#include "../columns/DataStore.hpp"
#include "../columns/HyPerCol.hpp"
#include "../columns/InterColComm.hpp"
#include "../io/LayerProbe.hpp"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLKernel.hpp"
#endif

namespace PV {

class HyPerLayer : public LayerDataInterface {

protected:

   // only subclasses can be constructed directly
   HyPerLayer(const char * name, HyPerCol * hc, int numChannels);

   virtual int initializeLayerId(int layerId);

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers() = 0;
   virtual int initializeThreadKernels() = 0;
#endif

private:
   int initialize_base(const char * name, HyPerCol * hc, int numChannels);

public:

   virtual ~HyPerLayer() = 0;

   static int copyToBuffer(pvdata_t * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);
   static int copyToBuffer(unsigned char * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);

   static int copyFromBuffer(const pvdata_t * buf, pvdata_t * data,
                             const PVLayerLoc * loc, bool extended, float scale);
   static int copyFromBuffer(const unsigned char * buf, pvdata_t * data,
                             const PVLayerLoc * loc, bool extended, float scale);

   // TODO - make protected
   PVLayer  * clayer;
   HyPerCol * parent;

   virtual int triggerReceive(InterColComm * comm);
   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState (float time, float dt);
   virtual int updateBorder(float time, float dt);
   virtual int publish(InterColComm * comm, float time);
   virtual int waitOnPublish(InterColComm * comm);

   virtual int updateV();
   virtual int setActivity();
   virtual int resetPhiBuffers();
   int resetBuffer(pvdata_t * buf, int numItems);

   virtual int reconstruct(HyPerConn * conn, PVLayerCube * cube);

   int initialize(PVLayerType type);
   int initFinish();

   int mirrorInteriorToBorder(int whichBorder, PVLayerCube * cube, PVLayerCube * borderCube);

   virtual int columnWillAddLayer(InterColComm * comm, int id);

   virtual int outputState(float time, bool last=false);
   virtual int writeState(const char * name, float time, bool last=false);
#ifdef OBSOLETE // (marked obsolete Jan 24, 2011)
   virtual int writeActivity(const char * filename, float time);
#endif // OBSOLETE
   virtual int writeActivity(float time);
   virtual int writeActivitySparse(float time);
   virtual int readState(const char * name, float * time);

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

   // Public access functions:

   const char * getName()            {return name;}

   int getNumNeurons()               {return clayer->numNeurons;}
   int getNumExtended()              {return clayer->numExtended;}

   int  getLayerId()                 {return clayer->layerId;}
   PVLayerType getLayerType()        {return clayer->layerType;}
   void setLayerId(int id)           {clayer->layerId = id;}

   PVLayer*  getCLayer()             {return clayer;}
   pvdata_t * getV()                 {return clayer->V;}           // name query
   pvdata_t * getChannel(ChannelType ch) {                         // name query
      return ch < this->numChannels ? phi[ch] : NULL;
   }
   int getXScale()                   {return clayer->xScale;}
   int getYScale()                   {return clayer->yScale;}

   HyPerCol* getParent()             {return parent;}
   void setParent(HyPerCol* parent)  {this->parent = parent;}

   bool useMirrorBCs()               {return this->mirrorBCflag;}

   // implementation of LayerDataInterface interface
   //
   const pvdata_t   * getLayerData();
   const PVLayerLoc * getLayerLoc()  { return &clayer->loc; }
   bool isExtended()                 { return true; }

   virtual int gatherToInteriorBuffer(unsigned char * buf);

protected:
   char * name;                 // well known name of layer

   int numChannels;             // number of channels
   pvdata_t * phi[MAX_CHANNELS];

   int numProbes;
   LayerProbe ** probes;

   bool mirrorBCflag;           // true when mirror BC are to be applied

   int ioAppend;                // controls opening of binary files
   float writeTime;             // time of next output
   float writeStep;             // output time interval

   // OpenCL variables
   //
#ifdef PV_USE_OPENCL
   CLKernel * krUpdate;        // CL kernel for update state call

   // OpenCL buffers
   //
   CLBuffer * clV;
   CLBuffer * clVth;
   CLBuffer * clPhiE;
   CLBuffer * clPhiI;
   CLBuffer * clPhiIB;
   CLBuffer * clActivity;
   CLBuffer * clPrevTime;
   CLBuffer * clParams;       // for transferring params to kernel


   int numEvents;             // number of events in event list
   int numWait;               // number of events to wait for
   cl_event * evList;         // event list
   cl_event   evUpdate;

   int nxl;  // local OpenCL grid size in x
   int nyl;  // local OpenCL grid size in y
#endif

   Timer * update_timer;
   Timer * recvsyn_timer;
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
