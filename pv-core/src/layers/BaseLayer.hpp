/*
 * BaseLayer.hpp
 *
 *  Created on: Jan 16, 2010
 *      Author: rasmussn
 */

#ifndef BASELAYER_HPP_
#define BASELAYER_HPP_

#include "../io/LayerProbe.hpp"
//#include "../columns/HyPerCol.hpp"
//#include "../utils/conversions.h"

namespace PV {

/*
 * Interface providing access to a layer's data
 */

class BaseLayer {
public:
   BaseLayer();
   virtual ~BaseLayer();

//   virtual const PVLayerLoc * getLayerLoc() = 0;
//   virtual const pvdata_t   * getLayerData(int delay=0) = 0;
//   virtual bool  isExtended() = 0;
//   virtual int   gatherToInteriorBuffer(unsigned char * buf) = 0;
//   virtual int publish(InterColComm * comm, double time){return PV_FAILURE;}
//   virtual int checkpointRead(){return PV_FAILURE;}
//
//   // mpi public wait method to ensure all targets have received synaptic input before proceeding to next time step
//   virtual int waitOnPublish(InterColComm * comm) = 0;
//   virtual int updateActiveIndices() = 0;
//   virtual int outputState(double timef, bool last=false) = 0;
//   virtual int communicateInitInfo() = 0;
//   virtual int allocateDataStructures() = 0;
//   virtual float addGpuTimers() = 0;
//   virtual void syncGpu() = 0;
//   virtual int resetGSynBuffers(double timef, double dt) = 0;
//   virtual int recvAllSynapticInput() = 0; // Calls recvSynapticInput for each conn and each arborID
//   virtual int updateStateWrapper (double time, double dt) = 0;
//   virtual void copyAllGSynToDevice() = 0;
//   virtual void copyAllGSynFromDevice() = 0;
//   virtual void copyAllVFromDevice() = 0;
//   virtual void copyAllActivityFromDevice() = 0;
//   virtual int updateBorder(double time, double dt) = 0;
//   virtual int checkErrorNotANumber(){return PV_SUCCESS;}
//   virtual int writeTimers(FILE* stream) = 0;
//   virtual int checkpointWrite(const char * cpDir) = 0;
//   virtual int outputProbeParams() = 0;
//   virtual int initializeState() = 0;
//   
//
//   int ioParams(enum ParamsIOFlag ioFlag);
//   HyPerCol* getParent()             {return parent;}
//   void setParent(HyPerCol* parent)  {this->parent = parent;}
//   int  getLayerId()                 {return layerId;}
//   void setLayerId(int id)           {layerId = id;}
//   PVDataType getDataType()          {return dataType;}
//   const char * getName()            {return name;}
//   int getPhase()                    {return this->phase;}
//
//   bool getInitInfoCommunicatedFlag() {return initInfoCommunicatedFlag;}
//   bool getDataStructuresAllocatedFlag() {return dataStructuresAllocatedFlag;}
//   bool getInitialValuesSetFlag() {return initialValuesSetFlag;}
//   // TODO The three routines below shouldn't be public, but HyPerCol needs to call them, so for now they are.
//   void setInitInfoCommunicatedFlag() {initInfoCommunicatedFlag = true;}
//   void setDataStructuresAllocatedFlag() {dataStructuresAllocatedFlag = true;}
//   void setInitialValuesSetFlag() {initialValuesSetFlag = true;}
//   int getNumDelayLevels()           {return numDelayLevels;}
//   virtual double calcTimeScale()          {return -1.0;};
//
//   bool getRecvGpu(){
//      return recvGpu;
//   }
//
//   bool getUpdateGpu(){
//      return updateGpu;
//   }
//
//   bool getSparseFlag()             {return this->sparseLayer;}
//   int getNumExtended()              {return clayer->numExtended;}
//
//   // TODO - make protected
//   PVLayer  * clayer;
//   double getLastUpdateTime() { return lastUpdateTime; }
//
//
//protected:
//   int initialize(const char * name, HyPerCol * hc);
//   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
//   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);
//   char* dataTypeString;
//   PVDataType dataType;
//
//   HyPerCol * parent;
//   char * name;                 // well known name of layer
//   int layerId;                 // unique ID that identifies layer in its parent HyPerCol
//   int phase;                   // All layers with phase 0 get updated before any with phase 1, etc.
//   bool initInfoCommunicatedFlag;
//   bool dataStructuresAllocatedFlag;
//   bool initialValuesSetFlag;
//   bool recvGpu;
//   bool updateGpu;
//
//   int numDelayLevels;          // The number of timesteps in the datastore ring buffer to store older timesteps for connections with delays
//   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
//   double lastUpdateTime; // The most recent time that the layer's activity is updated, used as a cue for publisher to exchange borders
//
//private:
//   int initialize_base();



};

} // namespace PV

#endif /* BASELAYER_HPP_ */
