/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP_
#define ANNLAYER_HPP_

#include "HyPerLayer.hpp"

#define NUM_ANN_EVENTS   3
#define EV_ANN_ACTIVITY  2

namespace PV {

class ANNLayer : public HyPerLayer {
public:
   ANNLayer(const char* name, HyPerCol * hc, int numChannels=MAX_CHANNELS);
   virtual ~ANNLayer();
   virtual int updateState(float time, float dt);
   // virtual int updateV();
   // virtual int applyVMax();
   // virtual int applyVThresh();
   pvdata_t getVThresh()        { return VThresh; }
   pvdata_t getVMax()           { return VMax; }
   pvdata_t getVMin()           { return VMin; }
protected:
   ANNLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   virtual int readVThreshParams(PVParams * params);
   pvdata_t VMax;
   pvdata_t VThresh;
   pvdata_t VMin;
#ifdef PV_USE_OPENCL
   virtual int getNumCLEvents() {return numEvents;}
   virtual const char * getKernelName() { return "ANNLayer_update_state"; }
   virtual int initializeThreadBuffers(const char * kernel_name);
   virtual int initializeThreadKernels(const char * kernel_name);
   //virtual int getEVActivity() {return EV_ANN_ACTIVITY;}
   int updateStateOpenCL(float time, float dt);
   //temporary method for debuging recievesynapticinput
public:
//   virtual void copyChannelExcFromDevice() {
//      getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(&evList[getEVGSynE()]);
//      clWaitForEvents(1, &evList[getEVGSynE()]);
//      clReleaseEvent(evList[getEVGSynE()]);
//   }
protected:
#endif // PV_USE_OPENCL

private:
   int initialize_base();
}; // end of class ANNLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
