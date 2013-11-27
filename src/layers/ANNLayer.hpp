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
   pvdata_t getVThresh()        { return VThresh; }
   pvdata_t getVMax()           { return VMax; }
   pvdata_t getVMin()           { return VMin; }
protected:
   ANNLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels=MAX_CHANNELS);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
   virtual int setActivity();
   virtual int readVThreshParams(PVParams * params);
   pvdata_t VMax;  // maximum membrane potential, larger values are set to VMax
   pvdata_t VMin;  // minimum membrane potential, smaller values are set to VMin
   pvdata_t VThresh;  // threshold potential, values smaller than VThresh are set to VMin
   pvdata_t VShift;  // shift potential, values above VThresh are shifted downward by this amount
                     // VShift == 0, hard threshold condition
                     // VShift == VThresh, soft threshold condition
   pvdata_t VWidth;  // The thresholding occurs linearly over the region [VThresh,VThresh+VWidth].  VWidth=0,VShift=0 is standard hard-thresholding
#ifdef PV_USE_OPENCL
   virtual int getNumCLEvents() {return numEvents;}
   virtual const char * getKernelName() { return "ANNLayer_update_state"; }
   virtual int initializeThreadBuffers(const char * kernel_name);
   virtual int initializeThreadKernels(const char * kernel_name);
   //virtual int getEVActivity() {return EV_ANN_ACTIVITY;}
   int updateStateOpenCL(double time, double dt);
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
