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

// Old member variables deprecated on Mar 20, 2014
#define VMin AMin
#define VMax AMax
#define VShift AShift

namespace PV {

class ANNLayer : public HyPerLayer {
public:
   ANNLayer(const char* name, HyPerCol * hc);
   virtual ~ANNLayer();
   pvdata_t getVThresh()        { return VThresh; }
   pvdata_t getAMax()           { return AMax; }
   pvdata_t getAMin()           { return AMin; }
   pvdata_t getAShift()         { return AShift; }
   pvdata_t getVWidth()         { return VWidth; }
protected:
   ANNLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);
   virtual int setActivity();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VShift(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);
   virtual int checkVThreshParams(PVParams * params);
   pvdata_t AMax;  // maximum membrane potential, larger values are set to AMax
   pvdata_t AMin;  // minimum membrane potential, smaller values are set to AMin
   pvdata_t VThresh;  // threshold potential, values smaller than VThresh are set to AMin
   pvdata_t AShift;  // shift potential, values above VThresh are shifted downward by this amount
                     // AShift == 0, hard threshold condition
                     // AShift == VThresh, soft threshold condition
   pvdata_t VWidth;  // The thresholding occurs linearly over the region [VThresh,VThresh+VWidth].  VWidth=0,AShift=0 is standard hard-thresholding
//#ifdef PV_USE_OPENCL
//   virtual int getNumCLEvents() {return numEvents;}
//   virtual const char * getKernelName() { return "ANNLayer_update_state"; }
//   virtual int initializeThreadBuffers(const char * kernel_name);
//   virtual int initializeThreadKernels(const char * kernel_name);
//   //virtual int getEVActivity() {return EV_ANN_ACTIVITY;}
//   int updateStateOpenCL(double time, double dt);
//   //temporary method for debuging recievesynapticinput
//public:
////   virtual void copyChannelExcFromDevice() {
////      getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(&evList[getEVGSynE()]);
////      clWaitForEvents(1, &evList[getEVGSynE()]);
////      clReleaseEvent(evList[getEVGSynE()]);
////   }
//protected:
//#endif // PV_USE_OPENCL

private:
   int initialize_base();
}; // end of class ANNLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
