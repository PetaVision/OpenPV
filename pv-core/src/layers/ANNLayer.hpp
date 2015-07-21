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

   virtual bool activityIsSpiking() { return false; }
protected:
   ANNLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);
   virtual int setActivity();
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   
   /** 
    * List of parameters used by the ANNLayer class
    * @name ANNLayer Parameters
    * @{
    */

   /**
    * @brief VThresh: The threshold value for the membrane potential.  Below this value, the
    * output activity will be AMin.  Above, it will obey the transfer function specified by
    * AMax, VWidth, and AShift.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief AMin: When membrane potential V is below the threshold VThresh, activity takes the value AMin.
    */
   virtual void ioParam_VMin(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief AMax: The maximum value of the output activity.
    */
   virtual void ioParam_VMax(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief AShift: When membrane potential V is above the threshold VThresh, activity is V-AShift
    * (but see VWidth for making a gradual transition at VThresh)
    */
   virtual void ioParam_VShift(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief VWidth: When the membrane potential is between VThresh and VThresh+VWidth, the activity changes linearly
    * between A=AMin when V=VThresh and A=VThresh+VWidth-AShift when V=VThresh+VWidth.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);
   /** @} */
   
   /**
    * @brief clearGSynInterval: the time interval after which GSyn is reset to zero.
    * @details Until this interval elapses, GSyn continues to accumulate from timestep to timestep.
    * If clearGSynInterval is zero or negative, GSyn clears every timestep.
    * If clearGSynInterval is infinite, the layer acts as an accumulator.
    */
   virtual void ioParam_clearGSynInterval(enum ParamsIOFlag ioFlag);

   virtual int checkVThreshParams(PVParams * params);

   virtual int resetGSynBuffers(double timef, double dt);

   virtual int checkpointRead(const char * cpDir, double * timeptr); // (const char * cpDir, double * timed);
   virtual int checkpointWrite(const char * cpDir);

   pvdata_t AMax;  // maximum membrane potential, larger values are set to AMax
   pvdata_t AMin;  // minimum membrane potential, smaller values are set to AMin
   pvdata_t VThresh;  // threshold potential, values smaller than VThresh are set to AMin
   pvdata_t AShift;  // shift potential, values above VThresh are shifted downward by this amount
                     // AShift == 0, hard threshold condition
                     // AShift == VThresh, soft threshold condition
   pvdata_t VWidth;  // The thresholding occurs linearly over the region [VThresh,VThresh+VWidth].  VWidth=0,AShift=0 is standard hard-thresholding
   double clearGSynInterval; // The interval between successive clears of GSyn
   double nextGSynClearTime; // The next time that the GSyn will be cleared.
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
