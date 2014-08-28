/*
 * ANNDivInh.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: kpeterson
 */

#ifndef ANNDIVINH_HPP_
#define ANNDIVINH_HPP_

#include "ANNLayer.hpp"

#define NUM_ANNDV_EVENTS   4
#define EV_ANNDV_GSYN_IB   2
//#define EV_ANNDV_ACTIVITY  3

namespace PV {

class ANNDivInh: public PV::ANNLayer {
public:
   ANNDivInh(const char* name, HyPerCol * hc);
   virtual ~ANNDivInh();

   virtual int updateState(double time, double dt);
   // virtual int updateV();

protected:
   ANNDivInh();
   int initialize(const char * name, HyPerCol * hc);

//#ifdef PV_USE_OPENCL
//   virtual int getNumCLEvents() {return numEvents;}
//   virtual const char * getKernelName() { return "ANNDivLayer_update_state"; }
//   virtual int initializeThreadBuffers(const char * kernel_name);
//   virtual int initializeThreadKernels(const char * kernel_name);
//   virtual int getEVGSynIB() {return EV_ANNDV_GSYN_IB;}
//   virtual inline int getGSynEvent(ChannelType ch) {
//      if(HyPerLayer::getGSynEvent(ch)>=0) return HyPerLayer::getGSynEvent(ch);
//      if(ch==CHANNEL_INHB) return getEVGSynIB();
//      return -1;
//   }
//   //virtual int getEVActivity() {return EV_ANNDV_ACTIVITY;}
//   int updateStateOpenCL(double time, double dt);
//   //virtual int triggerReceive(InterColComm* comm);
//#endif

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* ANNDIVINH_HPP_ */
