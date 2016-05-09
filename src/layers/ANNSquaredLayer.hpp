/*
 * ANNSquaredLayer.hpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#ifndef ANNSQUAREDLAYER_HPP_
#define ANNSQUAREDLAYER_HPP_

#include "ANNLayer.hpp"

#define NUM_ANNSQ_EVENTS   3
//#define EV_ANNSQ_ACTIVITY  2

namespace PV {

class ANNSquaredLayer: public PV::ANNLayer {
public:
   ANNSquaredLayer(const char* name, HyPerCol * hc);
   virtual ~ANNSquaredLayer();
   virtual int updateState(double time, double dt);

protected:
   ANNSquaredLayer();
   int initialize(const char * name, HyPerCol * hc);

//#ifdef PV_USE_OPENCL
//   virtual int getNumCLEvents() {return numEvents;}
//   virtual const char * getKernelName() { return "ANNSquaredLayer_update_state"; }
//   virtual int initializeThreadBuffers(const char * kernel_name);
//   virtual int initializeThreadKernels(const char * kernel_name);
//   //virtual int getEVActivity() {return EV_ANNSQ_ACTIVITY;}
//   int updateStateOpenCL(double time, double dt);
//#endif

private:
   int initialize_base();

}; // class ANNSquaredLayer

BaseObject * createANNSquaredLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* ANNSQUAREDLAYER_HPP_ */
