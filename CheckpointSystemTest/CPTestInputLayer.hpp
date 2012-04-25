/*
 * CPTestInputLayer.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef CPTESTINPUTLAYER_HPP_
#define CPTESTINPUTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class CPTestInputLayer : public ANNLayer {

public:
   CPTestInputLayer(const char * name, HyPerCol * hc);
   virtual ~CPTestInputLayer();
   virtual int updateState(float time, float dt);

protected:
   int initialize();
   virtual int initializeV();

#ifdef PV_USE_OPENCL

protected:
   virtual int getNumCLEvents() {return numEvents;}
   virtual const char * getKernelName() { return "ANNLayer_update_state"; }
   virtual int initializeThreadBuffers(const char * kernel_name);
   virtual int initializeThreadKernels(const char * kernel_name);
   int updateStateOpenCL(float time, float dt);

#endif // PV_USE_OPENCL

}; // end class CPTestInputLayer

}  // end of namespace PV block


#endif /* CPTESTINPUTLAYER_HPP_ */
