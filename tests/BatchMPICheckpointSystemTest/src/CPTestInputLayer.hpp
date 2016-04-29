/*
 * CPTestInputLayer.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef CPTESTINPUTLAYER_HPP_
#define CPTESTINPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class CPTestInputLayer : public ANNLayer {

public:
   CPTestInputLayer(const char * name, HyPerCol * hc);
   virtual ~CPTestInputLayer();
   virtual int allocateDataStructures();
   virtual int updateState(double timed, double dt);

protected:
   int initialize(const char * name, HyPerCol * hc);
   virtual int initializeV();

#ifdef PV_USE_OPENCL

protected:
   //virtual int getNumCLEvents() {return numEvents;}
   virtual const char * getKernelName() { return "ANNLayer_update_state"; }
   //virtual int initializeThreadBuffers(const char * kernel_name);
   //virtual int initializeThreadKernels(const char * kernel_name);
   int updateStateOpenCL(double timed, double dt);

#endif // PV_USE_OPENCL

}; // end class CPTestInputLayer

BaseObject * createCPTestInputLayer(char const * name, HyPerCol * hc);

}  // end of namespace PV block


#endif /* CPTESTINPUTLAYER_HPP_ */
