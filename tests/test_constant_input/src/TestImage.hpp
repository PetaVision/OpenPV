/*
 * TestImage.hpp
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#ifndef TESTIMAGE_HPP_
#define TESTIMAGE_HPP_

#include "../src/layers/HyPerLayer.hpp"

namespace PV {

class TestImage : public HyPerLayer {
  public:
   TestImage(const char *name, HyPerCol *hc);
   virtual int updateState(double timed, double dt);
   const float getConstantVal() { return val; }
   virtual bool activityIsSpiking() { return false; }
   virtual ~TestImage();

  protected:
   TestImage();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_constantVal(enum ParamsIOFlag ioFlag);
   virtual int allocateV();
   virtual int initializeActivity();

  private:
   int initialize_base();

   // Member variables
  private:
   float val;
};
}

#endif /* TESTIMAGE_HPP_ */
