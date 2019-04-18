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
   virtual Response::Status updateState(double timed, double dt) override;
   const float getConstantVal() { return val; }
   virtual bool activityIsSpiking() override { return false; }
   virtual ~TestImage();

  protected:
   TestImage();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_constantVal(enum ParamsIOFlag ioFlag);
   virtual void allocateV() override;
   virtual void initializeActivity() override;

  private:
   int initialize_base();

   // Member variables
  private:
   float val;
};
}

#endif /* TESTIMAGE_HPP_ */
