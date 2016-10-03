/*
 * InitGaborWeightsParams.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTSPARAMS_HPP_
#define INITGABORWEIGHTSPARAMS_HPP_

#include <weightinit/InitGauss2DWeightsParams.hpp>
#include <weightinit/InitWeightsParams.hpp>

namespace PV {

class InitGaborWeightsParams : public PV::InitGauss2DWeightsParams {
  public:
   InitGaborWeightsParams();
   InitGaborWeightsParams(const char *name, HyPerCol *hc);
   virtual ~InitGaborWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);

   // get/set methods:
   inline float getlambda() { return lambda; }
   inline float getphi() { return phi; }
   inline bool getinvert() { return invert; }

  protected:
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_lambda(enum ParamsIOFlag ioFlag);
   virtual void ioParam_phi(enum ParamsIOFlag ioFlag);
   virtual void ioParam_invert(enum ParamsIOFlag ioFlag);

  private:
   // params variables:
   int lambda;
   float phi;
   bool invert;
};

} /* namespace PV */
#endif /* INITGABORWEIGHTSPARAMS_HPP_ */
