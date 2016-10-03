/*
 * InitIdentWeightsParams.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITIDENTWEIGHTSPARAMS_HPP_
#define INITIDENTWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitIdentWeightsParams : public PV::InitWeightsParams {
  public:
   InitIdentWeightsParams();
   InitIdentWeightsParams(const char *name, HyPerCol *hc);
   virtual ~InitIdentWeightsParams();
   void calcOtherParams(int patchIndex);

  protected:
   virtual int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
};

} /* namespace PV */
#endif /* INITIDENTWEIGHTSPARAMS_HPP_ */
