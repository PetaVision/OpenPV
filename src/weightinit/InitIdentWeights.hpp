/*
 * InitIdentWeights.hpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#ifndef INITIDENTWEIGHTS_HPP_
#define INITIDENTWEIGHTS_HPP_

#include "InitOneToOneWeights.hpp"

namespace PV {

class InitIdentWeights : public PV::InitOneToOneWeights {
  protected:
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag) override;

  public:
   InitIdentWeights(char const *name, HyPerCol *hc);
   virtual ~InitIdentWeights();

  protected:
   InitIdentWeights();
   int initialize(char const *name, HyPerCol *hc);
}; // class InitIdentWeights

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
