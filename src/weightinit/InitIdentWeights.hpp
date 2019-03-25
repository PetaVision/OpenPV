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

class InitIdentWeights : public InitOneToOneWeights {
  protected:
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag) override;

  public:
   InitIdentWeights(char const *name, PVParams *params, Communicator const *comm);
   virtual ~InitIdentWeights();

  protected:
   InitIdentWeights();
   void initialize(char const *name, PVParams *params, Communicator const *comm);
}; // class InitIdentWeights

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
