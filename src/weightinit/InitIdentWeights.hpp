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
   virtual void ioParam_weightInit(ParamsIOSwitch ioSwitch) override;

  public:
   InitIdentWeights(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~InitIdentWeights();

  protected:
   InitIdentWeights();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
}; // class InitIdentWeights

} /* namespace PV */
#endif /* INITIDENTWEIGHTS_HPP_ */
