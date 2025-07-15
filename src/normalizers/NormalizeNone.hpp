/*
 * NormalizeNone.hpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#ifndef NORMALIZENONE_HPP_
#define NORMALIZENONE_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeNone : public NormalizeBase {
   // Member functions
  protected:
   virtual void ioParam_normalizeArborsIndividually(ParamsIOSwitch ioSwitch) override {}
   virtual void ioParam_normalizeOnInitialize(ParamsIOSwitch ioSwitch) override {}
   virtual void ioParam_normalizeOnWeightUpdate(ParamsIOSwitch ioSwitch) override {}

  public:
   NormalizeNone(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~NormalizeNone();

  protected:
   NormalizeNone();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // class NormalizeNone

} /* namespace PV */

#endif /* NORMALIZENONE_HPP_ */
