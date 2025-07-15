/*
 * NormalizeGroup.hpp
 *
 *  Created on: Jun 22, 2016
 *      Author: pschultz
 */

#ifndef NORMALIZEGROUP_HPP_
#define NORMALIZEGROUP_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeGroup : public NormalizeBase {
  public:
   NormalizeGroup(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~NormalizeGroup();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   NormalizeGroup();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   /**
    * NormalizeGroup does not read the normalizeArborsIndividually parameter, but inherits it from
    * its group head.
    */
   virtual void ioParam_normalizeArborsIndividually(ParamsIOSwitch ioSwitch) override;

   /**
    * NormalizeGroup does not read the normalizeOnInitialize parameter, but inherits it from its
    * group head.
    */
   virtual void ioParam_normalizeOnInitialize(ParamsIOSwitch ioSwitch) override;

   /**
    * NormalizeGroup does not read the normalizeOnWeightUpdate parameter, but inherits it from its
    * group head.
    */
   virtual void ioParam_normalizeOnWeightUpdate(ParamsIOSwitch ioSwitch) override;

   /**
    * The name of the normalizer that serves as the normalizer group head.
    * The group head cannot itself be a NormalizeGroup.
    */
   virtual void ioParam_normalizeGroupName(ParamsIOSwitch ioSwitch);

   /**
    * Overrides normalizeWeights to do nothing.
    * Instead, when the group head's normalizeWeights method is called,
    * the weights of all connections in the group are normalized together.
    */
   virtual int normalizeWeights() override;

   // Data members
  private:
   std::string mNormalizeGroupName;
   NormalizeBase *mGroupHead = nullptr;
}; // class NormalizeGroup

} /* namespace PV */

#endif /* NORMALIZEGROUP_HPP_ */
