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
   NormalizeGroup(char const *name, HyPerCol *hc);
   virtual ~NormalizeGroup();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   NormalizeGroup();
   int initialize(char const *name, HyPerCol *hc);

   /**
    * NormalizeGroup does not read the strength parameter, but inherits it from its group head.
    */
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag) override;

   /**
    * NormalizeGroup does not read the normalizeArborsIndividually parameter, but inherits it from
    * its group head.
    */
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) override;

   /**
    * NormalizeGroup does not read the normalizeOnInitialize parameter, but inherits it from its
    * group head.
    */
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) override;

   /**
    * NormalizeGroup does not read the normalizeOnWeightUpdate parameter, but inherits it from its
    * group head.
    */
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) override;

   /**
    * The name of the normalizer that serves as the normalizer group head.
    * The group head cannot itself be a NormalizeGroup.
    */
   virtual void ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag);

   /**
    * Overrides normalizeWeights to do nothing.
    * Instead, when the group head's normalizeWeights method is called,
    * the weights of all connections in the group are normalized together.
    */
   virtual int normalizeWeights() override;

  private:
   int initialize_base();

   // Data members
  private:
   char *normalizeGroupName = nullptr;
   NormalizeBase *groupHead = nullptr;
}; // class NormalizeGroup

} /* namespace PV */

#endif /* NORMALIZEGROUP_HPP_ */
