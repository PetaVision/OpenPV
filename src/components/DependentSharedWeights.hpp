/*
 * DependentSharedWeights.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef DEPENDENTSHAREDWEIGHTS_HPP_
#define DEPENDENTSHAREDWEIGHTS_HPP_

#include "components/SharedWeights.hpp"

namespace PV {

/**
 * A subclass of SharedWeights, which retrieves the sharedWeights flag from the connection
 * named in an OriginalConnNameParam component, instead of reading it from params.
 */
class DependentSharedWeights : public SharedWeights {
  protected:
   /**
    * List of parameters needed from the DependentSharedWeights class
    * @name DependentSharedWeights Parameters
    * @{
    */

   /**
    * @brief shareeWeihgts: DependentSharedWeightss does not use the sharedWeights parameter,
    * but uses the same setting as the original connection.
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of DependentSharedWeights parameters

  public:
   DependentSharedWeights(char const *name, HyPerCol *hc);
   virtual ~DependentSharedWeights();

   virtual void setObjectType() override;

  protected:
   DependentSharedWeights();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   char const *getOriginalConnName(std::map<std::string, Observer *> const hierarchy) const;
   SharedWeights *getOriginalSharedWeights(
         std::map<std::string, Observer *> const hierarchy,
         char const *originalConnName) const;

}; // class DependentSharedWeights

} // namespace PV

#endif // DEPENDENTSHAREDWEIGHTS_HPP_
