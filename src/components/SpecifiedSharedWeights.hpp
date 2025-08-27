/*
 * SpecifiedSharedWeights.hpp
 */

#ifndef SPECIFIEDSHAREDWEIGHTS_HPP_
#define SPECIFIEDSHAREDWEIGHTS_HPP_

#include "components/SharedWeights.hpp"

namespace PV {

/**
 * A class template derived from SharedWeights that sets SharedWeightsFlag to a specific value
 * instead of reading from params
 */

template <bool B>
class SpecifiedSharedWeights : public SharedWeights {
  protected:
   /**
    * List of parameters needed from the SpecifiedSharedWeights class
    * @name SpecifiedSharedWeights Parameters
    * @{
    */

   /**
    * @brief sharedWeights: SpecifiedSharedWeights does not use the sharedWeights parameter,
    * but uses the same setting as the original connection.
    */
   virtual void ioParam_sharedWeights(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of SpecifiedSharedWeights parameters

  public:
   SpecifiedSharedWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
      initialize(paramsIO, comm);
   }

   virtual ~SpecifiedSharedWeights() {}

  protected:
   SpecifiedSharedWeights() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
      SharedWeights::initialize(paramsIO, comm);
   }

   virtual void setObjectType() override {
      mObjectType = "SpecifiedSharedWeights";
   }

}; // class template SpecifiedSharedWeights

template <bool B>
void SpecifiedSharedWeights<B>::ioParam_sharedWeights(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mSharedWeightsFlag = B;
      mParamsIO->handleUnnecessaryParameter("sharedWeights");
   }
}

} // namespace PV

#endif // SPECIFIEDSHAREDWEIGHTS_HPP_
