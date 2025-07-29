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
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of SpecifiedSharedWeights parameters

  public:
   SpecifiedSharedWeights(char const *name, PVParams *params, Communicator const *comm) {
      initialize(name, params, comm);
   }

   virtual ~SpecifiedSharedWeights() {}

  protected:
   SpecifiedSharedWeights() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm) {
      SharedWeights::initialize(name, params, comm);
   }

   virtual void setObjectType() override {
      mObjectType = "SpecifiedSharedWeights";
   }

}; // class template SpecifiedSharedWeights

template <bool B>
void SpecifiedSharedWeights<B>::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeightsFlag = B;
      parameters()->handleUnnecessaryParameter(getName(), "sharedWeights");
   }
}

} // namespace PV

#endif // SPECIFIEDSHAREDWEIGHTS_HPP_
