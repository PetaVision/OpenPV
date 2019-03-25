/*
 * SharedWeights.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef SHAREDWEIGHTS_HPP_
#define SHAREDWEIGHTS_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the sharedWeights flag from parameters.
 * patch size. The dimensions are read from the sharedWeights parameter, and
 * retrieved using the getSharedWeights() method.
 */
class SharedWeights : public BaseObject {
  protected:
   /**
    * List of parameters needed from the SharedWeights class
    * @name SharedWeights Parameters
    * @{
    */

   /**
    * @brief sharedWeights: Boolean, defines if the weights use shared weights or not.
    * Defaults to true (shared weights).
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);

   /** @} */ // end of SharedWeights parameters

  public:
   SharedWeights(char const *name, PVParams *params, Communicator const *comm);

   virtual ~SharedWeights();

   bool getSharedWeights() const { return mSharedWeights; }

  protected:
   SharedWeights() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   bool mSharedWeights = true;
};

} // namespace PV

#endif // SHAREDWEIGHTS_HPP_
