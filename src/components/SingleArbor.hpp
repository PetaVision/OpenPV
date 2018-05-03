/*
 * SingleArbor.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef SINGLEARBOR_HPP_
#define SINGLEARBOR_HPP_

#include "components/ArborList.hpp"

namespace PV {

/**
 * A subclass of ArborList, where there is single axonal arbor, and the
 * numAxonalArbors parameter is not used (if present it must be equal to 1).
 */
class SingleArbor : public ArborList {
  protected:
   /**
    * List of parameters needed from the SingleArbor class
    * @name SingleArbor Parameters
    * @{
    */

   /**
    * @brief numAxonalArbors: SingleArbors does not use the numAxonalArbors parameter,
    * but sets the number of arbors to one.
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of SingleArbor parameters

  public:
   SingleArbor(char const *name, HyPerCol *hc);
   virtual ~SingleArbor();

   virtual void setObjectType() override;

  protected:
   SingleArbor();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

}; // class SingleArbor

} // namespace PV

#endif // SINGLEARBOR_HPP_
