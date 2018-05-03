/*
 * StrengthParam.hpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#ifndef STRENGTHPARAM_HPP_
#define STRENGTHPARAM_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the strength flag from parameters.
 * The strength is read from the strength floating-point parameter, and retrieved using the
 * getStrength() method.
 */
class StrengthParam : public BaseObject {
  protected:
   /**
    * List of parameters needed from the StrengthParam class
    * @name StrengthParam Parameters
    * @{
    */

   /**
    * @brief strength: specifies the value of the strength parameter. The NormalizeBase and
    * InitGauss2DWeights classes use this component to specify the strength of a connection.
    */
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);

   /** @} */ // end of StrengthParam parameters

  public:
   StrengthParam(char const *name, HyPerCol *hc);

   virtual ~StrengthParam();

   float getStrength() const { return mStrength; }

  protected:
   StrengthParam() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   float mStrength = 1.0f;
};

} // namespace PV

#endif // STRENGTHPARAM_HPP_
