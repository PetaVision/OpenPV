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
   StrengthParam(char const *name, PVParams *params, Communicator const *comm);

   virtual ~StrengthParam();

   float getStrength() const { return mStrength; }

   /**
    * Finds the connection with the given name, and checks whether the connection has a
    * StrengthParam component. If it does, it returns the existing component. If it doesn't,
    * it creates a StrengthParam object and adds it to the connection, and returns the new object.
    * Note that the returned object belongs to the connection and should not be freed except by
    * the connection's destructor.
    *
    * It is a fatal error if there is no connection with the given name in the
    * CommunicateInitInfoMessage object. Intended to be called during the CommunicateInitInfo stage,
    * by components that need a StrengthParam object, without requiring those that do not need
    * a strength parameter to read it.
    */
   static StrengthParam *ensureExists(
         std::shared_ptr<CommunicateInitInfoMessage const> message,
         char const *name,
         PVParams *params,
         Communicator const *comm);

  protected:
   StrengthParam() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   float mStrength = 1.0f;
};

} // namespace PV

#endif // STRENGTHPARAM_HPP_
