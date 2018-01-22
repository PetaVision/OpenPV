/*
 * OriginalConnNameParam.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef ORIGINALCONNNAMEPARAM_HPP_
#define ORIGINALCONNNAMEPARAM_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the originalConnName param, used by connection
 * types (TransposeConn, CloneConn, etc.) that are dependent on another connection.
 * patch size. The dimensions are read from the originalConnName parameter, and
 * retrieved using the getOriginalConnName() method.
 */
class OriginalConnNameParam : public BaseObject {
  protected:
   /**
    * List of parameters needed from the OriginalConnNameParam class
    * @name OriginalConnNameParam Parameters
    * @{
    */

   /**
    * @brief originalConnName: String parameter. It cannot be null or empty,
    * and must point to another connection in the hierarchy.
    */
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of OriginalConnNameParam parameters

  public:
   OriginalConnNameParam(char const *name, HyPerCol *hc);

   virtual ~OriginalConnNameParam();

   char const *getOriginalConnName() const { return mOriginalConnName; }

  protected:
   OriginalConnNameParam() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   char *mOriginalConnName = nullptr;
};

} // namespace PV

#endif // ORIGINALCONNNAMEPARAM_HPP_
