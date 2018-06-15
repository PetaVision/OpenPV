/*
 * OriginalConnNameParam.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef ORIGINALCONNNAMEPARAM_HPP_
#define ORIGINALCONNNAMEPARAM_HPP_

#include "components/LinkedObjectParam.hpp"

namespace PV {

/**
 * A component to contain the originalConnName param, used by connection
 * types (TransposeConn, CloneConn, etc.) that are dependent on another connection.
 * patch size. The dimensions are read from the originalConnName parameter, and
 * retrieved using the getOriginalConnName() method.
 */
class OriginalConnNameParam : public LinkedObjectParam {
  public:
   OriginalConnNameParam(char const *name, HyPerCol *hc);

   virtual ~OriginalConnNameParam();

  protected:
   OriginalConnNameParam() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // ORIGINALCONNNAMEPARAM_HPP_
