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
 */
class OriginalConnNameParam : public LinkedObjectParam {
  public:
   OriginalConnNameParam(char const *name, PVParams *params, Communicator *comm);

   virtual ~OriginalConnNameParam();

  protected:
   OriginalConnNameParam() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // ORIGINALCONNNAMEPARAM_HPP_
