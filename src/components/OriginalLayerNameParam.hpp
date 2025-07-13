/*
 * OriginalLayerNameParam.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef ORIGINALLAYERNAMEPARAM_HPP_
#define ORIGINALLAYERNAMEPARAM_HPP_

#include "components/LinkedObjectParam.hpp"

namespace PV {

/**
 * A component to contain the originalLayerName param, used by layer
 * types (CloneVLayer, RescaleLayer, etc.) that are dependent on another layer.
 */
class OriginalLayerNameParam : public LinkedObjectParam {
  public:
   OriginalLayerNameParam(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~OriginalLayerNameParam();

  protected:
   OriginalLayerNameParam() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // ORIGINALLAYERNAMEPARAM_HPP_
