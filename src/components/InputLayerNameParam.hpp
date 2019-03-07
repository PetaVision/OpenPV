/*
 * InputLayerNameParam.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef INPUTLAYERNAMEPARAM_HPP_
#define INPUTLAYERNAMEPARAM_HPP_

#include "components/LinkedObjectParam.hpp"

namespace PV {

/**
 * A component to contain the inputLayerName param, used by FilenameParsingGroundTruthLayer.
 */
class InputLayerNameParam : public LinkedObjectParam {
  public:
   InputLayerNameParam(char const *name, PVParams *params, Communicator const *comm);

   virtual ~InputLayerNameParam();

  protected:
   InputLayerNameParam() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // INPUTLAYERNAMEPARAM_HPP_
