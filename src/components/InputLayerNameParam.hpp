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
 * A component to contain the inputLayerName param, used by FilenameParsingLayer.
 */
class InputLayerNameParam : public LinkedObjectParam {
  public:
   InputLayerNameParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~InputLayerNameParam();

  protected:
   InputLayerNameParam() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;
};

} // namespace PV

#endif // INPUTLAYERNAMEPARAM_HPP_
