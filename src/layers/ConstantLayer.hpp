/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#ifndef CONSTANTLAYER_HPP_
#define CONSTANTLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class ConstantLayer : public HyPerLayer {
  protected:
   /**
    * List of parameters needed from the ConstantLayer class
    * @name HyPerLayer Parameters
    * @{
    */

  public:
   ConstantLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~ConstantLayer();

  protected:
   ConstantLayer();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   LayerUpdateController *createLayerUpdateController() override;

}; // class ConstantLayer

} /* namespace PV */
#endif
