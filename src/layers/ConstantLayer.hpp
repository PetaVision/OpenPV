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
   ConstantLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ConstantLayer();

  protected:
   ConstantLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   LayerUpdateController *createLayerUpdateController() override;
   LayerOutputComponent *createLayerOutput() override;

}; // class ConstantLayer

} /* namespace PV */
#endif
