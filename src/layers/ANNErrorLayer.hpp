/*
 * ANNErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNERRORLAYER_HPP__
#define ANNERRORLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * HyPerLayer subclass that applies a thresholding transfer function,
 * where |V|<threshold -> A=0 and |V|>threshold -> A=V.
 */
class ANNErrorLayer : public HyPerLayer {
  public:
   ANNErrorLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ANNErrorLayer();

  protected:
   ANNErrorLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
