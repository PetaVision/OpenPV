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
   ANNErrorLayer(const char *name, HyPerCol *hc);
   virtual ~ANNErrorLayer();

  protected:
   ANNErrorLayer() {}

   int initialize(const char *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif
