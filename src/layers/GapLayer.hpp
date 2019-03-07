/*
 * GapLayer.hpp
 * can be used to implement gap junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef GAPLAYER_HPP_
#define GAPLAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

/**
 * GapLayer can be used to implement gap junctions
 */
class GapLayer : public CloneVLayer {
  public:
   GapLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~GapLayer();

  protected:
   GapLayer() {}

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif /* GAPLAYER_HPP_ */
