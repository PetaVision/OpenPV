/*
 * ANNSquaredLayer.hpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#ifndef ANNSQUAREDLAYER_HPP__
#define ANNSQUAREDLAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

/**
 * ANNSquaredLayer
 * HyPerLayer subclass that squares the excitatory input (using SquaredInternalStateBuffer)
 * and then applies a thresholding transfer function (using ANNActivityBuffer).
 */
class ANNSquaredLayer : public HyPerLayer {
  public:
   ANNSquaredLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~ANNSquaredLayer();

  protected:
   ANNSquaredLayer() {}

   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;

   virtual Response::Status allocateDataStructures() override;
};

} // end namespace PV

#endif
