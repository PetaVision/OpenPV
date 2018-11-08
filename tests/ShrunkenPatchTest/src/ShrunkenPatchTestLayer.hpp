/*
 * ShrunkenPatchTestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef SHRUNKENPATCHTESTLAYER_HPP_
#define SHRUNKENPATCHTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ShrunkenPatchTestLayer : public PV::ANNLayer {
  public:
   ShrunkenPatchTestLayer(const char *name, PVParams *params, Communicator *comm);
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double time, double dt) override;
   virtual int publish(Communicator *comm, double timed) override;
   int setVtoGlobalPos();
   int setActivitytoGlobalPos();

  private:
   void initialize(const char *name, PVParams *params, Communicator *comm);

}; // end class ShrunkenPatchTestLayer

} /* namespace PV */
#endif /* SHRUNKENPATCHTESTLAYER_HPP_ */
