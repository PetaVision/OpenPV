/*
 * TestImage.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef TESTIMAGE_HPP_
#define TESTIMAGE_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {

class TestImage : public PV::HyPerLayer {
  public:
   TestImage(const char *name, PVParams *params, Communicator *comm);
   virtual ~TestImage();

   float getConstantVal() const;

  protected:
   TestImage();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual ActivityComponent *createActivityComponent() override;

}; // class TestImage

} // namespace PV

#endif /* TESTIMAGE_HPP_ */
