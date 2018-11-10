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
   TestImage(const char *name, HyPerCol *hc);
   virtual ~TestImage();

   float getConstantVal() const;

  protected:
   TestImage();
   int initialize(const char *name, HyPerCol *hc);
   virtual ActivityComponent *createActivityComponent() override;

}; // class TestImage

} // namespace PV

#endif /* TESTIMAGE_HPP_ */
