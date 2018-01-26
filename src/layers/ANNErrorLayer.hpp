/*
 * ANNErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNERRORLAYER_HPP_
#define ANNERRORLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNErrorLayer : public PV::ANNLayer {
  public:
   ANNErrorLayer(const char *name, HyPerCol *hc);
   virtual ~ANNErrorLayer();

  protected:
   ANNErrorLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters used by the ANNErrorLayer class
    * @name ANNErrorLayer Parameters
    * @{
    */

   /**
    * @brief: errScale: The input to the error layer is multiplied by errScale before applying the
    * threshold.
    */
   virtual void ioParam_errScale(enum ParamsIOFlag ioFlag);

   /**
    * @brief VThresh: Errors whose absolute value is below VThresh are truncated to zero.
    * @detail If VThresh is negative, no truncation takes place.  errScale is applied before
    * VThresh.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag) override {
      ANNLayer::ioParam_VThresh(ioFlag);
      return;
   }

   /**
    * @brief ANNErrorLayer does not use AMin.
    */
   virtual void ioParam_AMin(enum ParamsIOFlag ioFlag) override {}

   /**
    * @brief ANNErrorLayer does not use AMax.
    */
   virtual void ioParam_AMax(enum ParamsIOFlag ioFlag) override {}

   /**
    * @brief ANNErrorLayer does not use AShift.
    */
   virtual void ioParam_AShift(enum ParamsIOFlag ioFlag) override {}

   /**
    * @brief ANNErrorLayer does not use VWidth.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag) override {}
   /** @} */

   virtual int setVertices() override;
   virtual int checkVertices() const override;
   virtual Response::Status updateState(double time, double dt) override;

  private:
   int initialize_base();
   float errScale;
}; // class ANNErrorLayer

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
