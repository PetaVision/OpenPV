/*
 * LabelErrorLayer.hpp
 *
 *  Created on: Nov 30, 2013
 *      Author: garkenyon
 */

#ifndef LABELERRORLAYER_HPP_
#define LABELERRORLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class LabelErrorLayer : public PV::ANNLayer {
  public:
   LabelErrorLayer(const char *name, HyPerCol *hc);
   virtual ~LabelErrorLayer();

  protected:
   LabelErrorLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;
   void ioParam_errScale(enum ParamsIOFlag ioFlag);
   void ioParam_isBinary(enum ParamsIOFlag ioFlag);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
   float errScale;
   int isBinary;
}; // class LabelErrorLayer

} /* namespace PV */
#endif /* LABELERRORLAYER_HPP_ */
