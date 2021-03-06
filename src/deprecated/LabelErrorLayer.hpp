/*
 * LabelErrorLayer.hpp
 *
 *  Created on: Nov 30, 2013
 *      Author: garkenyon
 */

// LabelErrorLayer was deprecated on Aug 15, 2018.

#ifndef LABELERRORLAYER_HPP_
#define LABELERRORLAYER_HPP_

#include "layers/ANNLayer.hpp"

namespace PV {

class LabelErrorLayer : public ANNLayer {
  public:
   LabelErrorLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~LabelErrorLayer();

  protected:
   LabelErrorLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
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
