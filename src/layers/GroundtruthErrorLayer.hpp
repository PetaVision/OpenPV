#ifndef _GROUNDTRUTHERRORLAYER_HPP_
#define _GROUNDTRUTHERRORLAYER_HPP_

#include "HyPerLayer.hpp"

using namespace PV;

namespace PV {

class GroundtruthErrorLayer : public HyPerLayer {
 public:
  GroundtruthErrorLayer(const char *name, HyPerCol *hc);
  virtual ~GroundtruthErrorLayer();

 protected:
  GroundtruthErrorLayer();
  int initialize(const char *name, HyPerCol *hc);
  virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

  virtual void ioParam_thresholdCorrect(enum ParamsIOFlag ioFlag);
  virtual void ioParam_thresholdIncorrect(enum ParamsIOFlag ioFlag);
  virtual int updateState(double time, double dt);

 private:
  int initialize_base();
  float thresholdCorrect;
	float thresholdIncorrect;
};
}

#endif  // _GROUNDTRUTHERRORLAYER_HPP_
