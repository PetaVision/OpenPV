/*
 * ANNWeightedErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNWEIGHTEDERRORLAYER_HPP_
#define ANNWEIGHTEDERRORLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

  class ANNWeightedErrorLayer: public PV::ANNLayer {
  public:
    ANNWeightedErrorLayer(const char * name, HyPerCol * hc);
    virtual ~ANNWeightedErrorLayer();
  protected:
    ANNWeightedErrorLayer();
    int initialize(const char * name, HyPerCol * hc);
    int allocateDataStructures();
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_errScale(enum ParamsIOFlag ioFlag);
    virtual void ioParam_errWeightsFileName(enum ParamsIOFlag ioFlag);
    virtual int updateState(double time, double dt);
  private:
    int initialize_base();
    float errScale;
    float * errWeights;
    char * errWeightsFileName;

  };

} /* namespace PV */
#endif /* ANNWEIGHTEDERRORLAYER_HPP_ */
