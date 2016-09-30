/*
 * ANNSquaredLayer.hpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#ifndef ANNSQUAREDLAYER_HPP_
#define ANNSQUAREDLAYER_HPP_

#include "ANNLayer.hpp"

#define NUM_ANNSQ_EVENTS 3

namespace PV {

class ANNSquaredLayer : public PV::ANNLayer {
  public:
   ANNSquaredLayer(const char *name, HyPerCol *hc);
   virtual ~ANNSquaredLayer();
   virtual int updateState(double time, double dt);

  protected:
   ANNSquaredLayer();
   int initialize(const char *name, HyPerCol *hc);

  private:
   int initialize_base();

}; // class ANNSquaredLayer

} /* namespace PV */
#endif /* ANNSQUAREDLAYER_HPP_ */
