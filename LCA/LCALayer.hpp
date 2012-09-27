/*
 * LCALayer.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: pschultz
 */

#ifndef LCALAYER_HPP_
#define LCALAYER_HPP_

#include <src/layers/HyPerLayer.hpp>

namespace PV {

class LCALayer : public HyPerLayer {

// Methods
public:
   LCALayer(const char * name, HyPerCol * hc, int num_channels=MAX_CHANNELS);
   virtual ~LCALayer();
   virtual int updateState(float timef, float dt);

protected:
   LCALayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels);

private:
   int initialize_base();

// Member variables
protected:
   float thresholdSoftness;
}; // class LCALayer

} /* namespace PV */
#endif /* LCALAYER_HPP_ */
