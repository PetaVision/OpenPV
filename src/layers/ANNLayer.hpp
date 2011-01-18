/*
 * NonspikingLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP_
#define ANNLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class ANNLayer : public HyPerLayer {
public:
    ANNLayer(const char* name, HyPerCol * hc);
    ~ANNLayer();
protected:
    int initialize();
}; // end of class NonspikingLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
