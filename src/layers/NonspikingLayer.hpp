/*
 * NonspikingLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef NONSPIKINGLAYER_HPP_
#define NONSPIKINGLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class NonspikingLayer : public HyPerLayer {
public:
    NonspikingLayer(const char* name, HyPerCol * hc);
    ~NonspikingLayer();
protected:
    int initialize();
}; // end of class NonspikingLayer

}  // end namespace PV

#endif /* NONSPIKINGLAYER_HPP_ */
