/*
 * NonspikingLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP_
#define ANNLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class ANNLayer : public HyPerLayer {
public:
    ANNLayer(const char* name, HyPerCol * hc);
    ~ANNLayer();
    virtual int updateV();
    pvdata_t VThresh;
    pvdata_t VMax;
    pvdata_t VMin;
protected:
    int initialize();
}; // end of class ANNLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
