/*
 * GV1.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 *
 *  Subclass of V1 to override recvSynapticInput to include negative weights
 */

#ifndef GV1_HPP_
#define GV1_HPP_

#include <assert.h>
#include "../PetaVision/src/layers/V1.hpp"

namespace PV {
class GV1 : public V1 {
public:
    GV1(const char* name, HyPerCol * hc);
    GV1(const char* name, HyPerCol * hc, PVLayerType type);

    virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
}; // end class GV1

}  // end namespace PV

#endif /* GV1_HPP_ */
