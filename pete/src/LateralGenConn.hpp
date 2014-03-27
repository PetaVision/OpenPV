/*
 * LateralGenConn.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef LATERALGENCONN_HPP_
#define LATERALGENCONN_HPP_

#include <connections/GenerativeConn.hpp>
#include <string.h>

namespace PV {

class LateralGenConn : public GenerativeConn {
public:
    LateralGenConn();
    LateralGenConn(const char * name, HyPerCol * hc);
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc);
    virtual int communicateInitInfo();
    virtual int updateWeights(int axonID);

protected:
    PVPatch *** initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
          int numPatches, const char * filename);
};

}  // end of block for namespace PV

#endif /* LATERALGENCONN_HPP_ */
