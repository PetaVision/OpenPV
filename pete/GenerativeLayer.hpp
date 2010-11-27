/*
 * GenerativeLayer.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVELAYER_HPP_
#define GENERATIVELAYER_HPP_

#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "GV1.hpp"

namespace PV {

class GenerativeLayer : public GV1 {
public:
	GenerativeLayer(const char * name, HyPerCol * hc);
	GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type);
    int initialize(PVLayerType type);

    pvdata_t getRelaxation() {return relaxation;}
    int updateState(float time, float dt);
    virtual pvdata_t sparsityterm(pvdata_t v) { return logf(1+v*v);}
    virtual pvdata_t sparsitytermderivative(pvdata_t v) { return 2.0*v/(1+v*v); }

private:
    pvdata_t relaxation; // V(new) = V(old) - relaxation*(gradient wrt V)
};

}  // end namespace PV block

#endif /* GENERATIVELAYER_HPP_ */
