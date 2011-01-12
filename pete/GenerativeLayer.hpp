/*
 * GenerativeLayer.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVELAYER_HPP_
#define GENERATIVELAYER_HPP_

#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "../PetaVision/src/layers/NonspikingLayer.hpp"

namespace PV {

class GenerativeLayer : public NonspikingLayer {
public:
	GenerativeLayer(const char * name, HyPerCol * hc);
	GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type);
    int initialize();

    pvdata_t getRelaxation() {return relaxation;}
    pvdata_t getActivityThreshold() { return activityThreshold; }
    virtual pvdata_t sparsityterm(pvdata_t v) { return logf(1+v*v);}
    virtual pvdata_t sparsitytermderivative(pvdata_t v) { return 2.0*v/(1+v*v); }

protected:
    int updateV();
    int setActivity();

private:
    pvdata_t relaxation; // V(new) = V(old) - relaxation*(gradient wrt V)
    pvdata_t activityThreshold;
};

}  // end namespace PV block

#endif /* GENERATIVELAYER_HPP_ */
