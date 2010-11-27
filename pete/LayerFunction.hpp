/*
 * LayerFunction.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef LAYERFUNCTION_HPP_
#define LAYERFUNCTION_HPP_

#include <string.h>
#include "../PetaVision/src/include/pv_types.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

class LayerFunction {
public:
    LayerFunction(const char * name);
    ~LayerFunction();
    virtual pvdata_t evaluate(float time, HyPerLayer * l) {return 0;}
    char * getName() {return name;}
    void setName(const char * name);

protected:
    char * name;
};

}  // end namespace PV

#endif /* LAYERFUNCTION_HPP_ */
