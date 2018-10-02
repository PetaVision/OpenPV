/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include "columns/HyPerCol.hpp"
#include "weightupdaters/MomentumUpdater.hpp"

namespace PV {

MomentumConn::MomentumConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

MomentumConn::MomentumConn() {}

MomentumConn::~MomentumConn() {}

int MomentumConn::initialize(char const *name, HyPerCol *hc) {
   return HyPerConn::initialize(name, hc);
}

BaseWeightUpdater *MomentumConn::createWeightUpdater() { return new MomentumUpdater(name, parent); }

} // namespace PV
