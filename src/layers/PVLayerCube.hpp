/*
 * PVLayer.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef PVLAYERCUBE_H_
#define PVLAYERCUBE_H_

#include "include/pv_common.h"
#include "include/pv_types.h"
#include "utils/conversions.h"

namespace PV {

PVLayerCube *pvcube_new(const PVLayerLoc *loc, int numItems);
PVLayerCube *pvcube_init(PVLayerCube *cube, const PVLayerLoc *loc, int numItems);
int pvcube_delete(PVLayerCube *cube);
size_t pvcube_size(int numItems);
int pvcube_setAddr(PVLayerCube *cube);

} // end namespace PV

#endif /* PVLAYERCUBE_H_ */
