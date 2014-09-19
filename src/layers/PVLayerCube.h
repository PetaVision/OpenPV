/*
 * PVLayer.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef PVLAYERCUBE_H_
#define PVLAYERCUBE_H_

#include "../include/pv_common.h"
#include "../utils/conversions.h"
#include "../include/pv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

PVLayerCube * pvcube_new(const PVLayerLoc * loc, int numItems);
PVLayerCube * pvcube_init(PVLayerCube * cube, const PVLayerLoc * loc, int numItems);
int           pvcube_delete(PVLayerCube * cube);
size_t        pvcube_size(int numItems);
int           pvcube_setAddr(PVLayerCube * cube);

#ifdef OBSOLETE // Marked Obsolete Sept 11, 2014.  Anything that calls these functions are themselves obsolete.
PVPatch * pvpatch_new(int nx, int ny); // PVPatch * pvpatch_new(int nx, int ny, int nf);
int       pvpatch_delete(PVPatch * p);

pvdata_t * pvpatches_new(PVPatch ** patches, int nx, int ny, int nf, int nPatches);
int       pvpatch_inplace_delete(PVPatch * p);
#endif // OBSOLETE

#ifdef __cplusplus
}
#endif

#endif /* PVLAYERCUBE_H_ */
