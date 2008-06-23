#ifndef PV_NEIGHBOR_H_
#define PV_NEIGHBOR_H_

#include <pv.h>

int neighbor_index(PVState* s, int comm_id, int index);

int pv_north(PVState* s, int comm_id);
int pv_south(PVState* s, int comm_id);
int pv_east(PVState* s, int comm_id);
int pv_west(PVState* s, int comm_id);

int pv_northwest(PVState* s, int comm_id);
int pv_northeast(PVState* s, int comm_id);
int pv_southwest(PVState* s, int comm_id);
int pv_southeast(PVState* s, int comm_id);

#endif /*PV_NEIGHBOR_H_*/
