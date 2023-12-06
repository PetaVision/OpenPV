/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

enum ChannelType {
   CHANNEL_EXC      = 0,
   CHANNEL_INH      = 1,
   CHANNEL_INHB     = 2,
   CHANNEL_GAP      = 3,
   CHANNEL_NORM     = 4,
   CHANNEL_NOUPDATE = -1
};

typedef struct { unsigned int s1, s2, s3; } taus_state_t;

typedef struct taus_uint4_ {
   unsigned int s0;
   taus_state_t state;
} taus_uint4;

#endif /* PV_TYPES_H_ */
