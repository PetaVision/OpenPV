/*
 * fileread.h
 *
 *  Created on: Aug 15, 2008
 *      Author: dcoates
 */

#ifndef fileread_H_
#define fileread_H_

typedef struct fileread_params_
{
    float invert;
    float uncolor;            // if true, pixels>0 become 1.0
    float spikingFlag;        // spike as poisson?
    float poissonEdgeProb;    // if so, prob
    float poissonBlankProb;   // spike as poisson?
    float marginWidth; // width of margin around edge of figure in which only background activity allowed
    float beginStim;
    float endStim;
    const char* filename;
} fileread_params;

#ifdef __cplusplus
extern "C" {
#endif

int fileread_init(PVLayer*l);
void fileread_update(PVLayer *l);

#ifdef __cplusplus
}
#endif

#endif /* LIF_H_ */
