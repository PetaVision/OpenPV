/*
 * HyperLayer.c
 *
 *  Created on: Oct 3, 2008
 *      Author: rasmussn
 */

//#pragma FTT elemental, vectorize
static inline float xPos(int idx, int nx, float x0, float dx, int numFeatures)
{
    return x0 + dx*(0.5 + ((idx/numFeatures) % nx));
}

//#pragma FTT elemental, vectorize
static inline float yPos(int idx, int nx, float y0, float dy, int numFeatures)
{
    return y0 + dy*(0.5 + (idx/(nx*numFeatures)));
}

