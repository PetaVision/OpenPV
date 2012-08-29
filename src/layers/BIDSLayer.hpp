/*
 * BIDSLayer.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 *
 */

#ifndef INITBIDSLAYER_HPP_
#define INITBIDSLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIF.hpp"

typedef struct _BIDSCoords{
   int xCoord;
   int yCoord;
} BIDSCoords;

namespace PV {
class BIDSLayer : public PV::LIF {
public:
   BIDSLayer(const char* name, HyPerCol * hc); // The constructor called by other methods
   int updateState(float timef, float dt);
   int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);
   void setCoords(int numNodes, BIDSCoords * coords, int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy);
   BIDSCoords * getCoords();
   int numNodes;
protected:
  BIDSLayer();
  BIDSCoords * coords; //structure array pointer that holds the randomly generated coordinates for the specified number of BIDS nodes
  int initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name);
  // other methods and member variables
private:
  // other methods and member variables
};
}

#endif /* INITBIDSLAYER_HPP_ */
