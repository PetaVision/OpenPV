/*
 * BIDSLayer.hpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 *
 */

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
   int * randomIndices(int numMatrixCol, int numMatrixRow);
   void findCoordinates(int numMatrixCol, int numMatrixRow);
   int findWeights(int x1, int y1, int x2, int y2);
   void getCoords(int numNodes, BIDSCoords * coords);
protected:
  BIDSLayer();

  int initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name);
  // other methods and member variables
private:
  // other methods and member variables
};
}
