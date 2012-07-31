/*
 * PursuitLayer.hpp
 *
 *  Created on: Jul 24, 2012
 *      Author: pschultz
 */

#ifndef PURSUITLAYER_HPP_
#define PURSUITLAYER_HPP_

#include "GenerativeLayer.hpp"
#include "../connections/KernelConn.hpp"
#include "../io/fileio.hpp"
#include "../io/io.h"

namespace PV {

class PursuitLayer: public PV::ANNLayer {

// Member functions
public:
   PursuitLayer(const char * name, HyPerCol * hc, int num_channels=MAX_CHANNELS);
   virtual ~PursuitLayer();

   virtual int checkpointRead(const char * cpDir, float * timef);
   virtual int checkpointWrite(const char * cpDir);

   int updateState(float time, float dt);

   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int arborID);

protected:
   PursuitLayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels);
   int writeBufferFile1Feature(const char * filename, InterColComm * comm, double timed, pvdata_t * buffer);
   int constrainMinima();
   int filterMinEnergies(bool * mask, pvdata_t * smallestEnergyDrop);
private:
   int initialize_base();

// Member variables
public:

protected:
   pvdata_t * innerproducts;
   pvdata_t * wnormsq;
   pvdata_t * minimumLocations;
   pvdata_t * energyDrops;
   int * minFeatures;
   pvdata_t * minEnergySparse;
   int * foundFeatures;
   pvdata_t * minLocationsBestFeature;
   pvdata_t * gSynSparse;
   pvdata_t * minEnergyFiltered;

   double oldEnergy;
   double preEnergy;

   float firstUpdate;
   float updatePeriod;
   float nextUpdate;
   bool updateReady;

private:

};

} /* namespace PV */
#endif /* PURSUITLAYER_HPP_ */
