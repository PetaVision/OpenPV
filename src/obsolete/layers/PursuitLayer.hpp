/*
 * PursuitLayer.hpp
 *
 *  Created on: Jul 24, 2012
 *      Author: pschultz
 */

#ifndef PURSUITLAYER_HPP_
#define PURSUITLAYER_HPP_

#include "GenerativeLayer.hpp"
#include "../connections/HyPerConn.hpp"
#include "../io/fileio.hpp"
#include "../io/io.h"

namespace PV {

class PursuitLayer: public PV::ANNLayer {

// Member functions
public:
   PursuitLayer(const char * name, HyPerCol * hc);
   virtual ~PursuitLayer();

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);

   virtual int allocateDataStructures();

   int updateState(double time, double dt);

   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int arborID);

protected:
   PursuitLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_firstUpdate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_updatePeriod(enum ParamsIOFlag ioFlag);
   virtual int allocateV();
   virtual int initializeV();
   virtual int initializeActivity();
   int constrainMinima();
   int filterMinEnergies(bool * mask, pvdata_t * smallestEnergyDrop);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int read_gSynSparseFromCheckpoint(const char * cpDir, double * timeptr, const PVLayerLoc * flat_loc);
   virtual int read_foundFeaturesFromCheckpoint(const char * cpDir, double * timeptr, const PVLayerLoc * flat_loc);

private:
   int initialize_base();

// Member variables
public:

protected:
   pvdata_t * wnormsq;
   pvdata_t * minimumLocations;
   pvdata_t * energyDrops;
   int * minFeatures;
   pvdata_t * energyDropsBestFeature;
   int * foundFeatures;
   pvdata_t * minLocationsBestFeature;
   pvdata_t * gSynSparse;
   pvdata_t * minEnergyFiltered;

   double firstUpdate;
   double updatePeriod;
   double nextUpdate;
   bool updateReady;

private:

};

} /* namespace PV */
#endif /* PURSUITLAYER_HPP_ */
