/*
 * PursuitLayer.cpp
 *
 *  Created on: Jul 24, 2012
 *      Author: pschultz
 */

#include "PursuitLayer.hpp"

namespace PV {

PursuitLayer::PursuitLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

PursuitLayer::PursuitLayer()
{
   initialize_base();
}

PursuitLayer::~PursuitLayer()
{
   free(wnormsq); wnormsq = NULL;
   free(minimumLocations); minimumLocations = NULL;
   free(energyDrops); energyDrops = NULL;
   free(minFeatures); minFeatures = NULL;
   free(energyDropsBestFeature); energyDropsBestFeature = NULL;
   free(foundFeatures); foundFeatures = NULL;
   free(minLocationsBestFeature); minLocationsBestFeature = NULL;
   free(gSynSparse); gSynSparse = NULL;
}

int PursuitLayer::initialize_base() {
   wnormsq = NULL;
   minimumLocations = NULL;
   energyDrops = NULL;
   minFeatures = NULL;
   energyDropsBestFeature = NULL;
   foundFeatures = NULL;
   minLocationsBestFeature = NULL;
   gSynSparse = NULL;
   minEnergyFiltered = NULL;
   return PV_SUCCESS;
}


int PursuitLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);

   nextUpdate = firstUpdate;
   updateReady = false;

   return status;
}

int PursuitLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_firstUpdate(ioFlag);
   ioParam_updatePeriod(ioFlag);
   return status;
}

void PursuitLayer::ioParam_firstUpdate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "firstUpdate", &firstUpdate, 1.0);
}

void PursuitLayer::ioParam_updatePeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "updatePeriod", &updatePeriod, 1.0);
}

int PursuitLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();

   if (status == PV_SUCCESS) {
      status = allocateBuffer(&wnormsq, getLayerLoc()->nf, "wnormsq");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&minimumLocations, getNumNeurons(), "minimumLocations");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&energyDrops, getNumNeurons(), "energyDrops");
   }

   int xy = getLayerLoc()->nx * getLayerLoc()->ny;
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&minFeatures, xy, "minFeatures");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&energyDropsBestFeature, xy, "energyDropsBestFeature");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&foundFeatures, xy, "foundFeatures");
      for (int k=0; k<xy; k++) {
         foundFeatures[k]=-1;
      }
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&minLocationsBestFeature, xy, "minLocationsBestFeature");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&gSynSparse, xy, "gSynSparse");
   }
   if (status==PV_SUCCESS) {
      status = allocateBuffer(&minEnergyFiltered, xy, "minEnergyFiltered");
   }
   if (status != PV_SUCCESS) abort();

   return status;
}

int PursuitLayer::allocateV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int PursuitLayer::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int PursuitLayer::initializeActivity() {
   return PV_SUCCESS;
}

int PursuitLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = ANNLayer::readStateFromCheckpoint(cpDir, timeptr);
   PVLayerLoc flat_loc;
   memcpy(&flat_loc, getLayerLoc(), sizeof(PVLayerLoc));
   flat_loc.nf = 1;
   pvdata_t buffer1feature[flat_loc.nx*flat_loc.ny];
   status = read_gSynSparseFromCheckpoint(cpDir, timeptr, &flat_loc);
   status = read_foundFeaturesFromCheckpoint(cpDir, timeptr, &flat_loc);
   return status;
}

int PursuitLayer::read_gSynSparseFromCheckpoint(const char * cpDir, double * timeptr, const PVLayerLoc * flat_loc) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_foundFeatures.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &gSynSparse, 1, /*extended*/true, flat_loc);
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int PursuitLayer::read_foundFeaturesFromCheckpoint(const char * cpDir, double * timeptr, const PVLayerLoc * flat_loc) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_foundFeatures.pvp");
   pvdata_t buffer1feature[flat_loc->nx*flat_loc->ny];
   pvdata_t * buffer1ptr = buffer1feature;
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &buffer1ptr, 1, /*extended*/true, flat_loc);
   assert(status==PV_SUCCESS);
   free(filename);
   for (int k=0; k<flat_loc->nx*flat_loc->ny; k++) {
      foundFeatures[k] = (int) buffer1feature[k];
   }
   return status;
}

int PursuitLayer::checkpointRead(const char * cpDir, double * timef) {
   int status = ANNLayer::checkpointRead(cpDir, timef);

   status = parent->readScalarFromFile(cpDir, getName(), "nextUpdate", &nextUpdate, firstUpdate);
   assert(status == PV_SUCCESS);
   return status;
}

int PursuitLayer::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int chars_needed;

   int filenamesize = strlen(cpDir)+1+strlen(name)+29;
   // The +1 is for the slash between cpDir and name; the +29 needs to be large enough to hold the suffix (e.g. _minLocationsBestFeature.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);

   PVLayerLoc flat_loc;
   memcpy(&flat_loc, getLayerLoc(), sizeof(PVLayerLoc));
   flat_loc.nf = 1;

   chars_needed = snprintf(filename, filenamesize, "%s/%s_gSynSparse.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &gSynSparse, 1/*numbands*/, false/*extended*/, &flat_loc);
   pvdata_t buffer1feature[flat_loc.nx*flat_loc.ny];

   chars_needed = snprintf(filename, filenamesize, "%s/%s_foundFeatures.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   for (int k=0; k<flat_loc.nx*flat_loc.ny; k++) {
      buffer1feature[k] = (pvdata_t) foundFeatures[k];
   }
   pvdata_t * buffer1ptr = buffer1feature;
   writeBufferFile(filename, icComm, timed, &buffer1ptr, 1/*numbands*/, false/*extended*/, &flat_loc);

   status = parent->writeScalarToFile(cpDir, getName(), "nextUpdate", nextUpdate);
   assert(status==PV_SUCCESS);

   free(filename);
   return status;
}

int PursuitLayer::updateState(double time, double dt) {
   if (!updateReady) return PV_SUCCESS;
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nf = getLayerLoc()->nf;
   PVHalo const * halo = &getLayerLoc()->halo;
   pvdata_t * activity = getActivity();
   memset(activity, 0, getNumExtended()*sizeof(*activity));

   int nxy = nx*ny;
   for (int kxy=0; kxy<nxy; kxy++) {
      int kf = foundFeatures[kxy];
      if (kf>=0) {
         int kx = kxPos(kxy,nx,ny,1);
         int ky = kyPos(kxy,nx,ny,1);
         int kex = kIndex(kx+halo->lt, ky+halo->up, kf, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf); /* Is this correct? Before splitting x- and y- margin widths, the ny argument was ny*nb, which seems weird. */
         activity[kex] = gSynSparse[kxy];
      }
   }
   //resetGSynBuffers_HyPerLayer(getNumNeurons(), getNumChannels(), GSyn[0]);
   updateReady = false;
   return PV_SUCCESS;
}


int PursuitLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   if (parent->simulationTime()<nextUpdate) return PV_SUCCESS;
   nextUpdate += updatePeriod;
   recvsyn_timer->start();

   assert(arborID >= 0);

   if (conn->usingSharedWeights() == false) {
      fprintf(stderr, "Error: PursuitLayer can only be the postsynaptic layer of a connection using shared weights (this condition should be removed eventually).\n");
      abort();
   }

   HyPerLayer * pre = conn->preSynapticLayer();
   const PVLayerLoc * pre_loc = pre->getLayerLoc();
   if (pre_loc->nx != getLayerLoc()->nx || pre_loc->ny != getLayerLoc()->ny) {
      fprintf(stderr, "Error: PursuitLayer requires incoming connections to be one-to-one.\n");
      abort();
   }


#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif // DEBUG_OUTPUT


   const int numExtended = activity->numItems;
   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];
      if (a == 0.0f) continue;

      PVPatch * weights = conn->getWeights(kPre, arborID);

      // WARNING - assumes every value in weights maps into a valid value in GSyn
      //         - assumes patch stride sf is 1

      int nk  = conn->fPatchSize() * weights->nx;
      int ny  = weights->ny;
      int sy  = conn->getPostNonextStrides()->sy; // stride in layer
      int syw = conn->yPatchStride();             // stride in patch
      pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
      pvdata_t * gSynPatchStart = gSynPatchHead + conn->getGSynPatchStart(kPre, arborID);
      pvwdata_t * data = conn->get_wData(arborID,kPre);
      for (int y = 0; y < ny; y++) {
         (conn->accumulateFunctionPointer)(nk, gSynPatchStart + y*sy, a, data + y*syw, NULL);
      }
   }

   // Set |w(:,:,f)|^2.  Since this is a one-to-one connection with one presynaptic feature,
   // only have to do once for each feature of a single (x,y) site and then copy.
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int num_weights = nxp*nyp*nfp;
   assert(zUnitCellSize(pre->getXScale(), getXScale())==1);
   assert(zUnitCellSize(pre->getYScale(), getYScale())==1);
   assert(conn->getNumDataPatches()==1);

   for (int kf=0; kf<nfp; kf++) {
      pvwdata_t * weight = conn->get_wDataHead(arborID, 0);
      pvdata_t sum = 0.0;
      for (int k=0; k<num_weights; k+=nfp) {
         pvwdata_t w = weight[k + kf]; // Assumes stride in features is 1.
         //TODO-CER-2014.4.4 - convert weights
         sum += w*w;
      }
      wnormsq[kf] = sum;
   }

   pvdata_t * gSynStart = GSyn[conn->getChannel()];

   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nxy = nx*ny;

   // TODO: Can I compute energyDropsBestFeature and minLocationsBestFeature without storing all the energyDrops and minimumLocations?

   for (int kxy=0; kxy<nxy; kxy++) {
      for (int kf=0; kf<nfp; kf++) {
         int k=kxy*nfp+kf; // Assumes stride in features is 1.
         minimumLocations[k] = gSynStart[k]/wnormsq[kf];
         energyDrops[k] = -gSynStart[k]*minimumLocations[k]/2;
      }
   }

   for (int kxy=0; kxy<nxy; kxy++) {
      minFeatures[kxy] = -1;
      energyDropsBestFeature[kxy] = FLT_MAX;
      int index0 = kxy*nfp; // assumes stride in features is 1.
      if (foundFeatures[kxy]>=0) {
         energyDropsBestFeature[kxy] = energyDrops[kxy*nfp+foundFeatures[kxy]];
         minFeatures[kxy] = foundFeatures[kxy];
      }
      else {
         for (int kf=0; kf<nfp; kf++) {
            if (energyDrops[index0+kf] < energyDropsBestFeature[kxy]) {
               minFeatures[kxy] = kf;
               energyDropsBestFeature[kxy] = energyDrops[index0+kf];
            }
         }
      }
   }

   for (int kxy=0; kxy<nxy; kxy++) {
      assert(minFeatures[kxy]>=0 && minFeatures[kxy]<nfp);
      int baseindex = kxy*nfp;
      minLocationsBestFeature[kxy] = minimumLocations[baseindex+minFeatures[kxy]];
   }

   bool mask[nxy];
   memset(mask, false, nxy*sizeof(*mask));

   pvdata_t smallestEnergyDrop;
   int minloc;

   while (constrainMinima(), minloc = filterMinEnergies(mask, &smallestEnergyDrop), smallestEnergyDrop<FLT_MAX) {
      assert(foundFeatures[minloc]<0 || foundFeatures[minloc]==minFeatures[minloc]);
      foundFeatures[minloc] = minFeatures[minloc];
      gSynSparse[minloc] += minLocationsBestFeature[minloc];
      if (gSynSparse[minloc] < 1e-4) {
         gSynSparse[minloc]=0;
         foundFeatures[minloc] = -1;
      }

      int minlocx = kxPos(minloc,nx,ny,1);
      int maskstartx = minlocx-(nxp-1); if (maskstartx<0) maskstartx=0;
      int maskstopx = minlocx+nxp; if (maskstopx>nx) maskstopx=nx;
      int minlocy = kyPos(minloc,nx,ny,1);
      int maskstarty = minlocy-(nyp-1); if (maskstarty<0) maskstarty=0;
      int maskstopy = minlocy+nyp; if (maskstopy>ny) maskstopy=ny;
      for (int ky=maskstarty; ky<maskstopy; ky++) {
         for (int kx=maskstartx; kx<maskstopx; kx++) {
            mask[kIndex(kx,ky,0,nx,ny,1)]=true;
         }
      }
   }


   recvsyn_timer->stop();

   updateReady = true;

   return 0;
}

int PursuitLayer::constrainMinima() {
   int nxy = getLayerLoc()->nx * getLayerLoc()->ny;
   for (int kxy=0; kxy<nxy; kxy++) {
      if (foundFeatures[kxy]>=0 && minLocationsBestFeature[kxy]<-gSynSparse[kxy]) {
         pvdata_t b = (gSynSparse[kxy]+minLocationsBestFeature[kxy])/minLocationsBestFeature[kxy];
         energyDropsBestFeature[kxy] -= energyDropsBestFeature[kxy]*b*b;
         minLocationsBestFeature[kxy] = -gSynSparse[kxy];
      }
   }
   return PV_SUCCESS;
}

int PursuitLayer::filterMinEnergies(bool * mask, pvdata_t * smallestEnergyDrop) {
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nxy = nx * ny;
   memcpy(minEnergyFiltered, energyDropsBestFeature, nxy*sizeof(minEnergyFiltered[0]));
   for (int kxy=0; kxy<nxy; kxy++) {
      int fxy = foundFeatures[kxy];
      if ( (fxy<0 && minLocationsBestFeature[kxy]<=-gSynSparse[kxy]) || (fxy>=0 && fxy!=minFeatures[kxy]) || mask[kxy] ) {
         minEnergyFiltered[kxy]=FLT_MAX;
      }
   }
   *smallestEnergyDrop = FLT_MAX;
   int minloc = -1;
   for (int kxy=0; kxy<nxy; kxy++) {
      pvdata_t y = minEnergyFiltered[kxy];
      if (y<*smallestEnergyDrop) {
         *smallestEnergyDrop = y;
         minloc = kxy;
      }
   }
   return minloc;
}

} /* namespace PV */
