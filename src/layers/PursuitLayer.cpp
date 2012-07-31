/*
 * PursuitLayer.cpp
 *
 *  Created on: Jul 24, 2012
 *      Author: pschultz
 */

#include "PursuitLayer.hpp"

namespace PV {

PursuitLayer::PursuitLayer(const char * name, HyPerCol * hc, int num_channels) {
   initialize_base();
   initialize(name, hc, num_channels);
}

PursuitLayer::PursuitLayer()
{
   initialize_base();
}

PursuitLayer::~PursuitLayer()
{
   free(innerproducts); innerproducts = NULL;
   free(wnormsq); wnormsq = NULL;
   free(minimumLocations); minimumLocations = NULL;
   free(energyDrops); energyDrops = NULL;
   free(minFeatures); minFeatures = NULL;
   free(minEnergySparse); minEnergySparse = NULL;
   free(foundFeatures); foundFeatures = NULL;
   free(minLocationsBestFeature); minLocationsBestFeature = NULL;
   free(gSynSparse); gSynSparse = NULL;
}

int PursuitLayer::initialize_base() {
   innerproducts = NULL;
   wnormsq = NULL;
   minimumLocations = NULL;
   energyDrops = NULL;
   minFeatures = NULL;
   minEnergySparse = NULL;
   foundFeatures = NULL;
   minLocationsBestFeature = NULL;
   gSynSparse = NULL;
   minEnergyFiltered = NULL;
   return PV_SUCCESS;
}


int PursuitLayer::initialize(const char * name, HyPerCol * hc, int num_channels) {
   int status = ANNLayer::initialize(name, hc, MAX_CHANNELS);
   if (status==PV_SUCCESS) {
      innerproducts = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t));
      if (innerproducts == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for innerproducts.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      wnormsq = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t));
      if (wnormsq == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for wnormsq.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      minimumLocations = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t));
      if (minimumLocations == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for minimumLocations.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      energyDrops = (pvdata_t *) calloc(getNumNeurons(), sizeof(pvdata_t));
      if (energyDrops == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for minEnergies.  Error %d\n", errno);
         abort();
      }
   }
   int xy = getLayerLoc()->nx * getLayerLoc()->ny;
   if (status==PV_SUCCESS) {
      minFeatures = (int *) calloc(xy, sizeof(pvdata_t));
      if (minFeatures == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for minFeatures.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      minEnergySparse = (pvdata_t *) calloc(xy, sizeof(pvdata_t));
      if (minEnergySparse == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for gSynSparse.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      foundFeatures = (int *) calloc(xy, sizeof(pvdata_t));
      if (foundFeatures == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for foundFeatures.  Error %d\n", errno);
         abort();
      }
   }
   for (int k=0; k<xy; k++) {
      foundFeatures[k]=-1;
   }
   if (status==PV_SUCCESS) {
      minLocationsBestFeature = (pvdata_t *) calloc(xy, sizeof(pvdata_t));
      if (minLocationsBestFeature == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for minLocationsBestFeature.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      gSynSparse = (pvdata_t *) calloc(xy, sizeof(pvdata_t));
      if (gSynSparse == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for gSynSparse.  Error %d\n", errno);
         abort();
      }
   }
   if (status==PV_SUCCESS) {
      minEnergyFiltered = (pvdata_t *) calloc(xy, sizeof(pvdata_t));
      if (minEnergyFiltered == NULL) {
         fprintf(stderr, "PursuitLayer::initialize unable to allocate memory for minEnergyFiltered.  Error %d\n", errno);
         abort();
      }
   }

   PVParams * params = parent->parameters();
   firstUpdate = params->value(name, "firstUpdate", 1);
   updatePeriod = params->value(name, "updatePeriod", 1);
   nextUpdate = firstUpdate;
   updateReady = false;

   return status;
}

int PursuitLayer::checkpointRead(const char * cpDir, float * timef) {
   int status = HyPerLayer::checkpointRead(cpDir, timef);
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
   chars_needed = snprintf(filename, filenamesize, "%s/%s_innerproducts.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, innerproducts, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_wnormsq.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, wnormsq, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_minimumLocations.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, minimumLocations, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_energyDrops.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, energyDrops, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   PVLayerLoc flat_loc;
   memcpy(&flat_loc, getLayerLoc(), sizeof(PVLayerLoc));
   flat_loc.nf = 1;

   chars_needed = snprintf(filename, filenamesize, "%s/%s_minEnergySparse.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile1Feature(filename, icComm, timed, minEnergySparse);

   chars_needed = snprintf(filename, filenamesize, "%s/%s_minLocationsBestFeature.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile1Feature(filename, icComm, timed, minLocationsBestFeature);

   chars_needed = snprintf(filename, filenamesize, "%s/%s_gSynSparse.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile1Feature(filename, icComm, timed, gSynSparse);

   pvdata_t buffer1feature[flat_loc.nx*flat_loc.ny];

   chars_needed = snprintf(filename, filenamesize, "%s/%s_minFeatures.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   for (int k=0; k<flat_loc.nx*flat_loc.ny; k++) {
      buffer1feature[k] = (pvdata_t) minFeatures[k];
   }
   writeBufferFile1Feature(filename, icComm, timed, buffer1feature);

   chars_needed = snprintf(filename, filenamesize, "%s/%s_foundFeatures.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   for (int k=0; k<flat_loc.nx*flat_loc.ny; k++) {
      buffer1feature[k] = (pvdata_t) foundFeatures[k];
   }
   writeBufferFile1Feature(filename, icComm, timed, buffer1feature);

   chars_needed = snprintf(filename, filenamesize, "%s/%s_minEnergyFiltered.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   for (int k=0; k<flat_loc.nx*flat_loc.ny; k++) {
      buffer1feature[k] = (pvdata_t) minEnergyFiltered[k];
   }
   writeBufferFile1Feature(filename, icComm, timed, buffer1feature);

   free(filename);
   return status;
}

int PursuitLayer::writeBufferFile1Feature(const char * filename, InterColComm * comm, double timed, pvdata_t * buffer) {
   FILE * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   int rank = comm->commRank();
   if (writeFile == 0 && rank==0 ) {
      fprintf(stderr, "PursuitLayer::writeBufferFile1Feature error opening \"%s\" for writing.\n", filename);
      abort();
   }
   PVLayerLoc flat_loc;
   memcpy(&flat_loc, getLayerLoc(), sizeof(PVLayerLoc));
   flat_loc.nf = 1;
   if( rank == 0 ) {
      long fpos = ftell(writeFile);
      if (fpos == 0L) {
         int status = pvp_write_header(writeFile, comm, timed, &flat_loc, PVP_NONSPIKING_ACT_FILE_TYPE,
                                       PV_FLOAT_TYPE, /*numbands*/ 1, /*extended*/false, /*contiguous*/false, NUM_BIN_PARAMS, (size_t) flat_loc.nx*flat_loc.ny);
         if (status != PV_SUCCESS) return status;
      }
   }

   int status = PV_SUCCESS;
   if ( rank==0 && fwrite(&timed, sizeof(double), 1, writeFile) != 1 )              return -1;
   int status1 =  write_pvdata(writeFile, comm, timed, buffer, &flat_loc, PV_FLOAT_TYPE,
         /*extended*/ false, /*contiguous*/ false, PVP_NONSPIKING_ACT_FILE_TYPE);
   status = status1 != PV_SUCCESS ? status1 : status;
   pvp_close_file(writeFile, comm);
   writeFile = NULL;

   return status;
}

int PursuitLayer::updateState(float time, float dt) {
   if (!updateReady) return PV_SUCCESS;
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nf = getLayerLoc()->nf;
   int nb = getLayerLoc()->nb;
   pvdata_t * activity = getCLayer()->activity->data;
   memset(activity, 0, getNumExtended()*sizeof(*activity));

   int nxy = nx*ny;
   for (int kxy=0; kxy<nxy; kxy++) {
      int kf = foundFeatures[kxy];
      if (kf>=0) {
         int kx = kxPos(kxy,nx,ny,1);
         int ky = kyPos(kxy,nx,ny,1);
         int kex = kIndex(kx+nb, ky+nb, kf, nx+2*nb, ny*nb, nf);
         activity[kex] = gSynSparse[kxy];
      }
   }
   return PV_SUCCESS;
}


int PursuitLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   if (parent->simulationTime()<nextUpdate) return PV_SUCCESS;
   nextUpdate += updatePeriod;
   recvsyn_timer->start();

   assert(arborID >= 0);

   KernelConn * kconn = dynamic_cast<KernelConn *>(conn);
   if (kconn == NULL) {
      fprintf(stderr, "Error: PursuitLayer can only be the postsynaptic layer of KernelConns, not HyPerConns (this condition should be removed eventually).\n");
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


   memset(innerproducts, 0, getNumNeurons()*sizeof(*innerproducts));
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
      pvdata_t * ipBufferStart = &innerproducts[conn->getGSynPatchStart(kPre, arborID)-GSyn[conn->getChannel()]];
      pvdata_t * data = conn->get_wData(arborID,kPre);
      for (int y = 0; y < ny; y++) {
         (conn->accumulateFunctionPointer)(nk, ipBufferStart + y*sy, a, data + y*syw);
      }
   }

   // Set |w(:,:,f)|^2.  Since this is a one-to-one KernelConn with one presynaptic feature,
   // only have to do once for each feature of a single (x,y) site and then copy.
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   assert(zUnitCellSize(pre->getXScale(), getXScale())==1);
   assert(zUnitCellSize(pre->getYScale(), getYScale())==1);
   assert(conn->getNumDataPatches()==1);

   for (int kf=0; kf<nfp; kf++) {
      pvdata_t * weight = conn->get_wDataHead(arborID, 0);
      pvdata_t sum = 0.0;
      for (int kxy=0; kxy<nxp*nyp; kxy++) {
         pvdata_t w = weight[kxy*nfp + kf]; // Assumes stride in features is 1.
         sum += w*w;
      }
      wnormsq[kf] = sum;
   }
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nxy = nx*ny;
   for (int kxy=1; kxy<nxy; kxy++) {
      memcpy(&wnormsq[kxy*nfp], wnormsq, nfp*sizeof(pvdata_t));
   }

   for (int k=0; k<getNumNeurons(); k++) {
      minimumLocations[k] = innerproducts[k]/wnormsq[k];
      energyDrops[k] = -innerproducts[k]*minimumLocations[k]/2;
   }

   for (int kxy=0; kxy<nxy; kxy++) {
      minFeatures[kxy] = -1;
      minEnergySparse[kxy] = FLT_MAX;
      int index0 = kxy*nfp; // assumes stride in features is 1.
      if (foundFeatures[kxy]>=0) {
         minEnergySparse[kxy] = energyDrops[kxy*nfp+foundFeatures[kxy]];
         minFeatures[kxy] = foundFeatures[kxy];
      }
      else {
         for (int kf=0; kf<nfp; kf++) {
            if (energyDrops[index0+kf] < minEnergySparse[kxy]) {
               minFeatures[kxy] = kf;
               minEnergySparse[kxy] = energyDrops[index0+kf];
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
   memset(mask, false, nxy*sizeof(mask[0]));
   // memset(gSynSparse, 0, nxy*sizeof(*gSynSparse));

   pvdata_t smallestEnergyDrop;
   int minloc;
   constrainMinima();
   minloc = filterMinEnergies(mask, &smallestEnergyDrop);

   while (smallestEnergyDrop<FLT_MAX) {
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

      constrainMinima();
      minloc = filterMinEnergies(mask, &smallestEnergyDrop);
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
         minEnergySparse[kxy] -= minEnergySparse[kxy]*b*b;
         minLocationsBestFeature[kxy] = -gSynSparse[kxy];
      }
   }
   return PV_SUCCESS;
}

int PursuitLayer::filterMinEnergies(bool * mask, pvdata_t * smallestEnergyDrop) {
   int nxy = getLayerLoc()->nx * getLayerLoc()->ny;
   memcpy(minEnergyFiltered, minEnergySparse, nxy*sizeof(minEnergyFiltered[0]));
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
