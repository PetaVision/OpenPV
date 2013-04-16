/*
 * OjaKernelConn.cpp
 *
 *  Created on: Oct 10, 2012
 *      Author: pschultz
 */

#include "OjaKernelConn.hpp"

namespace PV {
OjaKernelConn::OjaKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                             const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
}

OjaKernelConn::OjaKernelConn()
{
   initialize_base();
}

OjaKernelConn::~OjaKernelConn() {
   free(inputFiringRate[0]);
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      inputFiringRate[arbor] = NULL;
   }
   free(inputFiringRate); inputFiringRate = NULL;
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      pvcube_delete(inputFiringRateCubes[arbor]); inputFiringRateCubes[arbor] = NULL;
   }
   free(inputFiringRateCubes); inputFiringRateCubes = NULL;
   free(outputFiringRate); outputFiringRate = NULL;
   Communicator::freeDatatypes(mpi_datatype); mpi_datatype = NULL;
}

int OjaKernelConn::initialize_base() {
   inputFiringRateCubes = NULL;
   inputFiringRate = NULL;
   mpi_datatype = NULL;
   outputFiringRate = NULL;
   return PV_SUCCESS;
}

int OjaKernelConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) {
   int status = KernelConn::initialize(name, hc, pre, post, filename, weightInit);

   int numarbors = numberOfAxonalArborLists(); assert(numarbors>0);
   int n_pre_ext = getNumWeightPatches();
   int n_post = post->getNumNeurons();
   inputFiringRateCubes = (PVLayerCube **) calloc(numarbors, sizeof(PVLayerCube *));
   if (inputFiringRateCubes == NULL) {
      fprintf(stderr, "OjaKernelConn \"%s\" error allocating inputFiringRateCubes", name);
      abort();
   }

   // Don't allocate cube's data in place, so that the data can be written to/ read from a pvp file at once
   inputFiringRate = (pvdata_t **) calloc(numarbors, sizeof(pvdata_t *));
   if (inputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rates\n", name);
      abort();
   }
   inputFiringRate = (pvdata_t **) calloc(numarbors, sizeof(pvdata_t *));
   if (inputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rate pointers\n", name);
      abort();
   }
   inputFiringRate[0] = (pvdata_t *) malloc(numarbors*n_pre_ext * sizeof(pvdata_t *));
   if (inputFiringRate[0]==NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rates\n", name);
      abort();
   }
   for (int arbor = 1; arbor<numarbors; arbor++) {
      inputFiringRate[arbor] = inputFiringRate[0]+arbor*n_pre_ext;
   }
   for (int k=0; k<numarbors*n_pre_ext; k++) {
      inputFiringRate[0][k] = getInputTargetRate();
   }
   for (int arbor = 0; arbor<numarbors; arbor++) {
      inputFiringRateCubes[arbor] = (PVLayerCube *) calloc(1, sizeof(PVLayerCube));
      if (inputFiringRateCubes[arbor]==NULL) {
         fprintf(stderr, "inputFiringRateCubes[arbor]==NULL.  This computer fails.\n");
         abort();
      }
      inputFiringRateCubes[arbor]->size = pvcube_size(n_pre_ext); // Should be okay even though cube's data is not in place, since the mirrorTo functions don't use the size field
      inputFiringRateCubes[arbor]->numItems = pvcube_size(n_pre_ext);
      memcpy(&(inputFiringRateCubes[arbor]->loc), pre->getLayerLoc(), sizeof(PVLayerLoc));
      inputFiringRateCubes[arbor]->data = inputFiringRate[arbor];
   }

   // Output firing rate doesn't need arbors since all arbors go to the same output, or a cube since we don't have to exchange borders.
   outputFiringRate = (pvdata_t *) malloc(n_post * sizeof(pvdata_t *));
   for (int k=0; k<n_post; k++) {
      outputFiringRate[k] = getOutputTargetRate();
   }
   if (outputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for output firing rates\n", name);
      abort();
   }
   mpi_datatype = Communicator::newDatatypes(pre->getLayerLoc());
   return status;
}

int OjaKernelConn::setParams(PVParams * params) {
   int status = KernelConn::setParams(params);
   readLearningTime(params);
   readInputTargetRate(params);
   readOutputTargetRate(params);
   readIntegrationTime(params);
   readAlphaMultiplier(params);
   return status;
}

int OjaKernelConn::updateState(double timef, double dt) {

   float decayfactor = expf(-parent->getDeltaTime()/integrationTime);
   // Update output firing rate
   const PVLayerLoc * postloc = post->getLayerLoc();
   for (int kpost=0; kpost<post->getNumNeurons(); kpost++) {
      int kpostext = kIndexExtended(kpost, postloc->nx, postloc->ny, postloc->nf, postloc->nb);
      outputFiringRate[kpost] = decayfactor * (outputFiringRate[kpost] + post->getLayerData()[kpostext]/integrationTime);
   }
   // Update input firing rates
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      pvdata_t * input_rate = inputFiringRate[arbor];
      for (int kex=0; kex<getNumWeightPatches(); kex++) {
         input_rate[kex] = decayfactor * (input_rate[kex] + pre->getLayerData(getDelay(arbor))[kex]/integrationTime);
      }
   }

   // HyPerConn::updateState calls update_dW and updateWeights
   int status = KernelConn::updateState(timef, dt);

   return status;
}

int OjaKernelConn::updateWeights(int axonId) {
   lastUpdateTime = parent->simulationTime();
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      pvdata_t * w_data_start = get_wDataStart(kAxon);
      for( int k=0; k<nxp*nyp*nfp*getNumDataPatches(); k++ ) {
         pvdata_t w = w_data_start[k];
         w += (weightUpdatePeriod/parent->getDeltaTime())*get_dwDataStart(kAxon)[k];
         if (w < 0.0f) w = 0.0f;
         w_data_start[k] = w;
      }
   }
   return PV_BREAK;
}


int OjaKernelConn::update_dW(int axonId) {
   int status = PV_SUCCESS;
   assert(axonId>=0 && axonId<this->numberOfAxonalArborLists());
   pvdata_t * input_rate = inputFiringRate[axonId];

   // Update weights
   int syg = post->getLayerLoc()->nf * post->getLayerLoc()->nx;

   float inputBurstRate = pre->getMaxRate();
   float outputBurstRate = 1000.0f/getIntegrationTime();
   float alpha = alphaMultiplier*inputBurstRate/outputBurstRate;

   for (int kex=0; kex<getNumWeightPatches(); kex++) {
      PVPatch * weights = getWeights(kex,axonId);
      int offset = getGSynPatchStart(kex,axonId)-post->getChannel(channel);
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvdata_t * dwdata = get_dwData(axonId, kex);
      const pvdata_t * wdata = get_wData(axonId, kex);
      int lineoffsetw = 0;
      int lineoffsetg = 0;
      pvdata_t inputFR = input_rate[kex];
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            pvdata_t outputFR = outputFiringRate[offset+lineoffsetg+k];
            dwdata[lineoffsetw + k] += (inputFR - alpha*wdata[lineoffsetw+k]*outputFR)*outputFR;
         }
         lineoffsetw += syp;
         lineoffsetg += syg;
      }
   }
   // Multiply by dt*learningRate, normalize by dividing by inputTargetRate*outputTargetRate, and average by dividing by (numNeurons/numKernels)
   int numKernelIndices = getNumDataPatches();
   int divisor = pre->getNumNeurons()/numKernelIndices;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   float scalefactor = parent->getDeltaTime()/getLearningTime()/(getInputTargetRate()*getOutputTargetRate())/((float) divisor);
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp*nyp*nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] *= scalefactor;
      }
   }
   // Averaging across processes is done in reduceKernels().

   return status;
}

int OjaKernelConn::checkpointRead(const char * cpDir, double * timef) {
   int status = KernelConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_outputFiringRate.pvp", cpDir, name);
   if (chars_needed>=PV_PATH_MAX) {
      if (parent->columnId()==0) {
         fprintf(stderr, "OjaKernelConn::checkpointRead error: Path \"%s/%s_outputFiringRate.pvp\" is too long.\n", cpDir, name);
         abort();
      }
   }
   double timed;
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &outputFiringRate, 1, /*extended*/false, post->getLayerLoc());

   const PVLayerLoc * preloc = pre->getLayerLoc();
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, inputFiringRate, numberOfAxonalArborLists(), /*extended*/true, preloc);

   // Apply mirror boundary conditions

   // Exchange borders
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      if ( pre->useMirrorBCs() ) {
         for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
            pre->mirrorInteriorToBorder(borderId, inputFiringRateCubes[arbor], inputFiringRateCubes[arbor]);
         }
      }
      parent->icCommunicator()->exchange(inputFiringRate[0], mpi_datatype, preloc);
   }

   return status;
}
int OjaKernelConn::checkpointWrite(const char * cpDir) {
   int status = KernelConn::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_outputFiringRate.pvp", cpDir, name);
   if (chars_needed>=PV_PATH_MAX) {
      if (parent->columnId()==0) {
         fprintf(stderr, "OjaKernelConn::checkpointWrite error: Path \"%s/%s_outputFiringRate.pvp\" is too long.\n", cpDir, name);
         abort();
      }
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &outputFiringRate, 1, /*extended*/ false, post->getLayerLoc());
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), inputFiringRate, numberOfAxonalArborLists(), /*extended*/ true, pre->getLayerLoc());
   return status;
}

} /* namespace PV */
