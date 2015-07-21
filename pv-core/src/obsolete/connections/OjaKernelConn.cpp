/*
 * OjaKernelConn.cpp
 *
 *  Created on: Oct 10, 2012
 *      Author: pschultz
 */

#include "OjaKernelConn.hpp"

namespace PV {
OjaKernelConn::OjaKernelConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
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

int OjaKernelConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int OjaKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_learningTime(ioFlag);
   ioParam_inputTargetRate(ioFlag);
   ioParam_outputTargetRate(ioFlag);
   ioParam_integrationTime(ioFlag);
   ioParam_alphaMultiplier(ioFlag);
   ioParam_dWUpdatePeriod(ioFlag);
   return status;
}

// TODO: make sure code works in non-shared weight case
void OjaKernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void OjaKernelConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   HyPerConn::ioParam_initialWeightUpdateTime(ioFlag);
   if (ioFlag==PARAMS_IO_READ) {
      dWUpdateTime = weightUpdateTime;
   }
}

void OjaKernelConn::ioParam_learningTime(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "learningTime", &learningTime, 1.0f);
}

void OjaKernelConn::ioParam_inputTargetRate(enum ParamsIOFlag ioFlag) {
   float rateInHertz = 1000.0f * inputTargetRate;
   parent->ioParamValue(ioFlag, name, "inputTargetRate", &rateInHertz, 1.0f);
   if (ioFlag == PARAMS_IO_READ) inputTargetRate = 0.001f * rateInHertz;
}

void OjaKernelConn::ioParam_outputTargetRate(enum ParamsIOFlag ioFlag) {
   float rateInHertz = 1000.0f * outputTargetRate;
   parent->ioParamValue(ioFlag, name, "outputTargetRate", &rateInHertz, 1.0f);
   if (ioFlag == PARAMS_IO_READ) outputTargetRate = 0.001f * rateInHertz;
}

void OjaKernelConn::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "integrationTime", &integrationTime, 1.0f);
}

void OjaKernelConn::ioParam_alphaMultiplier(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "alphaMultiplier", &alphaMultiplier, 1.0f);
}

void OjaKernelConn::ioParam_dWUpdatePeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dWUpdatePeriod", &dWUpdatePeriod, 1.0);
}

int OjaKernelConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   int numarbors = numberOfAxonalArborLists(); assert(numarbors>0);

   // Don't allocate cube's data in place, so that the data can be written to/ read from a pvp file at once
   inputFiringRate = (pvdata_t **) calloc(numarbors, sizeof(pvdata_t *));
   if (inputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rate pointers\n", name);
      abort();
   }
   int n_pre_ext = getNumWeightPatches();
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

   inputFiringRateCubes = (PVLayerCube **) calloc(numarbors, sizeof(PVLayerCube *));
   if (inputFiringRateCubes == NULL) {
      fprintf(stderr, "OjaKernelConn \"%s\" error allocating inputFiringRateCubes", name);
      abort();
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
   mpi_datatype = Communicator::newDatatypes(pre->getLayerLoc());

   // Output firing rate doesn't need arbors since all arbors go to the same output, or a cube since we don't have to exchange borders.
   int n_post = post->getNumNeurons();
   outputFiringRate = (pvdata_t *) malloc(n_post * sizeof(pvdata_t *));
   for (int k=0; k<n_post; k++) {
      outputFiringRate[k] = getOutputTargetRate();
   }
   if (outputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for output firing rates\n", name);
      abort();
   }
   return status;
}

int OjaKernelConn::updateState(double timef, double dt)
{
   update_timer->start();

   float decayfactor = expf(-parent->getDeltaTime()/integrationTime);
   // Update output firing rate
   const PVLayerLoc * postloc = post->getLayerLoc();
   for (int kpost=0; kpost<post->getNumNeurons(); kpost++) {
      int kpostext = kIndexExtended(kpost, postloc->nx, postloc->ny, postloc->nf, postloc->halo.lt, postloc->halo.rt, postloc->halo.dn, postloc->halo.up);
      outputFiringRate[kpost] = decayfactor * (outputFiringRate[kpost] + post->getLayerData()[kpostext]/integrationTime);
   }
   // Update input firing rates
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      pvdata_t * input_rate = inputFiringRate[arbor];
      for (int kex=0; kex<getNumWeightPatches(); kex++) {
         input_rate[kex] = decayfactor * (input_rate[kex] + pre->getLayerData(getDelay(arbor))[kex]/integrationTime);
      }
   }

   // update timer already in HyPerConn::updateState, don't call twice
   update_timer->stop();

   // HyPerConn::updateState calls update_dW and updateWeights
   return HyPerConn::updateState(timef, dt);
}

int OjaKernelConn::updateWeights(int axonId) {
   lastUpdateTime = parent->simulationTime();
   float timestepsPerUpdate = weightUpdatePeriod/parent->getDeltaTime(); // Number of timesteps between weight updates;
   // dw needs to be multiplied by this quantity since updateWeights is only called this often.
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      pvwdata_t * w_data_start = get_wDataStart(kAxon);
      for( int k=0; k<nxp*nyp*nfp*getNumDataPatches(); k++ ) {
         //TODO-CER-2014.4.3 - weight conversion
         pvwdata_t w = w_data_start[k];
         w += timestepsPerUpdate*get_dwDataStart(kAxon)[k];
         if (w < 0.0f) w = 0.0f;
         w_data_start[k] = w;
      }
   }
   return PV_BREAK;
}

int OjaKernelConn::calc_dW(int arborId) {
   int status = PV_SUCCESS;
   if (parent->simulationTime() < dWUpdateTime) {
      return status;
   }
   dWUpdateTime += dWUpdatePeriod;
   status = HyPerConn::calc_dW(arborId);
   return status;
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
      int offset = getGSynPatchStart(kex,axonId); //-post->getChannel(channel);
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvwdata_t * dwdata = get_dwData(axonId, kex);
      const pvwdata_t * wdata = get_wData(axonId, kex);
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
      pvwdata_t * dwpatchdata = get_dwDataHead(axonId,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] *= scalefactor;
      }
   }
   // Averaging across processes is done in reduceKernels().

   return status;
}

int OjaKernelConn::checkpointRead(const char * cpDir, double * timef) {
   int status = HyPerConn::checkpointRead(cpDir, timef);
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
   status = HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &outputFiringRate, 1, /*extended*/false, post->getLayerLoc());
   assert(status==PV_SUCCESS);

   const PVLayerLoc * preloc = pre->getLayerLoc();
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   status = HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, inputFiringRate, numberOfAxonalArborLists(), /*extended*/true, preloc);
   assert(status==PV_SUCCESS);

   status = parent->readScalarFromFile(cpDir, getName(), "next_dWUpdate", &dWUpdateTime, weightUpdateTime);
   assert(status==PV_SUCCESS);

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
   int status = HyPerConn::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_outputFiringRate.pvp", cpDir, name);
   if (chars_needed>=PV_PATH_MAX) {
      if (parent->columnId()==0) {
         fprintf(stderr, "OjaKernelConn::checkpointWrite error: Path \"%s/%s_outputFiringRate.pvp\" is too long.\n", cpDir, name);
         abort();
      }
   }
   status = HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &outputFiringRate, 1, /*extended*/ false, post->getLayerLoc());
   assert(status == PV_SUCCESS);
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   status = HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), inputFiringRate, numberOfAxonalArborLists(), /*extended*/ true, pre->getLayerLoc());
   assert(status == PV_SUCCESS);
   status = parent->writeScalarToFile(cpDir, getName(), "next_dWUpdate", dWUpdateTime);
   assert(status == PV_SUCCESS);

   return status;
}

} /* namespace PV */
