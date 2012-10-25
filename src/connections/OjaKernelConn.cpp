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
   free(inputFiringRate[0]); inputFiringRate[0] = NULL;
   free(inputFiringRate); inputFiringRate = NULL;
   free(outputFiringRate); outputFiringRate = NULL;
}

int OjaKernelConn::initialize_base() {
   return PV_SUCCESS;
}

int OjaKernelConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) {
   int status = KernelConn::initialize(name, hc, pre, post, filename, weightInit);
   learningTime = readLearningTime();
   inputTargetRate = 0.001*readInputTargetRate(); // params file specifies target rates
   outputTargetRate = 0.001*readOutputTargetRate();
   integrationTime = readIntegrationTime();

   int numarbors = numberOfAxonalArborLists();
   int n_pre_ext = getNumWeightPatches();
   int n_post = post->getNumNeurons();
   inputFiringRate = (pvdata_t **) calloc(numarbors, sizeof(pvdata_t *));
   if (inputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rates\n", name);
      abort();
   }
   inputFiringRate[0] = (pvdata_t *) calloc(n_pre_ext*numarbors, sizeof(pvdata_t));
   if (inputFiringRate[0] == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for input firing rates\n", name);
      abort();
   }
   for (int arbor=1; arbor<numarbors; arbor++) {
      inputFiringRate[arbor] = inputFiringRate[0] + arbor*n_post;
   }
   outputFiringRate = (pvdata_t *) calloc(n_post, sizeof(pvdata_t *));
   if (outputFiringRate == NULL) {
      fprintf(stderr, "OjaKernelConn::initialize error for layer \"%s\": unable to allocate memory for output firing rates\n", name);
      abort();
   }
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


   // HyPerConn::updateState calls update_dW and updateWeights; we override update_dW but there is no need to override updateWeights
   int status = KernelConn::updateState(timef, dt);

   return status;
}

int OjaKernelConn::update_dW(int axonId) {
   int status = PV_SUCCESS;
   assert(axonId>=0 && axonId<this->numberOfAxonalArborLists());
   pvdata_t * input_rate = inputFiringRate[axonId];

   // Update weights
   int syg = post->getLayerLoc()->nf * post->getLayerLoc()->nx;

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
            if(offset+lineoffsetg+k<0 || offset+lineoffsetg+k>=post->getNumNeurons()) {
               fprintf(stderr, "%s offset=%d, lineoffseta=%d, k=%d, numNeurons=%d\n", name, offset, lineoffsetg, k, post->getNumNeurons());
               abort();
            }
            pvdata_t outputFR = outputFiringRate[offset+lineoffsetg+k];
            dwdata[lineoffsetw + k] += (inputFR - wdata[lineoffsetw+k]*outputFR)*outputFR;
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
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, outputFiringRate, 1, /*extended*/false, /*contiguous*/false, post->getLayerLoc());

   const PVLayerLoc * preloc = pre->getLayerLoc();
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, inputFiringRate[0], numberOfAxonalArborLists(), /*extended*/true, /*contiguous*/false, preloc);

   // Apply mirror boundary conditions

   // Exchange borders
   MPI_Datatype * mpi_datatype = Communicator::newDatatypes(preloc);
   PVLayerCube cube;
   memcpy(&cube.loc, preloc, sizeof(PVLayerLoc));
   cube.numItems = pre->getNumExtended();
   cube.size = sizeof(pvdata_t)*cube.numItems + sizeof(PVLayerCube); // Should be okay even though cube's data is not in place, since the mirrorTo functions don't use the size field
   for (int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
      cube.data = inputFiringRate[0];
      if ( pre->useMirrorBCs() ) {
         for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
            pre->mirrorInteriorToBorder(borderId, &cube, &cube);
         }
      }
      parent->icCommunicator()->exchange(inputFiringRate[0], mpi_datatype, preloc);
   }
   Communicator::freeDatatypes(mpi_datatype);

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
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), outputFiringRate, 1, /*extended*/ false, /*contiguous*/ false, post->getLayerLoc());
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_inputFiringRate.pvp", cpDir, name);
   assert(chars_needed<PV_PATH_MAX);
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), inputFiringRate[0], numberOfAxonalArborLists(), /*extended*/ true, /*contiguous*/ false, pre->getLayerLoc());
   return status;
}

} /* namespace PV */
