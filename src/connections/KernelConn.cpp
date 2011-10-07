/*
 * KernelConn.cpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#include "KernelConn.hpp"
#include <assert.h>
#include <float.h>
#include "../io/io.h"

namespace PV {

KernelConn::KernelConn()
{
   initialize_base();
}

#ifdef OBSOLETE // marked obsolete Jul 22, 2011.  Other constructor has been given filename with a default of NULL
KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel) : HyPerConn()
{
   KernelConn::initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}
#endif // OBSOLETE

#ifdef OBSOLETE // marked obsolete Jul 21, 2011.  No routine calls it, and it doesn't make sense to define a connection without specifying a channel.
KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post) : HyPerConn()
{
   KernelConn::initialize_base();
   initialize(name, hc, pre, post, CHANNEL_EXC, NULL); // use default channel
}
#endif // OBSOLETE

KernelConn::   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel, const char * filename, InitWeights *weightInit) : HyPerConn()
{
   KernelConn::initialize_base();
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
   // HyPerConn::initialize is not virtual
}

KernelConn::~KernelConn() {
   free(kernelPatches);
   if (dKernelPatches != NULL) {free(dKernelPatches);};
#ifdef PV_USE_MPI
   free(mpiReductionBuffer);
#endif // PV_USE_MPI
}

int KernelConn::initialize_base()
{
   kernelPatches = NULL;
   dKernelPatches = NULL;
   lastUpdateTime = 0.f;
   plasticityFlag = false;
   tmpPatch = NULL;
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   return PV_SUCCESS; // return HyPerConn::initialize_base();
   // KernelConn constructor calls HyPerConn::HyPerConn(), which
   // calls HyPerConn::initialize_base().
}
int KernelConn::initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename) {
   return KernelConn::initialize(name, hc, pre, post, channel, filename, NULL);
}

int KernelConn::initialize( const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit ) {
   PVParams * params = hc->parameters();
   symmetrizeWeightsFlag = params->value(name, "symmetrizeWeights",0);
   HyPerConn::initialize(name, hc, pre, post, channel, filename, weightInit);
   weightUpdateTime = initializeUpdateTime(params);
#ifdef PV_USE_MPI
   // preallocate buffer for MPI_Allreduce call in reduceKernels
   //int axonID = 0; // for now, only one axonal arbor
   const int numPatches = numDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;
   mpiReductionBuffer = (pvdata_t *) malloc(localSize*sizeof(pvdata_t));
   if(mpiReductionBuffer == NULL) {
      fprintf(stderr, "KernelConn::initialize:Unable to allocate memory\n");
      exit(PV_FAILURE);
   }
#endif // PV_USE_MPI
   return PV_SUCCESS;
}

int KernelConn::createArbors() {
   HyPerConn::createArbors();
   kernelPatches = (PVPatch***) calloc(numberOfAxonalArborLists(), sizeof(PVPatch**));
   assert(kernelPatches!=NULL);
   dKernelPatches = (PVPatch***) calloc(numberOfAxonalArborLists(), sizeof(PVPatch**));
   assert(dKernelPatches!=NULL);
   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++){
      dKernelPatches[arborId] = NULL;
   }
   return PV_SUCCESS; //should we check if allocation was successful?
}

int KernelConn::setWPatches(PVPatch ** patches, int arborId){
   int status = HyPerConn::setWPatches(patches, arborId);
   assert(status == 0);
   assert(kernelPatches[arborId] == NULL);
   kernelPatches[arborId] = tmpPatch;  // tmpPatch stores most recently allocated PVPatch**
   tmpPatch = NULL;  // assert(tmpPatch == NULL) before assigning tmpPatch;
   return PV_SUCCESS;
}

int KernelConn::setdWPatches(PVPatch ** patches, int arborId){
   int status = HyPerConn::setdWPatches(patches, arborId);
   assert(status == 0);
   assert(dKernelPatches[arborId] == NULL);
   dKernelPatches[arborId] = tmpPatch;  // tmpPatch stores most recently allocated PVPatch**
   tmpPatch = NULL;  // assert(tmpPatch == NULL) before assigning tmpPatch;
   return PV_SUCCESS;
}

PVPatch ** KernelConn::allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   //const int arbor = 0;
   int numKernelPatches = numDataPatches();

   assert(tmpPatch == NULL);
   tmpPatch = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   assert(tmpPatch != NULL);
   //setKernelPatches(newKernelPatch, axonId);

   for (int kernelIndex = 0; kernelIndex < numKernelPatches; kernelIndex++) {
      //kernelPatches[axonId][kernelIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      tmpPatch[kernelIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      assert(tmpPatch[kernelIndex] != NULL );
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      patches[patchIndex] = pvpatch_new(nxPatch, nyPatch, nfPatch);
   }
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      int kernelIndex = this->patchIndexToKernelIndex(patchIndex);
      patches[patchIndex]->data = tmpPatch[kernelIndex]->data;
   }
   return patches;
}

int KernelConn::initializeUpdateTime(PVParams * params) {
   if( plasticityFlag ) {
      float defaultUpdatePeriod = 1.f;
      weightUpdatePeriod = params->value(name, "weightUpdatePeriod", defaultUpdatePeriod);
   }
   return PV_SUCCESS;
}

int KernelConn::shrinkPatches(int arborId) {
   int numPatches = numDataPatches();
   for (int kex = 0; kex < numPatches; kex++) {
      PVAxonalArbor * arbor = axonalArbor(kernelIndexToPatchIndex(kex), arborId);

      HyPerConn::shrinkPatch(arbor);
   } // loop over arbors (pre-synaptic neurons)

   return 0;
}


/*TODO  createWeights currently breaks in this subclass if called more than once,
 * fix interface by adding extra dataPatches argument to overloaded method
 * so asserts are unnecessary
 */
PVPatch ** KernelConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   //assert(numberOfAxonalArborLists() == 1);

   assert(patches == NULL);

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   //assert(kernelPatches == NULL);
   allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);

   return patches;
}

int KernelConn::deleteWeights()
{
   //const int arbor = 0;

   for(int n=0;n<numberOfAxonalArborLists(); n++) {
      for (int k = 0; k < numDataPatches(); k++) {
         pvpatch_inplace_delete(kernelPatches[n][k]);
      }
      free(kernelPatches[n]);
   }
   free(kernelPatches);

   return HyPerConn::deleteWeights();
}

PVPatch ***  KernelConn::initializeWeights(PVPatch *** arbors, int numPatches,
      const char * filename)
{
   //int arbor = 0;
   int numKernelPatches = numDataPatches();
   HyPerConn::initializeWeights(kernelPatches, numKernelPatches, filename);
   return arbors;
}

PVPatch ** KernelConn::readWeights(PVPatch ** patches, int numPatches,
      const char * filename)
{
   //HyPerConn::readWeights(patches, numPatches, filename);

   return patches;
}

int KernelConn::numDataPatches()
{
   int nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int numKernelPatches = pre->clayer->loc.nf * nxKernel * nyKernel;
   return numKernelPatches;
}

float KernelConn::minWeight(int arborId)
{
   //const int axonID = 0;
   const int numKernels = numDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float min_weight = FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = kernelPatches[arborId][iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         min_weight = (min_weight < kernelWeights[iWeight]) ? min_weight
               : kernelWeights[iWeight];
      }
   }
   return min_weight;
}

float KernelConn::maxWeight(int arborId)
{
   //const int axonID = 0;
   const int numKernels = numDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float max_weight = -FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = kernelPatches[arborId][iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         max_weight = (max_weight > kernelWeights[iWeight]) ? max_weight
               : kernelWeights[iWeight];
      }
   }
   return max_weight;
}

int KernelConn::calc_dW(int axonId){
   // zero dWeightPatches
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->numDataPatches(); kKernel++){
         PVPatch * dKernelPatch = dKernelPatches[kAxon][kKernel];
         int syPatch = dKernelPatch->sy;
         int nkPatch = dKernelPatch->nf * dKernelPatch->nx;
         float * dWeights = dKernelPatch->data;
         for(int kyPatch = 0; kyPatch < dKernelPatch->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWeights[kPatch] = 0.0f;
            }
            dWeights += syPatch;
         }
      }
   }
   // Generally, divide dWeights by (number of *non*-extended neurons divided by number of kernels)
   // This isn't done here because the dWeights is set to zero in this method.
   return PV_BREAK;
}

int KernelConn::updateState(float time, float dt) {
   update_timer->start();
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   if( time >= weightUpdateTime) {
      computeNewWeightUpdateTime(time, weightUpdateTime);
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = calc_dW(axonID);  // calculate changes in weights
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }

#ifdef PV_USE_MPI
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = reduceKernels(axonID);  // combine partial changes in each column
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }
#endif // PV_USE_MPI

      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = updateWeights(axonID);  // calculate new weights from changes
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }
      if( normalize_flag ) {
         for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
            status = normalizeWeights(kernelPatches[axonID], numDataPatches(), axonID);
            if (status == PV_BREAK) {break;}
            assert(status == PV_SUCCESS);
         }
      }  // normalize_flag
   } // time > weightUpdateTime

update_timer->stop();
return PV_SUCCESS;
} // updateState

int KernelConn::updateWeights(int axonId){
   lastUpdateTime = parent->simulationTime();
   // add dKernelPatches to KernelPatches
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->numDataPatches(); kKernel++){
         PVPatch * kernelPatch = kernelPatches[kAxon][kKernel];
         PVPatch * dKernelPatch = dKernelPatches[kAxon][kKernel];
         int syPatch = kernelPatch->sy;
         int nkPatch = kernelPatch->nf * kernelPatch->nx;
         float * weights = kernelPatch->data;
         float * dWeights = dKernelPatch->data;
         for(int kyPatch = 0; kyPatch < kernelPatch->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               weights[kPatch] += dWeights[kPatch];
            }
            weights += syPatch;
            dWeights += syPatch;
         }
      }
   }
   return PV_BREAK;
}

float KernelConn::computeNewWeightUpdateTime(float time, float currentUpdateTime) {
   // Is only called by KernelConn::updateState if plasticityFlag is true
   weightUpdateTime += weightUpdatePeriod;
   return weightUpdateTime;
}

#ifdef PV_USE_MPI
int KernelConn::reduceKernels(const int axonID) {
   const int numPatches = numDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;

   //Now uses member variable mpiReductionBuffer
   // pvdata_t * buf = (pvdata_t *) malloc(localSize*sizeof(pvdata_t));
   // if(buf == NULL) {
   //    fprintf(stderr, "KernelConn::reduceKernels:Unable to allocate memory\n");
   //    exit(1);
   // }

   // Copy this column's weights into mpiReductionBuffer
   int idx = 0;
   for (int k = 0; k < numPatches; k++) {
      //PVPatch * p = kernelPatches[axonID][k];
      PVPatch * p = dKernelPatches[axonID][k];
      const pvdata_t * data = p->data;

      const int sxp = p->sx;
      const int syp = p->sy;
      const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < p->nf; f++) {
               mpiReductionBuffer[idx] = data[x*sxp + y*syp + f*sfp];
               idx++;
            }
         }
      }
   }

   // MPI_Allreduce combines all processors' buffers and puts the common result
   // into each processor's buffer.
   Communicator * comm = parent->icCommunicator();
   const MPI_Comm mpi_comm = comm->communicator();
   int ierr;
   ierr = MPI_Allreduce(MPI_IN_PLACE, mpiReductionBuffer, localSize, MPI_FLOAT, MPI_SUM, mpi_comm);
   // TODO error handling

   // mpiReductionBuffer now holds the sum over all processes.
   // Divide by number of processes to get average and copy back to patches
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nProcs = nxProcs * nyProcs;
   idx = 0;
   for (int k = 0; k < numPatches; k++) {
      //PVPatch * p = kernelPatches[axonID][k];
      PVPatch * p = dKernelPatches[axonID][k];
      pvdata_t * data = p->data;

      const int sxp = p->sx;
      const int syp = p->sy;
      const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < p->nf; f++) {
               data[x*sxp + y*syp + f*sfp] = mpiReductionBuffer[idx]/nProcs;
               idx++;
            }
         }
      }
   }

   // free(buf);
   return PV_SUCCESS;
}
#endif // PV_USE_MPI

int KernelConn::correctPIndex(int patchIndex) {
   return kernelIndexToPatchIndex(patchIndex);
}

#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
int KernelConn::gauss2DCalcWeights(PVPatch * wp, int kKernel, int no, int numFlanks,
                                   float shift, float rotate, float aspect, float sigma,
                                   float r2Max, float strength,
                                   float deltaThetaMax, float thetaMax, float bowtieFlag,
                                   float bowtieAngle)
{
   int kPatch;
   kPatch = kernelIndexToPatchIndex(kKernel);
   return HyPerConn::gauss2DCalcWeights(wp, kPatch, no, numFlanks,
                                        shift, rotate, aspect, sigma, r2Max, strength,
                                        deltaThetaMax, thetaMax, bowtieFlag, bowtieAngle);
}

int KernelConn::cocircCalcWeights(PVPatch * wp, int kKernel, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
      float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
      float aspect, float rotate, float sigma, float r2Max, float strength)
{
   int kPatch;
   kPatch = kernelIndexToPatchIndex(kKernel);
   return HyPerConn::cocircCalcWeights(wp, kPatch, noPre, noPost, sigma_cocirc,
         sigma_kurve, sigma_chord, delta_theta_max, cocirc_self, delta_radius_curvature,
         numFlanks, shift, aspect, rotate, sigma, r2Max, strength);
}
#endif // OBSOLETE

int KernelConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   const int num_kernels = numDataPatches();
   if (this->numberOfAxonalArborLists() == 1) {
      status = HyPerConn::normalizeWeights(kernelPatches[arborId], num_kernels, arborId);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      if ( symmetrizeWeightsFlag ){
         status = symmetrizeWeights(kernelPatches[arborId], num_kernels, arborId);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      }
   } // numberOfAxonalArborLists() == 1
   else {
      for (int kPatch = 0; kPatch < numPatches; kPatch++) {
         float sumAll = 0.0f;
         float sum2All = 0.0f;
         float maxAll = 0.0f;
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            float sum, sum2, maxVal;
            status = sumWeights(kernelPatches[kArbor][kPatch], &sum, &sum2, &maxVal);
            sumAll += sum;
            sum2All += sum2;
            maxAll = maxVal > maxAll ? maxVal : maxAll;
         } // kArbor
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            status = scaleWeights(kernelPatches[kArbor][kPatch], sumAll, sum2All, maxAll);
         } // kArbor
      } // kPatch < numPatches
      status = PV_BREAK;
   } // numberOfAxonalArborLists() != 1
   return status;
}

int KernelConn::symmetrizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   printf("Entering KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   assert(pre->clayer->loc.nf==post->clayer->loc.nf);
   PVPatch ** symPatches;
   symPatches = (PVPatch**) calloc(sizeof(PVPatch*), numPatches);
   assert(symPatches != NULL);
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      symPatches[iKernel] = pvpatch_inplace_new(nxp, nyp, nfp);
      assert(symPatches[iKernel] != NULL );
   }
   const int sy = nxp * nfp;
   const float deltaTheta = PI / nfp;
   const float offsetTheta = 0.5f * deltaTheta;
   const int kyMid = nyp / 2;
   const int kxMid = nxp / 2;
   for (int iSymKernel = 0; iSymKernel < numPatches; iSymKernel++) {
      PVPatch * symWp = symPatches[iSymKernel];
      pvdata_t * symW = symWp->data;
      float symTheta = offsetTheta + iSymKernel * deltaTheta;
      for (int kySym = 0; kySym < nyp; kySym++) {
         float dySym = kySym - kyMid;
         for (int kxSym = 0; kxSym < nxp; kxSym++) {
            float dxSym = kxSym - kxMid;
            float distSym = sqrt(dxSym * dxSym + dySym * dySym);
            if (distSym > abs(kxMid > kyMid ? kxMid : kyMid)) {
               continue;
            }
            float dyPrime = dySym * cos(symTheta) - dxSym * sin(symTheta);
            float dxPrime = dxSym * cos(symTheta) + dySym * sin(symTheta);
            for (int kfSym = 0; kfSym < nfp; kfSym++) {
               int kDf = kfSym - iSymKernel;
               int iSymW = kfSym + nfp * kxSym + sy * kySym;
               for (int iKernel = 0; iKernel < nfp; iKernel++) {
                  PVPatch * kerWp = getKernelPatch(arborId, iKernel);
                  pvdata_t * kerW = kerWp->data;
                  int kfRot = iKernel + kDf;
                  if (kfRot < 0) {
                     kfRot = nfp + kfRot;
                  }
                  else {
                     kfRot = kfRot % nfp;
                  }
                  float rotTheta = offsetTheta + iKernel * deltaTheta;
                  float yRot = dyPrime * cos(rotTheta) + dxPrime * sin(rotTheta);
                  float xRot = dxPrime * cos(rotTheta) - dyPrime * sin(rotTheta);
                  yRot += kyMid;
                  xRot += kxMid;
                  // should find nearest neighbors and do weighted average
                  int kyRot = yRot + 0.5f;
                  int kxRot = xRot + 0.5f;
                  int iRotW = kfRot + nfp * kxRot + sy * kyRot;
                  symW[iSymW] += kerW[iRotW] / nfp;
               } // kfRot
            } // kfSymm
         } // kxSym
      } // kySym
   } // iKernel
   const int num_weights = nfp * nxp * nyp;
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      PVPatch * kerWp = kernelPatches[0][iKernel];
      pvdata_t * kerW = kerWp->data;
      PVPatch * symWp = symPatches[iKernel];
      pvdata_t * symW = symWp->data;
      for (int iW = 0; iW < num_weights; iW++) {
         kerW[iW] = symW[iW];
      }
   } // iKernel
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      pvpatch_inplace_delete(symPatches[iKernel]);
   } // iKernel
   free(symPatches);
   printf("Exiting KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   return status;
}

int KernelConn::writeWeights(float time, bool last)
{
   //const int arbor = 0;
   this->fileType = PVP_KERNEL_FILE_TYPE;
   const int numPatches = numDataPatches();
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      if(HyPerConn::writeWeights(kernelPatches[arborId], numPatches, NULL, time, last, arborId))
         return 1;
   }
   return 0;
}

} // namespace PV

