/*
 * MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include <cstring>

namespace PV {

MomentumConn::MomentumConn() { initialize_base(); }

MomentumConn::MomentumConn(
      const char *name,
      HyPerCol *hc,
      InitWeights *weightInitializer,
      NormalizeBase *weightNormalizer)
      : HyPerConn() {
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

MomentumConn::~MomentumConn() {
   if (momentumMethod) {
      free(momentumMethod);
   }
   if (prev_dwDataStart) {
      free(prev_dwDataStart[0]);
      free(prev_dwDataStart);
   }
}

int MomentumConn::initialize_base() {
   prev_dwDataStart = NULL;
   momentumTau      = .25;
   momentumMethod   = NULL;
   momentumDecay    = 0;
   timeBatchPeriod  = 1;
   timeBatchIdx     = -1;
   return PV_SUCCESS;
}

int MomentumConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   if (status == PV_POSTPONE) {
      return status;
   }
   if (!plasticityFlag)
      return status;
   int sx       = nfp;
   int sy       = sx * nxp;
   int sp       = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   // Allocate dw buffer for previous dw
   prev_dwDataStart = (pvwdata_t **)pvCalloc(numAxons, sizeof(pvwdata_t *));
   prev_dwDataStart[0] =
         (pvwdata_t *)pvCalloc(numAxons * nxp * nyp * nfp * nPatches, sizeof(pvwdata_t));
   for (int arborId = 0; arborId < numAxons; arborId++) {
      prev_dwDataStart[arborId] = (prev_dwDataStart[0] + sp * nPatches * arborId);
      assert(prev_dwDataStart[arborId] != NULL);
   } // loop over arbors

   return PV_SUCCESS;
}

int MomentumConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_momentumMethod(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumDecay(ioFlag);
   ioParam_batchPeriod(ioFlag);
   return status;
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag) {
   if (plasticityFlag) {
      float defaultVal = 0;
      if (strcmp(momentumMethod, "simple") == 0) {
         defaultVal = .25;
      }
      else if (strcmp(momentumMethod, "viscosity") == 0) {
         defaultVal = 100;
      }
      else if (strcmp(momentumMethod, "alex") == 0) {
         defaultVal = .9;
      }

      parent->parameters()->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, defaultVal);
   }
}

/**
 * @brief momentumMethod: The momentum method to use
 * @details Assuming a = dwMax * pre * post
 * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
 * viscosity: deltaW(t) = (deltaW(t-1) * exp(-1/momentumTau)) + a
 * alex: deltaW(t) = momentumTau * delta(t-1) - momentumDecay * dwMax * w(t) - a
 */
void MomentumConn::ioParam_momentumMethod(enum ParamsIOFlag ioFlag) {
   if (plasticityFlag) {
      parent->parameters()->ioParamStringRequired(ioFlag, name, "momentumMethod", &momentumMethod);
      if (strcmp(momentumMethod, "simple") != 0 && strcmp(momentumMethod, "viscosity") != 0
          && strcmp(momentumMethod, "alex")) {
         pvError() << "MomentumConn " << name << ": momentumMethod of " << momentumMethod
                   << " is not known, options are \"simple\", \"viscosity\", and \"alex\"\n";
      }
   }
}

void MomentumConn::ioParam_momentumDecay(enum ParamsIOFlag ioFlag) {
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "momentumDecay", &momentumDecay, momentumDecay);
      if (momentumDecay < 0 || momentumDecay > 1) {
         pvError() << "MomentumConn " << name
                   << ": momentumDecay must be between 0 and 1 inclusive\n";
      }
   }
}

void MomentumConn::ioParam_batchPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (plasticityFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "batchPeriod", &timeBatchPeriod, timeBatchPeriod);
   }
}

int MomentumConn::calc_dW() {
   assert(plasticityFlag);
   int status;
   timeBatchIdx = (timeBatchIdx + 1) % timeBatchPeriod;

   // Clear at time 0, update at time timeBatchPeriod - 1
   bool need_update_w = false;
   bool need_clear_dw = false;
   if (timeBatchIdx == 0) {
      need_clear_dw = true;
   }

   // If updating next timestep, update weights here
   if ((timeBatchIdx + 1) % timeBatchPeriod == 0) {
      need_update_w = true;
   }

   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      // Clear every batch period
      if (need_clear_dw) {
         status = initialize_dW(arborId);
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
   }

   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      // Sum up parts every timestep
      status = update_dW(arborId);
      if (status == PV_BREAK) {
         break;
      }
      assert(status == PV_SUCCESS);
   }

   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      // Reduce only when we need to update
      if (need_update_w) {
         status = reduce_dW(arborId);
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
   }

   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      // Normalize only when reduced
      if (need_update_w) {
         status = normalize_dW(arborId);
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
   }
   return PV_SUCCESS;
}

int MomentumConn::updateWeights(int arborId) {
   if (timeBatchIdx != timeBatchPeriod - 1) {
      return PV_SUCCESS;
   }
   // Add momentum right before updateWeights
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      applyMomentum(arborId);
   }

   // Saved to prevweights
   assert(prev_dwDataStart);
   std::memcpy(
         *prev_dwDataStart,
         *get_dwDataStart(),
         sizeof(pvwdata_t) * numberOfAxonalArborLists() * nxp * nyp * nfp * getNumDataPatches());

   // add dw to w
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      pvwdata_t *w_data_start = get_wDataStart(kArbor);
      for (long int k = 0; k < patchStartIndex(getNumDataPatches()); k++) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

int MomentumConn::applyMomentum(int arbor_ID) {
   int nExt              = preSynapticLayer()->getNumExtended();
   const PVLayerLoc *loc = preSynapticLayer()->getLayerLoc();
   if (sharedWeights) {
      int numKernels = getNumDataPatches();
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
         pvwdata_t *dwdata_start  = get_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t *prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t *wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
         if (!strcmp(momentumMethod, "simple")) {
            for (int k = 0; k < nxp * nyp * nfp; k++) {
               dwdata_start[k] += momentumTau * prev_dw_start[k] - momentumDecay * wdata_start[k];
            }
         }
         else if (!strcmp(momentumMethod, "viscosity")) {
            for (int k = 0; k < nxp * nyp * nfp; k++) {
               dwdata_start[k] = (prev_dw_start[k] * expf(-1.0f / momentumTau)) + dwdata_start[k]
                                 - momentumDecay * wdata_start[k];
            }
         }
         else if (!strcmp(momentumMethod, "alex")) {
            for (int k = 0; k < nxp * nyp * nfp; k++) {
               // weight_inc[i] := momW * weight_inc[i-1] - wc * epsW * weights[i-1] + epsW *
               // weight_grads[i]
               //   weights[i] := weights[i-1] + weight_inc[i]
               dwdata_start[k] = momentumTau * prev_dw_start[k]
                                 - momentumDecay * getDWMax() * wdata_start[k] + dwdata_start[k];
            }
         }
      }
   }
   else {
      pvWarn() << "Momentum not implemented for non-shared weights, not implementing momentum\n";
   }
   return PV_SUCCESS;
}

// TODO checkpointing not working with batching, must write checkpoint exactly at period
int MomentumConn::checkpointWrite(const char *cpDir) {
   HyPerConn::checkpointWrite(cpDir);
   if (!plasticityFlag)
      return PV_SUCCESS;
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_prev_dW.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      if (parent->getCommunicator()->commRank() == 0) {
         pvErrorNoExit().printf(
               "HyPerConn::checkpointFilename: path \"%s/%s_W.pvp\" is too long.\n", cpDir, name);
      }
      abort();
   }
   PVPatch ***patches_arg = sharedWeights ? NULL : get_wPatches();
   int status             = writeWeights(
         patches_arg,
         prev_dwDataStart,
         getNumDataPatches(),
         filename,
         parent->simulationTime(),
         writeCompressedCheckpoints,
         /*last*/ true);
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
}

int MomentumConn::checkpointRead(const char *cpDir, double *timeptr) {
   HyPerConn::checkpointRead(cpDir, timeptr);
   if (!plasticityFlag)
      return PV_SUCCESS;
   clearWeights(prev_dwDataStart, getNumDataPatches(), nxp, nyp, nfp);
   char *path             = parent->pathInCheckpoint(cpDir, getName(), "_prev_dW.pvp");
   PVPatch ***patches_arg = sharedWeights ? NULL : get_wPatches();
   double filetime        = 0.0;
   int status             = PV::readWeights(
         patches_arg,
         prev_dwDataStart,
         numberOfAxonalArborLists(),
         getNumDataPatches(),
         nxp,
         nyp,
         nfp,
         path,
         parent->getCommunicator(),
         &filetime,
         pre->getLayerLoc());
   if (parent->columnId() == 0 && timeptr && *timeptr != filetime) {
      pvWarn().printf(
            "\"%s\" checkpoint has timestamp %g instead of the expected value %g.\n",
            path,
            filetime,
            *timeptr);
   }
   free(path);
   return status;
}

} // end namespace PV
