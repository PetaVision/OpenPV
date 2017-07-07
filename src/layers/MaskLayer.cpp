
/*
 * MaskLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#include "MaskLayer.hpp"

namespace PV {

MaskLayer::MaskLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

MaskLayer::MaskLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

MaskLayer::~MaskLayer() {
   if (maskLayerName) {
      free(maskLayerName);
   }
   if (features) {
      free(features);
   }
   if (maskMethod) {
      free(maskMethod);
   }
}

int MaskLayer::initialize_base() {
   maskLayerName = NULL;
   maskLayer     = NULL;
   maskMethod    = NULL;
   features      = NULL;

   return PV_SUCCESS;
}

int MaskLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_maskMethod(ioFlag);
   ioParam_maskLayerName(ioFlag);
   ioParam_featureIdxs(ioFlag);
   return status;
}

void MaskLayer::ioParam_maskMethod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "maskMethod", &maskMethod);
   // Check valid methods
   if (strcmp(maskMethod, "layer") == 0) {
   }
   else if (strcmp(maskMethod, "invertLayer") == 0) {
   }
   else if (strcmp(maskMethod, "maskFeatures") == 0) {
   }
   else if (strcmp(maskMethod, "noMaskFeatures") == 0) {
   }
   else {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: \"%s\" is not a valid maskMethod. Options are \"layer\", \"invertLayer\", "
               "\"maskFeatures\", or \"noMaskFeatures\".\n",
               getDescription_c(),
               maskMethod);
      }
      exit(-1);
   }
}

void MaskLayer::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if (strcmp(maskMethod, "layer") == 0 || strcmp(maskMethod, "invertLayer") == 0) {
      parent->parameters()->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
   }
}

void MaskLayer::ioParam_featureIdxs(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if (strcmp(maskMethod, "maskFeatures") == 0 || strcmp(maskMethod, "noMaskFeatures") == 0) {
      parent->parameters()->ioParamArray(
            ioFlag, name, "featureIdxs", &features, &numSpecifiedFeatures);
      if (numSpecifiedFeatures == 0) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: MaskLayer must specify at least one feature for maskMethod \"%s\".\n",
                  getDescription_c(),
                  maskMethod);
         }
         exit(-1);
      }
   }
}

int MaskLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = ANNLayer::communicateInitInfo(message);
   if (strcmp(maskMethod, "layer") == 0 || strcmp(maskMethod, "invertLayer") == 0) {
      maskLayer = message->lookup<HyPerLayer>(std::string(maskLayerName));
      if (maskLayer == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                  getDescription_c(),
                  maskLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc *maskLoc = maskLayer->getLayerLoc();
      const PVLayerLoc *loc     = getLayerLoc();
      assert(maskLoc != NULL && loc != NULL);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (parent->columnId() == 0) {
            ErrorLog(errorMessage);
            errorMessage.printf(
                  "%s: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                  getDescription_c(),
                  maskLayerName);
            errorMessage.printf(
                  "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                  maskLoc->nxGlobal,
                  maskLoc->nyGlobal,
                  maskLoc->nf,
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      if (maskLoc->nf != 1 && maskLoc->nf != loc->nf) {
         if (parent->columnId() == 0) {
            ErrorLog(errorMessage);
            errorMessage.printf(
                  "%s: maskLayerName \"%s\" must either have the same number of features as this "
                  "layer, or one feature.\n",
                  getDescription_c(),
                  maskLayerName);
            errorMessage.printf(
                  "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                  maskLoc->nxGlobal,
                  maskLoc->nyGlobal,
                  maskLoc->nf,
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      assert(maskLoc->nx == loc->nx && maskLoc->ny == loc->ny);
   }
   else {
      // Check for in bounds featureIdxs
      assert(features);
      const PVLayerLoc *loc = getLayerLoc();
      for (int f = 0; f < numSpecifiedFeatures; f++) {
         if (features[f] < 0 || features[f] >= loc->nf) {
            Fatal() << "Specified feature " << features[f] << "out of bounds\n";
         }
      }
   }

   return status;
}

int MaskLayer::updateState(double time, double dt) {
   ANNLayer::updateState(time, dt);

   float *A              = getCLayer()->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   const PVLayerLoc *loc = getLayerLoc();

   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int num_neurons = nx * ny * nf;
   int nbatch      = loc->nbatch;

   int method                       = -1;
   const int METHOD_LAYER           = 0;
   const int METHOD_INVERT_LAYER    = 1;
   const int METHOD_FEATURES        = 2;
   const int METHOD_INVERT_FEATURES = 3;

   if (strcmp(maskMethod, "layer") == 0) {
      method = METHOD_LAYER;
   }
   else if (strcmp(maskMethod, "invertLayer") == 0) {
      method = METHOD_INVERT_LAYER;
   }
   else if (strcmp(maskMethod, "maskFeatures") == 0) {
      method = METHOD_FEATURES;
   }
   else if (strcmp(maskMethod, "noMaskFeatures") == 0) {
      method = METHOD_INVERT_FEATURES;
   }

   for (int b = 0; b < nbatch; b++) {
      float *ABatch = A + b * getNumExtended();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int ni = 0; ni < num_neurons; ni++) {
         int kThisRes = ni;
         int kThisExt = kIndexExtended(
               ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         float maskVal = 1;

         switch (method) {
            case METHOD_LAYER: {
               const PVLayerLoc *maskLoc = maskLayer->getLayerLoc();
               float *maskActivity       = maskLayer->getActivity();
               float *maskActivityBatch  = maskActivity + b * maskLayer->getNumExtended();
               int kMaskRes;
               if (maskLoc->nf == 1) {
                  kMaskRes = ni / nf;
               }
               else {
                  kMaskRes = ni;
               }
               int kMaskExt = kIndexExtended(
                     kMaskRes,
                     nx,
                     ny,
                     maskLoc->nf,
                     maskLoc->halo.lt,
                     maskLoc->halo.rt,
                     maskLoc->halo.dn,
                     maskLoc->halo.up);
               maskVal = maskActivityBatch[kMaskExt];
            } break;
            case METHOD_INVERT_LAYER: {
               const PVLayerLoc *maskLoc = maskLayer->getLayerLoc();
               float *maskActivity       = maskLayer->getActivity();
               float *maskActivityBatch  = maskActivity + b * maskLayer->getNumExtended();
               int kMaskRes;
               if (maskLoc->nf == 1) {
                  kMaskRes = ni / nf;
               }
               else {
                  kMaskRes = ni;
               }
               int kMaskExt = kIndexExtended(
                     kMaskRes,
                     nx,
                     ny,
                     maskLoc->nf,
                     maskLoc->halo.lt,
                     maskLoc->halo.rt,
                     maskLoc->halo.dn,
                     maskLoc->halo.up);
               if (maskActivityBatch[kMaskExt]) {
                  maskVal = 0;
               }
            } break;
            case METHOD_FEATURES: {
               // Calculate feature index of ni
               int featureNum = featureIndex(ni, nx, ny, nf);
               maskVal        = 1; // If nothing specified, copy everything
               for (int specF = 0; specF < numSpecifiedFeatures; specF++) {
                  if (featureNum == features[specF]) {
                     maskVal = 0;
                     break;
                  }
               }
            } break;
            case METHOD_INVERT_FEATURES: {
               // Calculate feature index of ni
               int featureNum = featureIndex(ni, nx, ny, nf);
               maskVal        = 0; // If nothing specified, copy nothing
               for (int specF = 0; specF < numSpecifiedFeatures; specF++) {
                  if (featureNum == features[specF]) {
                     maskVal = 1;
                     break;
                  }
               }
            } break;
            default: break;
         }

         // Set value to 0, otherwise, updateState from ANNLayer should have taken care of it
         if (maskVal == 0) {
            ABatch[kThisExt] = 0;
         }
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
