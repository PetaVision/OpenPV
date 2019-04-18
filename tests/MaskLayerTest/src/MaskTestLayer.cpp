#include "MaskTestLayer.hpp"

namespace PV {

MaskTestLayer::MaskTestLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   ANNLayer::initialize(name, hc);
}

MaskTestLayer::~MaskTestLayer() {
   if (maskMethod) {
      free(maskMethod);
   }
}
int MaskTestLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_maskMethod(ioFlag);
   return status;
}

void MaskTestLayer::ioParam_maskMethod(enum ParamsIOFlag ioFlag) {
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
               "%s: \"%s\" is not a valid maskMethod. Options are \"invertLayer\", "
               "\"maskFeatures\", or \"noMaskFeatures\".\n",
               getDescription_c(),
               maskMethod);
      }
      exit(-1);
   }
}

Response::Status MaskTestLayer::updateState(double timef, double dt) {
   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;

   bool isCorrect = true;
   for (int b = 0; b < loc->nbatch; b++) {
      float *GSynExt  = getChannel(CHANNEL_EXC) + b * getNumNeurons(); // gated
      float *GSynInh  = getChannel(CHANNEL_INH) + b * getNumNeurons(); // gt
      float *GSynInhB = getChannel(CHANNEL_INHB) + b * getNumNeurons(); // mask

      // Grab the activity layer of current layer
      // We only care about restricted space

      for (int k = 0; k < getNumNeurons(); k++) {
         if (strcmp(maskMethod, "layer") == 0) {
            // ErrorLog() << "Connection " << name << " Mismatch at " << k << ": actual value:
            // " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
            if (GSynInhB[k]) {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                  isCorrect = false;
               }
            }
         }
         else if (strcmp(maskMethod, "invertLayer") == 0) {
            // ErrorLog() << "Connection " << name << " Mismatch at " << k << ": actual value:
            // " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
            if (!GSynInhB[k]) {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                  isCorrect = false;
               }
            }
         }
         else if (strcmp(maskMethod, "maskFeatures") == 0) {
            int featureIdx = featureIndex(k, nx, ny, nf);
            // Param files specifies idxs 0 and 2 out of 3 total features
            if (featureIdx == 0 || featureIdx == 2) {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
         }
         else if (strcmp(maskMethod, "noMaskFeatures") == 0) {
            int featureIdx = featureIndex(k, nx, ny, nf);
            // Param files specifies idxs 0 and 2 out of 3 total features
            if (featureIdx == 0 || featureIdx == 2) {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << name << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                  isCorrect = false;
               }
            }
         }
      }
   }

   if (!isCorrect) {
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

} /* namespace PV */
