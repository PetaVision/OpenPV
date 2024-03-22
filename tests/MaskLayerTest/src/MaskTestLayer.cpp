#include "MaskTestLayer.hpp"

namespace PV {

MaskTestLayer::MaskTestLayer(const char *name, PVParams *params, Communicator const *comm) {
   ANNLayer::initialize(name, params, comm);
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
   parameters()->ioParamStringRequired(ioFlag, getName(), "maskMethod", &maskMethod);
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
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: \"%s\" is not a valid maskMethod. Options are \"invertLayer\", "
               "\"maskFeatures\", or \"noMaskFeatures\".\n",
               getDescription_c(),
               maskMethod);
      }
      exit(EXIT_FAILURE);
   }
}

Response::Status MaskTestLayer::checkUpdateState(double timef, double dt) {
   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;

   bool isCorrect = true;
   for (int b = 0; b < loc->nbatch; b++) {
      float const *GSynExt  = mLayerInput->getBufferData(b, CHANNEL_EXC); // gated
      float const *GSynInh  = mLayerInput->getBufferData(b, CHANNEL_INH); // gt
      float const *GSynInhB = mLayerInput->getBufferData(b, CHANNEL_INHB); // mask

      // Grab the activity layer of current layer
      // We only care about restricted space

      for (int k = 0; k < getNumNeurons(); k++) {
         if (strcmp(maskMethod, "layer") == 0) {
            if (GSynInhB[k]) {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
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
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
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
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != GSynInh[k]) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
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
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
                             << ": actual value: " << GSynExt[k]
                             << " Expected value: " << GSynInh[k] << ".\n";
                  isCorrect = false;
               }
            }
            else {
               if (GSynExt[k] != 0) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at " << k
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
