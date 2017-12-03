/*
 * HyPerConn.cpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#include "HyPerConn.hpp"
#include "columns/HyPerCol.hpp"
#include "delivery/HyPerDeliveryFacade.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

HyPerConn::HyPerConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

HyPerConn::HyPerConn() {}

HyPerConn::~HyPerConn() {}

int HyPerConn::initialize(char const *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

void HyPerConn::defineComponents() {
   BaseConnection::defineComponents();
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addObserver(mWeightsPair);
   }
   mWeightInitializer = createWeightInitializer();
   if (mWeightInitializer) {
      addObserver(mWeightInitializer);
   }
   mWeightNormalizer = createWeightNormalizer();
   if (mWeightNormalizer) {
      addObserver(mWeightNormalizer);
   }
}

WeightsPair *HyPerConn::createWeightsPair() { return new WeightsPair(name, parent); }

InitWeights *HyPerConn::createWeightInitializer() {
   char *weightInitTypeString;
   parent->parameters()->ioParamString(
         PARAMS_IO_READ,
         name,
         "weightInitType",
         &weightInitTypeString,
         nullptr,
         true /*warnIfAbsent*/);
   FatalIf(
         weightInitTypeString == nullptr or weightInitTypeString[0] == '\0',
         "%s must set weightInitType.\n",
         getDescription_c());
   BaseObject *baseObject = nullptr;
   try {
      baseObject = Factory::instance()->createByKeyword(weightInitTypeString, name, parent);
   } catch (const std::exception &e) {
      Fatal() << getDescription() << " unable to create weightInitializer: " << e.what() << "\n";
   }
   auto *weightInitializer = dynamic_cast<InitWeights *>(baseObject);
   FatalIf(
         weightInitializer == nullptr,
         "%s unable to create weightInitializer: %s is not an InitWeights keyword.\n",
         getDescription_c(),
         weightInitTypeString);

   free(weightInitTypeString);

   return weightInitializer;
}

NormalizeBase *HyPerConn::createWeightNormalizer() {
   NormalizeBase *normalizer = nullptr;
   parent->parameters()->ioParamString(
         PARAMS_IO_READ, name, "normalizeMethod", &mNormalizeMethod, NULL, true
         /*warnIfAbsent*/);
   if (mNormalizeMethod == nullptr) {
      if (parent->columnId() == 0) {
         Fatal().printf(
               "%s: specifying a normalizeMethod string is required.\n", getDescription_c());
      }
   }
   if (!strcmp(mNormalizeMethod, "")) {
      free(mNormalizeMethod);
      mNormalizeMethod = strdup("none");
   }
   if (strcmp(mNormalizeMethod, "none")) {
      BaseObject *baseObj = Factory::instance()->createByKeyword(mNormalizeMethod, name, parent);
      if (baseObj == nullptr) {
         if (parent->columnId() == 0) {
            Fatal() << getDescription_c() << ": normalizeMethod \"" << mNormalizeMethod
                    << "\" is not recognized." << std::endl;
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      normalizer = dynamic_cast<NormalizeBase *>(baseObj);
      if (normalizer == nullptr) {
         pvAssert(baseObj);
         if (parent->columnId() == 0) {
            Fatal() << getDescription_c() << ": normalizeMethod \"" << mNormalizeMethod
                    << "\" is not a recognized normalization method." << std::endl;
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   return normalizer;
}

BaseDelivery *HyPerConn::createDeliveryObject() { return new HyPerDeliveryFacade(name, parent); }

// WeightUpdater *HyPerConn::createWeightUpdater() {
//    return new WeightUpdater(name, parent);
// }

int HyPerConn::initializeState() {
   auto *initWeights =
         mapLookupByType<InitWeights>(mComponentTable.getObjectMap(), getDescription());
   return initWeights->initializeWeights();
}

} // namespace PV
