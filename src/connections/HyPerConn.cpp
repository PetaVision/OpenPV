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
#include "utils/TransposeWeights.hpp"
#include "weightupdaters/HebbianUpdater.hpp"

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
   char *weightInitTypeString = nullptr;
   parent->parameters()->ioParamString(
         PARAMS_IO_READ,
         name,
         "weightInitType",
         &weightInitTypeString,
         nullptr,
         true /*warnIfAbsent*/);
   // Note: The weightInitType string param gets read both here and by the
   // InitWeights::ioParam_weightInitType() method. It is read here because we need
   // to know the weight init type in order to instantiate the correct class. It is read in
   // InitWeights to store the value, in order to print it into the generated params file.
   // We don't write weightInitType in a HyPerConn method because we'd like to keep
   // all the WeightInitializer params together in the generated file, and
   // BaseConnection::ioParamsFillGroup() calls the components' ioParams() methods
   // in a loop, without knowing which component is which.

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
   char *normalizeMethod     = nullptr;
   parent->parameters()->ioParamString(
         PARAMS_IO_READ, name, "normalizeMethod", &normalizeMethod, nullptr, true /*warnIfAbsent*/);
   // Note: The normalizeMethod string param gets read both here and by the
   // NormalizeBase::ioParam_weightInitType() method. It is read here because we need
   // to know the normalization method in order to instantiate the correct class. It is read in
   // NormalizeBase to store the value, in order to print it into the generated params file.
   // We don't write normalizeMethod in a HyPerConn method because we'd like to keep
   // all the WeightNormalizer params together in the generated file, and
   // BaseConnection::ioParamsFillGroup() calls the components' ioParams() methods
   // in a loop, without knowing which component is which.

   if (normalizeMethod == nullptr) {
      if (parent->columnId() == 0) {
         Fatal().printf(
               "%s: specifying a normalizeMethod string is required.\n", getDescription_c());
      }
   }
   if (!strcmp(normalizeMethod, "")) {
      free(normalizeMethod);
      normalizeMethod = strdup("none");
   }
   if (strcmp(normalizeMethod, "none")) {
      BaseObject *baseObj = Factory::instance()->createByKeyword(normalizeMethod, name, parent);
      if (baseObj == nullptr) {
         if (parent->columnId() == 0) {
            Fatal() << getDescription_c() << ": normalizeMethod \"" << normalizeMethod
                    << "\" is not recognized." << std::endl;
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      normalizer = dynamic_cast<NormalizeBase *>(baseObj);
      if (normalizer == nullptr) {
         pvAssert(baseObj);
         if (parent->columnId() == 0) {
            Fatal() << getDescription_c() << ": normalizeMethod \"" << normalizeMethod
                    << "\" is not a recognized normalization method." << std::endl;
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   free(normalizeMethod);
   return normalizer;
}

BaseDelivery *HyPerConn::createDeliveryObject() { return new HyPerDeliveryFacade(name, parent); }

BaseWeightUpdater *HyPerConn::createWeightUpdater() { return new HebbianUpdater(name, parent); }

int HyPerConn::initializeState() {
   auto *initWeights =
         mapLookupByType<InitWeights>(mComponentTable.getObjectMap(), getDescription());
   int status = initWeights->initializeWeights();
   return status;
}

} // namespace PV
