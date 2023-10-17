/*
 * LayerGeometry.cpp
 *
 *  Created on: Apr 6, 2018
 *      Author: pschultz
 */

#include "LayerGeometry.hpp"
#include "observerpattern/ObserverTable.hpp"
#include <cmath>
#include <cstring>

namespace PV {

LayerGeometry::LayerGeometry(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

LayerGeometry::LayerGeometry() {}

LayerGeometry::~LayerGeometry() {}

void LayerGeometry::initialize(char const *name, PVParams *params, Communicator const *comm) {
   std::memset(&mLayerLoc, 0, sizeof(mLayerLoc));
   BaseObject::initialize(name, params, comm);
}

void LayerGeometry::setObjectType() { mObjectType = "LayerGeometry"; }

int LayerGeometry::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_broadcastFlag(ioFlag);
   ioParam_nxScale(ioFlag);
   ioParam_nyScale(ioFlag);
   ioParam_nf(ioFlag);
   return PV_SUCCESS;
}

void LayerGeometry::ioParam_broadcastFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "broadcastFlag", &mBroadcastFlag, mBroadcastFlag);
}

void LayerGeometry::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "broadcastFlag"));
   if (mBroadcastFlag) {
      if (ioFlag == PARAMS_IO_READ and parameters()->present(name, "broadcastFlag")) {
         WarnLog().printf(
               "%s has broadcastFlag = true; therefore nxScale is ignored.\n",
               getDescription_c());
      }
   }
   else {
      parameters()->ioParamValue(ioFlag, name, "nxScale", &mNxScale, mNxScale);
   }
}

void LayerGeometry::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "broadcastFlag"));
   if (mBroadcastFlag) {
      if (ioFlag == PARAMS_IO_READ and parameters()->present(name, "broadcastFlag")) {
         WarnLog().printf(
               "%s has broadcastFlag = true; therefore nyScale is ignored.\n",
               getDescription_c());
      }
   }
   else {
      parameters()->ioParamValue(ioFlag, name, "nyScale", &mNyScale, mNyScale);
   }
}

void LayerGeometry::ioParam_nf(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "nf", &mNumFeatures, mNumFeatures);
}

Response::Status
LayerGeometry::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   setLayerLoc(&mLayerLoc, message);

   mNumNeurons           = mLayerLoc.nx * mLayerLoc.ny * mLayerLoc.nf;
   mNumNeuronsAllBatches = mNumNeurons * mLayerLoc.nbatch;

   updateNumExtended();

   double xScaled = -log2((double)mNxScale);
   mXScale        = (int)nearbyint(xScaled);

   double yScaled = -log2((double)mNyScale);
   mYScale        = (int)nearbyint(yScaled);

   return Response::SUCCESS;
}

void LayerGeometry::updateNumExtended() {
   int const nxExt        = (mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt);
   int const nyExt        = (mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up);
   mNumExtended           = nxExt * nyExt * mLayerLoc.nf;
   mNumExtendedAllBatches = mNumExtended * mLayerLoc.nbatch;
}

void LayerGeometry::setLayerLoc(
      PVLayerLoc *layerLoc,
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = PV_SUCCESS;

   Communicator const *icComm = mCommunicator;

   if (mBroadcastFlag) {
      int nxLayer        = icComm->numCommColumns();
      mNxScale           = nxLayer / message->mNxGlobal;
      layerLoc->nxGlobal = nxLayer;

      int nyLayer        = icComm->numCommRows();
      mNyScale           = nyLayer / message->mNyGlobal;
      layerLoc->nyGlobal = nyLayer;
      // For broadcast layers, NxScale and NyScale shouldn't be used; they're set here just in case
   }
   else {
      int statusx = calculateScaledSize(&layerLoc->nxGlobal, mNxScale, message->mNxGlobal, 'x'); 
      int statusy = calculateScaledSize(&layerLoc->nyGlobal, mNyScale, message->mNyGlobal, 'y');
      if (statusx != PV_SUCCESS or statusy != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      layerLoc->nbatchGlobal = message->mNBatchGlobal;
      layerLoc->nf           = mNumFeatures;

      // partition input space based on the number of processor columns and rows
      status = setLocalLayerLocFields(layerLoc, icComm, std::string(getName()));

      // halo is initialized to zero in constructor, and can be changed by calls
      // to requireMarginWidth. We don't change the values here.
   }

   // If there is an error, make sure that MPI doesn't kill the run before process 0 reports
   // the error.
   MPI_Barrier(icComm->communicator());
   if (status != PV_SUCCESS) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog().printf("setLayerLoc failed for layer \"%s\".\n", getName());
      }
      exit(EXIT_FAILURE);
   }
}

int LayerGeometry::calculateScaledSize(int *scaledSize, float scaleFactor, int baseSize, char axis) {
   float scaledSizeFloat = scaleFactor * static_cast<float>(baseSize);
   float closestInt      = std::nearbyint(scaledSizeFloat);
   float discrep         = scaledSizeFloat - closestInt;
   int status = PV_SUCCESS;
   if (discrep and std::abs(discrep) > 0.0001f) {
      ErrorLog(errorMessage);
      errorMessage.printf(
            "n%cScale of layer \"%s\" is incompatible with size of column.\n",
            axis,
            getName());
      errorMessage.printf(
            "Column n%c=%d multiplied by n%cScale=%f must be an integer.\n",
            axis,
            baseSize,
            axis,
            (double)scaleFactor);
      status = PV_FAILURE;
   }
   *scaledSize = static_cast<int>(closestInt);
   return status;
}

int LayerGeometry::setLocalLayerLocFields(PVLayerLoc *layerLoc, Communicator const *icComm, std::string const &label) {
   int status = PV_SUCCESS;

   bool isRoot = icComm->globalCommRank() == 0;
   if (checkRemainder(layerLoc->nxGlobal, icComm->numCommColumns(), std::string("x"), label, isRoot)) {
      status = PV_FAILURE;
   }
   if (checkRemainder(layerLoc->nyGlobal, icComm->numCommRows(), std::string("y"), label, isRoot)) {
      status = PV_FAILURE;
   }
   if (checkRemainder(layerLoc->nbatchGlobal, icComm->numCommBatches(), std::string("batch"), label, isRoot)) {
      status = PV_FAILURE;
   }

   if (status == PV_SUCCESS) {
      layerLoc->nx = layerLoc->nxGlobal / icComm->numCommColumns();
      assert(layerLoc->nxGlobal == layerLoc->nx * icComm->numCommColumns());

      layerLoc->ny = layerLoc->nyGlobal / icComm->numCommRows();
      assert(layerLoc->nyGlobal == layerLoc->ny * icComm->numCommRows());

      layerLoc->nbatch = layerLoc->nbatchGlobal / icComm->numCommBatches();
      assert(layerLoc->nbatchGlobal == layerLoc->nbatch * icComm->numCommBatches());

      layerLoc->kx0 = layerLoc->nx * icComm->commColumn();
      layerLoc->ky0 = layerLoc->ny * icComm->commRow();
      layerLoc->kb0 = layerLoc->nbatch * icComm->commBatch();
   }

   return status;
}

bool LayerGeometry::checkRemainder(
      int globalSize, int numProcesses, std::string axis, std::string const &label, bool printErr) {
   bool hasRemainder = false;

   if (globalSize % numProcesses != 0) {
      if (printErr) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Size of layer \"%s\" is not compatible with the mpi configuration.\n",
               label.c_str());
         errorMessage.printf(
               "The layer has %d pixels in the %s dimension, and there are %d mpi processes\n",
               globalSize,
               axis.c_str(),
               numProcesses);
         errorMessage.printf(
               "in that dimension, but %d does not divide %d.\n", numProcesses, globalSize);
      }
      hasRemainder = true;
   }
   return hasRemainder;
}

void LayerGeometry::requireMarginWidth(int marginWidthNeeded, char axis) {
   int *startMargin = nullptr, *endMargin = nullptr; // lt/rt for x-axis; dn/up for y-axis
   switch (axis) {
      case 'x':
         startMargin = &mLayerLoc.halo.lt;
         endMargin   = &mLayerLoc.halo.rt;
         break;
      case 'y':
         startMargin = &mLayerLoc.halo.dn;
         endMargin   = &mLayerLoc.halo.up;
         break;
   }
   pvAssert(*startMargin == *endMargin);
   if (*startMargin < marginWidthNeeded) {
      InfoLog().printf(
            "%s adjusting %c-margin from %d to %d\n",
            getDescription_c(),
            axis,
            *startMargin,
            marginWidthNeeded);
      *startMargin = marginWidthNeeded;
      *endMargin   = marginWidthNeeded;
      for (auto &g : mSynchronizedMargins) {
         g->requireMarginWidth(marginWidthNeeded, axis);
      }
   }
   pvAssert(*startMargin >= marginWidthNeeded);

   // Update numExtended and numExtendedAllBatches.
   updateNumExtended();
}

void LayerGeometry::synchronizeMarginWidths(LayerGeometry *geometry1, LayerGeometry *geometry2) {
   geometry1->synchronizeMarginWidth(geometry2);
   geometry2->synchronizeMarginWidth(geometry1);
}

void LayerGeometry::synchronizeMarginWidth(LayerGeometry *otherGeometry) {
   if (otherGeometry == this) {
      return;
   }
   for (auto &g : mSynchronizedMargins) {
      if (g == otherGeometry) {
         return;
      }
   }
   mSynchronizedMargins.push_back(otherGeometry);
   requireMarginWidth(otherGeometry->getLayerLoc()->halo.lt, 'x');
   requireMarginWidth(otherGeometry->getLayerLoc()->halo.dn, 'y');
   otherGeometry->requireMarginWidth(getLayerLoc()->halo.lt, 'x');
   otherGeometry->requireMarginWidth(getLayerLoc()->halo.dn, 'y');
}

} // namespace PV
