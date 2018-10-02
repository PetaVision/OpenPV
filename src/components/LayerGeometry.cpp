/*
 * LayerGeometry.cpp
 *
 *  Created on: Apr 6, 2018
 *      Author: pschultz
 */

#include "LayerGeometry.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObserverTableComponent.hpp"
#include <cstring>

namespace PV {

LayerGeometry::LayerGeometry(char const *name, HyPerCol *hc) { initialize(name, hc); }

LayerGeometry::LayerGeometry() {}

LayerGeometry::~LayerGeometry() {}

int LayerGeometry::initialize(char const *name, HyPerCol *hc) {
   std::memset(&mLayerLoc, 0, sizeof(mLayerLoc));
   return BaseObject::initialize(name, hc);
}

void LayerGeometry::setObjectType() { mObjectType = "LayerGeometry"; }

int LayerGeometry::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_nxScale(ioFlag);
   ioParam_nyScale(ioFlag);
   ioParam_nf(ioFlag);
   return PV_SUCCESS;
}

void LayerGeometry::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "nxScale", &mNxScale, mNxScale);
}

void LayerGeometry::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "nyScale", &mNyScale, mNyScale);
}

void LayerGeometry::ioParam_nf(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "nf", &mNumFeatures, mNumFeatures);
}

Response::Status
LayerGeometry::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   communicateLayerGeometry(message);
   // At this point, nxScale, nyScale, and nfScale are known.
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

void LayerGeometry::communicateLayerGeometry(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {}

void LayerGeometry::setLayerLoc(
      PVLayerLoc *layerLoc,
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = PV_SUCCESS;

   Communicator *icComm = parent->getCommunicator();

   float nxglobalfloat = mNxScale * message->mNxGlobal;
   layerLoc->nxGlobal  = (int)nearbyintf(nxglobalfloat);
   if (std::fabs(nxglobalfloat - layerLoc->nxGlobal) > 0.0001f) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "nxScale of layer \"%s\" is incompatible with size of column.\n", getName());
         errorMessage.printf(
               "Column nx %d multiplied by nxScale %f must be an integer.\n",
               (double)message->mNxGlobal,
               (double)mNxScale);
      }
      status = PV_FAILURE;
   }

   float nyglobalfloat = mNyScale * message->mNyGlobal;
   layerLoc->nyGlobal  = (int)nearbyintf(nyglobalfloat);
   if (std::fabs(nyglobalfloat - layerLoc->nyGlobal) > 0.0001f) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "nyScale of layer \"%s\" is incompatible with size of column.\n", getName());
         errorMessage.printf(
               "Column ny %d multiplied by nyScale %f must be an integer.\n",
               (double)message->mNyGlobal,
               (double)mNyScale);
      }
      status = PV_FAILURE;
   }

   // partition input space based on the number of processor
   // columns and rows
   //

   if (layerLoc->nxGlobal % icComm->numCommColumns() != 0) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Size of HyPerLayer \"%s\" is not compatible with the mpi configuration.\n", name);
         errorMessage.printf(
               "The layer has %d pixels horizontally, and there are %d mpi processes in a row, but "
               "%d does not divide %d.\n",
               layerLoc->nxGlobal,
               icComm->numCommColumns(),
               icComm->numCommColumns(),
               layerLoc->nxGlobal);
      }
      status = PV_FAILURE;
   }
   if (layerLoc->nyGlobal % icComm->numCommRows() != 0) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Size of HyPerLayer \"%s\" is not compatible with the mpi configuration.\n", name);
         errorMessage.printf(
               "The layer has %d pixels vertically, and there are %d mpi processes in a column, "
               "but %d does not divide %d.\n",
               layerLoc->nyGlobal,
               icComm->numCommRows(),
               icComm->numCommRows(),
               layerLoc->nyGlobal);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the
   // run before process 0 reports the error.
   if (status != PV_SUCCESS) {
      if (icComm->globalCommRank() == 0) {
         ErrorLog().printf("setLayerLoc failed for %s.\n", getDescription_c());
      }
      exit(EXIT_FAILURE);
   }
   layerLoc->nx = layerLoc->nxGlobal / icComm->numCommColumns();
   layerLoc->ny = layerLoc->nyGlobal / icComm->numCommRows();
   assert(layerLoc->nxGlobal == layerLoc->nx * icComm->numCommColumns());
   assert(layerLoc->nyGlobal == layerLoc->ny * icComm->numCommRows());

   layerLoc->kx0 = layerLoc->nx * icComm->commColumn();
   layerLoc->ky0 = layerLoc->ny * icComm->commRow();

   layerLoc->nf = mNumFeatures;

   layerLoc->nbatchGlobal = message->mNBatchGlobal;

   int const nBatch = message->mNBatchGlobal / icComm->numCommBatches(); // integer arithmetic
   pvAssert(nBatch * icComm->numCommBatches() == message->mNBatchGlobal); // checked in HyPerCol
   layerLoc->nbatch = nBatch;
   layerLoc->kb0    = icComm->commBatch() * nBatch;
   // halo is initialized to zero in constructor, and can be changed by calls
   // to requireMarginWidth. We don't change the values here.
}

int LayerGeometry::requireMarginWidth(int marginWidthNeeded, char axis) {
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
   PVLayerLoc const *loc = getLayerLoc();
   updateNumExtended();
   return *startMargin;
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
