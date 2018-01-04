/*
 * ImpliedWeightsPair.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "ImpliedWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ImpliedWeights.hpp"

namespace PV {

ImpliedWeightsPair::ImpliedWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

ImpliedWeightsPair::~ImpliedWeightsPair() {}

int ImpliedWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

int ImpliedWeightsPair::setDescription() {
   description.clear();
   description.append("ImpliedWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

void ImpliedWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag = PARAMS_IO_READ) {
      mSharedWeights = false;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
}

void ImpliedWeightsPair::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag = PARAMS_IO_READ) {
      mWriteStep = -1;
      parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
   }
}

void ImpliedWeightsPair::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag = PARAMS_IO_READ) {
      mInitialWriteTime = 0.0;
      parent->parameters()->handleUnnecessaryParameter(name, "initialWriteTime");
   }
}

void ImpliedWeightsPair::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag = PARAMS_IO_READ) {
      mWriteCompressedWeights = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedWeights");
   }
}

void ImpliedWeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag = PARAMS_IO_READ) {
      mWriteCompressedCheckpoints = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
}

void ImpliedWeightsPair::needPre() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPre can be called.\n",
         getDescription_c());
   if (mPreWeights == nullptr) {
      mPreWeights = new ImpliedWeights(
            std::string(name),
            mPatchSizeX,
            mPatchSizeY,
            mPatchSizeF,
            mConnectionData->getPre()->getLayerLoc(),
            mConnectionData->getPost()->getLayerLoc(),
            mConnectionData->getNumAxonalArbors(),
            mSharedWeights,
            -std::numeric_limits<double>::infinity() /*timestamp*/);
   }
}

void ImpliedWeightsPair::needPost() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPost can be called.\n",
         getDescription_c());
   if (mPostWeights == nullptr) {
      PVLayerLoc const *preLoc  = mConnectionData->getPre()->getLayerLoc();
      PVLayerLoc const *postLoc = mConnectionData->getPost()->getLayerLoc();
      mPostWeights              = new ImpliedWeights(
            std::string(name),
            calcPostPatchSize(mPatchSizeX, preLoc->nx, postLoc->nx),
            calcPostPatchSize(mPatchSizeY, preLoc->ny, postLoc->ny),
            preLoc->nf /* number of features in post patch */,
            postLoc,
            preLoc,
            mConnectionData->getNumAxonalArbors(),
            mSharedWeights,
            -std::numeric_limits<double>::infinity() /*timestamp*/);
   }
}

int ImpliedWeightsPair::registerData(Checkpointer *checkpointer) {
   // Bypass WeightsPair
   return PV_SUCCESS;
}

void ImpliedWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

int ImpliedWeightsPair::readStateFromCheckpoint(Checkpointer *checkpointer) {
   // Bypass WeightsPair
   return PV_SUCCESS;
}

void ImpliedWeightsPair::outputState(double timestamp) {}

} // namespace PV
