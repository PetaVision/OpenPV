/*
 * LocalizationBBFindProbe.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include "LocalizationBBFindProbe.hpp"
#include "BBFind.hpp"
#include <limits>
#include <sstream>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

LocalizationBBFindProbe::LocalizationBBFindProbe(const char *probeName, PV::HyPerCol *hc) {
   initialize_base();
   initialize(probeName, hc);
}

LocalizationBBFindProbe::LocalizationBBFindProbe() { initialize_base(); }

int LocalizationBBFindProbe::initialize_base() {
   // Set BBFind parameters to their defaults
   framesPerMap         = bbfinder.getFramesPerMap();
   threshold            = bbfinder.getThreshold();
   contrast             = bbfinder.getContrast();
   contrastStrength     = bbfinder.getContrastStrength();
   prevInfluence        = bbfinder.getPrevInfluence();
   accumulateAmount     = bbfinder.getAccumulateAmount();
   prevLeakTau          = bbfinder.getPrevLeakTau();
   minBlobSize          = bbfinder.getMinBlobSize();
   boundingboxGuessSize = bbfinder.getBBGuessSize();
   slidingAverageSize   = bbfinder.getSlidingAverageSize();
   maxRectangleMemory   = bbfinder.getMaxRectangleMemory();
   detectionWait        = bbfinder.getDetectionWait();
   internalMapWidth     = bbfinder.getInternalMapWidth();
   internalMapHeight    = bbfinder.getInternalMapHeight();
   return PV_SUCCESS;
}

int LocalizationBBFindProbe::initialize(const char *probeName, PV::HyPerCol *hc) {
   int status = LocalizationProbe::initialize(probeName, hc);
   return status;
}

int LocalizationBBFindProbe::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = LocalizationProbe::ioParamsFillGroup(ioFlag);
   ioParam_framesPerMap(ioFlag);
   ioParam_threshold(ioFlag);
   ioParam_contrast(ioFlag);
   ioParam_contrastStrength(ioFlag);
   ioParam_prevInfluence(ioFlag);
   ioParam_accumulateAmount(ioFlag);
   ioParam_prevLeakTau(ioFlag);
   ioParam_minBlobSize(ioFlag);
   ioParam_boundingboxGuessSize(ioFlag);
   ioParam_slidingAverageSize(ioFlag);
   ioParam_maxRectangleMemory(ioFlag);
   ioParam_detectionWait(ioFlag);
   ioParam_drawMontage(ioFlag);
   ioParam_heatMapMaximum(ioFlag);
   ioParam_heatMapMontageDir(ioFlag);
   ioParam_imageBlendCoeff(ioFlag);
   ioParam_boundingBoxLineWidth(ioFlag);
   ioParam_displayCommand(ioFlag);
   return status;
}

void LocalizationBBFindProbe::ioParam_framesPerMap(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "framesPerMap",
         &framesPerMap,
         framesPerMap,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_threshold(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "threshold", &threshold, threshold, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_contrast(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "contrast", &contrast, contrast, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_contrastStrength(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "contrastStrength",
         &contrastStrength,
         contrastStrength,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_prevInfluence(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "prevInfluence", &prevInfluence, prevInfluence, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_accumulateAmount(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "accumulateAmount",
         &accumulateAmount,
         accumulateAmount,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_prevLeakTau(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "prevLeakTau", &prevLeakTau, prevLeakTau, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_minBlobSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "minBlobSize", &minBlobSize, minBlobSize, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_boundingboxGuessSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "boundingboxGuessSize",
         &boundingboxGuessSize,
         boundingboxGuessSize,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_slidingAverageSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "slidingAverageSize",
         &slidingAverageSize,
         slidingAverageSize,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_maxRectangleMemory(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "maxRectangleMemory",
         &maxRectangleMemory,
         maxRectangleMemory,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_detectionWait(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "detectionWait", &detectionWait, detectionWait, true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_internalMapWidth(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "internalMapWidth",
         &internalMapWidth,
         internalMapWidth,
         true /*warnIfAbsent*/);
}

void LocalizationBBFindProbe::ioParam_internalMapHeight(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "internalMapHeight",
         &internalMapHeight,
         internalMapHeight,
         true /*warnIfAbsent*/);
}

int LocalizationBBFindProbe::communicateInitInfo(CommunicateInitInfoMessage const *message) {
   int status = LocalizationProbe::communicateInitInfo(message);
   bbfinder.setImageSize(imageLayer->getLayerLoc()->nxGlobal, imageLayer->getLayerLoc()->nyGlobal);
   bbfinder.setFramesPerMap(framesPerMap);
   bbfinder.setThreshold(threshold);
   bbfinder.setContrast(contrast);
   bbfinder.setContrastStrength(contrastStrength);
   bbfinder.setPrevInfluence(prevInfluence);
   bbfinder.setAccumulateAmount(accumulateAmount);
   bbfinder.setPrevLeakTau(prevLeakTau);
   bbfinder.setMinBlobSize(minBlobSize);
   bbfinder.setBBGuessSize(boundingboxGuessSize);
   bbfinder.setSlidingAverageSize(slidingAverageSize);
   bbfinder.setMaxRectangleMemory(maxRectangleMemory);
   bbfinder.setDetectionWait(detectionWait);
   bbfinder.setInternalMapSize(internalMapWidth, internalMapHeight);
   return status;
}

int LocalizationBBFindProbe::calcValues(double timevalue) {
   detections.clear();
   unsigned detectionIndex = 0U;

   // Gather the confidence layers into root process for BBFind
   float confidenceLocal[targetLayer->getNumExtended()];
   PV::Communicator *icComm = parent->getCommunicator();
   int const rootProcess    = 0;
   if (parent->columnId() == rootProcess) {
      PVLayerLoc const *loc = targetLayer->getLayerLoc();
      PVHalo const *halo    = &loc->halo;
      int const nx          = loc->nx;
      int const ny          = loc->ny;
      int const nf          = loc->nf;
      int const nxGlobal    = loc->nxGlobal;
      int const nyGlobal    = loc->nyGlobal;
      float confidenceGlobal[targetLayer->getNumGlobalNeurons()];
      PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
      for (int rank = 0; rank < icComm->commSize(); rank++) {
         int const kx0 = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns()) * nx;
         int const ky0 = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns()) * ny;
         float const *localPart = NULL;
         if (rank == 0) {
            localPart = targetLayer->getLayerData();
         }
         else {
            localPart = confidenceLocal;
            MPI_Recv(
                  confidenceLocal,
                  targetLayer->getNumExtended(),
                  MPI_FLOAT,
                  rank,
                  137,
                  icComm->communicator(),
                  MPI_STATUS_IGNORE);
         }
         for (int y = 0; y < targetLayer->getLayerLoc()->ny; y++) {
            int localStart = kIndex(
                  halo->lt,
                  y + halo->up,
                  0,
                  nx + halo->lt + halo->rt,
                  ny + halo->dn + halo->up,
                  nf);
            int globalStart = kIndex(kx0, y + ky0, 0, nxGlobal, nyGlobal, nf);
            memcpy(&confidenceGlobal[globalStart], &localPart[localStart], sizeof(float) * nx * nf);
         }

         bbfinder.giveMap(
               BBFind::bufferToMap3(
                     confidenceGlobal,
                     nxGlobal,
                     nyGlobal,
                     nf,
                     displayedCategories,
                     numDisplayedCategories));
         for (int n = 0; n < framesPerMap; n++) {
            bbfinder.detect();
         }
         BBFind::Rectangles R = bbfinder.getDetections();
         unsigned rsize       = R.size();
         for (unsigned d = 0; d < rsize; d++) {
            list<BBFind::Rectangle> const *rlist = &R.at(d);
            if (rlist->empty()) {
               continue;
            }
            for (list<BBFind::Rectangle>::const_iterator listiter = rlist->begin();
                 listiter != rlist->end();
                 listiter++) {
               BBFind::Rectangle const &r = *listiter;
               LocalizationData bbox;
               bbox.left   = std::min(std::max(r.left(), 0), imageLoc->nxGlobal);
               bbox.right  = std::min(std::max(r.right(), 0), imageLoc->nxGlobal);
               bbox.top    = std::min(std::max(r.top(), 0), imageLoc->nyGlobal);
               bbox.bottom = std::min(std::max(r.bottom(), 0), imageLoc->nyGlobal);
               if (bbox.right - bbox.left >= minBoundingBoxWidth
                   && bbox.bottom - bbox.top >= minBoundingBoxHeight) {
                  bbox.feature        = displayedCategories[d] - 1;
                  bbox.displayedIndex = d;
                  bbox.score = computeBoxConfidence(bbox, confidenceGlobal, nxGlobal, nyGlobal, nf);
                  if (bbox.score == bbox.score) {
                     detections.push_back(bbox);
                     detectionIndex++;
                  }
               }
            }
         }
      }
   }
   else {
      /*** The const_cast is used because older versions of MPI_Send use void* instead of void
       * const* in the first argument.  Do not use const_cast. ***/
      float *buf = const_cast<float *>(targetLayer->getLayerData());
      MPI_Send(buf, targetLayer->getNumExtended(), MPI_FLOAT, 0, 137, icComm->communicator());
   }
   MPI_Bcast(&detectionIndex, 1, MPI_UNSIGNED, rootProcess, icComm->communicator());

   assert(getNumValues() == 1);
   double *values = getValuesBuffer();
   *values        = detectionIndex;
   return PV_SUCCESS;
}

double LocalizationBBFindProbe::computeBoxConfidence(
      LocalizationData const &bbox,
      float const *buffer,
      int nx,
      int ny,
      int nf) {
   double score = 0.0f;
   int count    = 0;
   for (int y = bbox.top; y < bbox.bottom; y++) {
      for (int x = bbox.left; x < bbox.right; x++) {
         int xt = (int)floor(x / imageDilationX);
         int yt = (int)floor(y / imageDilationY);
         assert(xt >= 0 && xt < nx && yt >= 0 && yt < ny);
         int f     = bbox.feature;
         int i     = bbox.displayedIndex;
         int index = kIndex(xt, yt, f, nx, ny, nf);
         float a   = buffer[index];
         if (a >= detectionThreshold[i]) {
            score += (double)buffer[index];
            count++;
         }
      }
   }
   score /= (double)count;
   return score;
}

LocalizationBBFindProbe::~LocalizationBBFindProbe() {}

PV::BaseObject *createLocalizationBBFindProbe(char const *name, PV::HyPerCol *hc) {
   return hc ? new LocalizationBBFindProbe(name, hc) : NULL;
}
