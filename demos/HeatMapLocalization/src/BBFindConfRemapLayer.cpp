/*
 * BBFindConfRemapLayer.cpp
 *
 *  Created on: May 17, 2016
 *      Author: pschultz
 */

#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVAlloc.hpp"
#include "BBFindConfRemapLayer.hpp"

BBFindConfRemapLayer::BBFindConfRemapLayer(char const * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

BBFindConfRemapLayer::BBFindConfRemapLayer() {
   initialize_base();
}

int BBFindConfRemapLayer::initialize_base() {
   // Set boundingboxFinder param defaults.
   // Instantiate the first boundingboxFinder and use its defaults to set the param defaults
   BBFind bb0;
   framesPerMap = bb0.getFramesPerMap();
   threshold = bb0.getThreshold();
   contrast = bb0.getContrast();
   contrastStrength = bb0.getContrastStrength();
   prevInfluence = bb0.getPrevInfluence();
   accumulateAmount = bb0.getAccumulateAmount();
   prevLeakTau = bb0.getPrevLeakTau();
   boundingboxGuessSize = bb0.getBBGuessSize();
   slidingAverageSize = bb0.getSlidingAverageSize();
   maxRectangleMemory = bb0.getMaxRectangleMemory();
   detectionWait = bb0.getDetectionWait();
   internalMapWidth = bb0.getInternalMapWidth();
   internalMapHeight = bb0.getInternalMapHeight();
   imageWidth = bb0.getImageWidth();
   imageHeight = bb0.getImageHeight();
   boundingboxFinder.push_back(bb0);
   return PV_SUCCESS;
}

int BBFindConfRemapLayer::initialize(char const * name, PV::HyPerCol * hc) {
   return HyPerLayer::initialize(name, hc);
}

int BBFindConfRemapLayer::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_imageLayer(ioFlag);
   ioParam_displayedCategories(ioFlag);
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
   ioParam_internalMapWidth(ioFlag);
   ioParam_internalMapHeight(ioFlag);
   return status;
}

void BBFindConfRemapLayer::ioParam_displayedCategories(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamArray(ioFlag, this->getName(), "displayedCategories", &displayedCategories, &numDisplayedCategories);
}

void BBFindConfRemapLayer::ioParam_imageLayer(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(ioFlag, name, "imageLayer", &imageLayerName, "", true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_framesPerMap(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "framesPerMap", &framesPerMap, framesPerMap, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_threshold(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "threshold", &threshold, threshold, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_contrast(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "contrast", &contrast, contrast, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_contrastStrength(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "contrastStrength", &contrastStrength, contrastStrength, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_prevInfluence(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "prevInfluence", &prevInfluence, prevInfluence, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_accumulateAmount(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "accumulateAmount", &accumulateAmount, accumulateAmount, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_prevLeakTau(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "prevLeakTau", &prevLeakTau, prevLeakTau, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_minBlobSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "minBlobSize", &minBlobSize, minBlobSize, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_boundingboxGuessSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "boundingboxGuessSize", &boundingboxGuessSize, boundingboxGuessSize, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_slidingAverageSize(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "slidingAverageSize", &slidingAverageSize, slidingAverageSize, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_maxRectangleMemory(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "maxRectangleMemory", &maxRectangleMemory, maxRectangleMemory, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_detectionWait(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "detectionWait", &detectionWait, detectionWait, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_internalMapWidth(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "internalMapWidth", &internalMapWidth, internalMapWidth, true/*warnIfAbsent*/);
}

void BBFindConfRemapLayer::ioParam_internalMapHeight(enum PV::ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "internalMapHeight", &internalMapHeight, internalMapHeight, true/*warnIfAbsent*/);
}

int BBFindConfRemapLayer::communicateInitInfo() {
   if (imageLayerName && imageLayerName[0]) {
      imageLayer = parent->getLayerFromName(imageLayerName);
      if (imageLayer==nullptr) {
         if (parent->columnId()==0) {
            Fatal() << getDescription_c() << ": imageLayer \"" << imageLayerName << "\" does not refer to a layer in the column." << std::endl;
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         pvExitFailure("");
      }
      imageWidth = imageLayer->getLayerLoc()->nxGlobal;
      imageHeight = imageLayer->getLayerLoc()->nyGlobal;
   }
   else {
      imageWidth = boundingboxFinder.front().getInternalMapWidth();
      imageHeight = boundingboxFinder.front().getInternalMapHeight();
   }

   if (numDisplayedCategories==0) {
      assert(displayedCategories==nullptr);
      numDisplayedCategories = getLayerLoc()->nf;
      displayedCategories = (int *) pvMallocError(sizeof(*displayedCategories)*(size_t) numDisplayedCategories, "%s: unable to allocate %d values for displayedCategories array",
            getDescription_c(), numDisplayedCategories);
      for (int k=0; k<numDisplayedCategories; k++) { displayedCategories[k] = k+1; }
   }
   else {
      for (int k=0; k<numDisplayedCategories; k++) {
         int cat = displayedCategories[k];
         if (cat <=0 || cat > getLayerLoc()->nf) {
            if (parent->columnId()==0) {
               Fatal() << getDescription_c() << ": displayedCategories element " << k+1 << " is " << cat << ", outside the range [1," << getLayerLoc()->nf << "]." << std::endl;
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            pvExitFailure("");
         }
      }
   }

   BBFind& bbf0 = boundingboxFinder.front();
   setBoundingBoxFinderParams(bbf0);
   return PV_SUCCESS;
}

void BBFindConfRemapLayer::setBoundingBoxFinderParams(BBFind& bbf) {
   bbf.setImageSize(imageWidth, imageHeight);
   bbf.setFramesPerMap(framesPerMap);
   bbf.setThreshold(threshold);
   bbf.setContrast(contrast);
   bbf.setContrastStrength(contrastStrength);
   bbf.setPrevInfluence(prevInfluence);
   bbf.setAccumulateAmount(accumulateAmount);
   bbf.setPrevLeakTau(prevLeakTau);
   bbf.setMinBlobSize(minBlobSize);
   bbf.setBBGuessSize(boundingboxGuessSize);
   bbf.setSlidingAverageSize(slidingAverageSize);
   bbf.setMaxRectangleMemory(maxRectangleMemory);
   bbf.setDetectionWait(detectionWait);
   bbf.setInternalMapSize(internalMapWidth, internalMapHeight);
}

int BBFindConfRemapLayer::allocateDataStructures() {
   int const nbatch = parent->getNBatch();
   boundingboxFinder.resize(nbatch);
   for (int b=0; b<nbatch; b++) {
      BBFind& bbf1 = boundingboxFinder[b];
      setBoundingBoxFinderParams(bbf1);
   }
   return PV_SUCCESS;
}

double BBFindConfRemapLayer::getDeltaUpdateTime() {
   if (imageLayer) {
      return imageLayer->getDeltaUpdateTime();
   }
   else {
      return parent->getDeltaTime();
   }
}

int BBFindConfRemapLayer::updateState(double t, double dt) {
   // Compute V from GSyn
   if (getNumChannels() == 1){
      applyGSyn_HyPerLayer1Channel(getLayerLoc()->nbatch, getNumNeurons(), getV(), GSyn[0]);
   }
   else{
      applyGSyn_HyPerLayer(getLayerLoc()->nbatch, getNumNeurons(), getV(), GSyn[0]);
   }

   // Gather the V buffers into root process for BBFind, then scatter the activity.
   float confidenceLocal[getNumNeurons()];
   PV::Communicator * icComm = parent->getCommunicator();
   PVLayerLoc const * loc = getLayerLoc();
   PVHalo const * halo = &loc->halo;
   int const nx = loc->nx;
   int const ny = loc->ny;
   int const nf = loc->nf;
   int const rootProcess = 0;
   for (int b=0; b<parent->getNBatch(); b++) {
      memset(confidenceLocal, 0, sizeof(*confidenceLocal)*getNumNeurons());
      if (parent->columnId()==rootProcess) {
         int const nxGlobal = loc->nxGlobal;
         int const nyGlobal = loc->nyGlobal;
         float confidenceGlobal[getNumGlobalNeurons()];
         PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
         for (int rank=0; rank<icComm->commSize(); rank++) {
            int const kx0 = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns())*nx;
            int const ky0 = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns())*ny;
            float const * localPart = nullptr;
            if (rank==rootProcess) {
               localPart = getV()+b*getNumNeurons();
            }
            else {
               localPart = confidenceLocal;
               MPI_Recv(confidenceLocal, getNumNeurons(), MPI_FLOAT, rank, 137, icComm->communicator(), MPI_STATUS_IGNORE);
            }
            for (int y=0; y<ny; y++) {
               int localStart = kIndex(halo->lt, y+halo->up, 0, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf);
               int globalStart = kIndex(kx0, y+ky0, 0, nxGlobal, nyGlobal, nf);
               memcpy(&confidenceGlobal[globalStart], &localPart[localStart], sizeof(float)*nx*nf);
            }

            boundingboxFinder[b].giveMap(BBFind::bufferToMap3(confidenceGlobal, nxGlobal, nyGlobal, nf, displayedCategories, numDisplayedCategories));
            for (int n=0; n<framesPerMap; n++) {
               boundingboxFinder[b].detect();
            }
            BBFind::Map3 const outValues = boundingboxFinder[b].getOrigSizedConfMap();
            #ifdef PV_USE_OPENMP_THREADS
            #pragma omp parallel for
            #endif
            for (int k=0; k<getNumNeurons(); k+=nf) {
               int x = kxPos(k, nx, ny, nf);
               int y = kyPos(k, nx, ny, nf);
               for (int index=0; index<numDisplayedCategories; index++) {
                  int f = displayedCategories[index]-1;
                  confidenceLocal[k+f] = outValues[index][y+ky0][x+kx0];
               }
            }
            if (rank!=rootProcess) {
               MPI_Send(confidenceLocal, getNumNeurons(), MPI_FLOAT, rank, 138, icComm->communicator());
            }
         }
      }
      else {
         float * buf = getV()+b*getNumNeurons(); // should be "float const *" but older versions of MPI declare MPI_Send to use void* instead of void const*.
         MPI_Send(buf, getNumNeurons(), MPI_FLOAT, 0, 137, icComm->communicator());
         MPI_Recv(confidenceLocal, getNumNeurons(), MPI_FLOAT, 0, 138, icComm->communicator(), MPI_STATUS_IGNORE);
      }
      float * A = clayer->activity->data+b*getNumExtended();
      for (int y=0; y<ny; y++) {
         int kRes = kIndex(0, y, 0, nx, ny, nf);
         int kExt = kIndex(halo->lt, y+halo->up, 0, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf);
         memcpy(&A[kExt], &confidenceLocal[kRes], nx*nf);
      }
   }

   return PV_SUCCESS;
}

BBFindConfRemapLayer::~BBFindConfRemapLayer() {
   free(displayedCategories);
}

PV::BaseObject * createBBFindConfRemapLayer(char const * name, PV::HyPerCol * hc) {
   return hc ? new BBFindConfRemapLayer(name, hc) : nullptr;
}

