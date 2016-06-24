/*
 * BBFindConfRemapProbe.cpp
 *
 *  Created on: May 18, 2016
 *      Author: pschultz
 */

#include <limits>
#include <stdexcept>
#include <gdal.h>
#include <gdal_priv.h>
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVAlloc.hpp"
#include "BBFindConfRemapProbe.hpp"

BBFindConfRemapProbe::BBFindConfRemapProbe(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

BBFindConfRemapProbe::BBFindConfRemapProbe() {
   initialize_base();
}

int BBFindConfRemapProbe::initialize_base() {
   return PV_SUCCESS;
}

int BBFindConfRemapProbe::initialize(char const * name, PV::HyPerCol * hc) {
   return LayerProbe::initialize(name, hc);
}

int BBFindConfRemapProbe::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_imageLayer(ioFlag);
   ioParam_reconLayer(ioFlag);
   ioParam_classNamesFile(ioFlag);
   ioParam_minBoundingBoxWidth(ioFlag);
   ioParam_minBoundingBoxHeight(ioFlag);
   ioParam_drawMontage(ioFlag);
   ioParam_heatMapMontageDir(ioFlag);
   ioParam_heatMapThreshold(ioFlag);
   ioParam_heatMapMaximum(ioFlag);
   ioParam_imageBlendCoeff(ioFlag);
   ioParam_boundingBoxLineWidth(ioFlag);
   ioParam_displayCommand(ioFlag);
   return status;
}

void BBFindConfRemapProbe::ioParam_imageLayer(enum PV::ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "imageLayer", &imageLayerName);
}

void BBFindConfRemapProbe::ioParam_reconLayer(enum PV::ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "reconLayer", &reconLayerName);
}

void BBFindConfRemapProbe::ioParam_classNamesFile(enum PV::ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "classNamesFile", &classNamesFile, "");
}

void BBFindConfRemapProbe::ioParam_minBoundingBoxWidth(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->ioParamValue(ioFlag, this->getName(), "minBoundingBoxWidth", &minBoundingBoxWidth, minBoundingBoxWidth, true/*warnIfAbsent*/);
}

void BBFindConfRemapProbe::ioParam_minBoundingBoxHeight(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->ioParamValue(ioFlag, this->getName(), "minBoundingBoxHeight", &minBoundingBoxHeight, minBoundingBoxHeight, true/*warnIfAbsent*/);
}

void BBFindConfRemapProbe::ioParam_drawMontage(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->ioParamValue(ioFlag, this->getName(), "drawMontage", &drawMontage, drawMontage, true/*warnIfAbsent*/);
#ifdef PV_USE_GDAL
   GDALAllRegister();
#else // PV_USE_GDAL
   if (ioFlag==PARAMS_IO_READ) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: PetaVision must be compiled with GDAL to use BBFindConfRemapProbe with drawMontage set.\n",
               getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      return PV_FAILURE;
   }
#endif // PV_USE_GDAL
}

void BBFindConfRemapProbe::ioParam_heatMapMontageDir(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamStringRequired(ioFlag, this->getName(), "heatMapMontageDir", &heatMapMontageDir);
   }
}

void BBFindConfRemapProbe::ioParam_heatMapThreshold(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      parent->ioParamArray(ioFlag, name, "heatMapThreshold", &heatMapThreshold, &numHeatMapThresholds);
   }
}

void BBFindConfRemapProbe::ioParam_heatMapMaximum(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      parent->ioParamArray(ioFlag, name, "heatMapMaximum", &heatMapMaximum, &numHeatMapMaxima);
   }
}

void BBFindConfRemapProbe::ioParam_imageBlendCoeff(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "imageBlendCoeff", &imageBlendCoeff, imageBlendCoeff/*default value*/, true/*warnIfAbsent*/);
   }
}

void BBFindConfRemapProbe::ioParam_boundingBoxLineWidth(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "boundingBoxLineWidth", &boundingBoxLineWidth, boundingBoxLineWidth/*default value*/, true/*warnIfAbsent*/);
   }
}

void BBFindConfRemapProbe::ioParam_displayCommand(enum PV::ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamString(ioFlag, this->getName(), "displayCommand", &displayCommand, NULL, true/*warnIfAbsent*/);
   }
}

void BBFindConfRemapProbe::setOutputFilenameBase(char const * fn) {
   free(outputFilenameBase);
   int status = PV_SUCCESS;
   std::string fnString(fn);
   size_t lastSlash = fnString.rfind("/");
   if (lastSlash != std::string::npos) {
      fnString.erase(0, lastSlash+1);
   }

   size_t lastDot = fnString.rfind(".");
   if (lastDot != std::string::npos) {
      fnString.erase(lastDot);
   }
   if (fnString.empty()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "LocalizationProbe::setOutputFilenameBase error: string \"%s\" is empty after removing directory and extension.\n", fn);
      }
      status = PV_FAILURE;
      outputFilenameBase = NULL;
   }
   else {
      outputFilenameBase = strdup(fnString.c_str());
      if (outputFilenameBase==NULL) {
         fprintf(stderr, "LocalizationProbe::setOutputFilenameBase failed with filename \"%s\": %s\n", fn, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
}

int BBFindConfRemapProbe::communicateInitInfo() {
   int status = PV::LayerProbe::communicateInitInfo();

   assert(targetLayer);
   targetBBFindConfRemapLayer = dynamic_cast<BBFindConfRemapLayer*>(targetLayer);
   if (targetBBFindConfRemapLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: targetLayer \"%s\" must be a BBFindLayer.\n",
               getKeyword(), name, this->targetName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   setLayerFromParam(&imageLayer, "imageLayer", imageLayerName);
   setLayerFromParam(&reconLayer, "reconLayer", reconLayerName);
   if (status != PV_SUCCESS) {
      // error messages get printed by setLayerFromParam
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   // Get the names labeling each feature from the class names file.  Only the root process stores these values.
   if (parent->columnId()==0) {
      int const nf = targetLayer->getLayerLoc()->nf;
      classNames = (char **) malloc(nf * sizeof(char *));
      if (classNames == NULL) {
         fprintf(stderr, "%s \"%s\" unable to allocate classNames: %s\n", getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      if (strcmp(classNamesFile,"")) {
         std::ifstream * classNamesStream = new std::ifstream(classNamesFile);
         if (classNamesStream->fail()) {
            fprintf(stderr, "%s \"%s\": unable to open classNamesFile \"%s\".\n", getKeyword(), name, classNamesFile);
            exit(EXIT_FAILURE);
         }
         for (int k=0; k<nf; k++) {
            // Need to clean this section up: handle too-long lines, premature eof, other issues
            char oneclass[1024];
            classNamesStream->getline(oneclass, 1024);
            classNames[k] = strdup(oneclass);
            if (classNames[k] == NULL) {
               fprintf(stderr, "%s \"%s\" unable to allocate class name %d from \"%s\": %s\n",
                     getKeyword(), name, k, classNamesFile, strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
      }
      else {
         printf("classNamesFile was not set in params file; Class names will be feature numbers (one-indexed).\n");
         for (int k=0; k<nf; k++) {
            std::stringstream classNameString("");
            classNameString << "Feature " << k+1;
            classNames[k] = strdup(classNameString.str().c_str());
            if (classNames[k]==NULL) {
               fprintf(stderr, "%s \"%s\": unable to allocate className %d: %s\n", getKeyword(), name, k, strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
      }
   }
   imageDilationX = pow(2.0, targetLayer->getXScale() - imageLayer->getXScale());
   imageDilationY = pow(2.0, targetLayer->getYScale() - imageLayer->getYScale());

   return status;
}

void BBFindConfRemapProbe::setLayerFromParam(PV::HyPerLayer ** layer, char const * layerType, char const * layerName) {
   PV::HyPerLayer * l = parent->getLayerFromName(layerName);
   if (l==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: %s \"%s\" does not refer to a layer in the column.\n",
               getKeyword(), name, layerType, layerName);
      }
      throw std::invalid_argument("setLayerFromParam:bad layer name");
   }

   int const nf = l->getLayerLoc()->nf;
   if (drawMontage && nf != 3) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: If the drawMontage flag is set, the %s must have exactly three features (\"%s\" has %d).\n",
               getKeyword(), name, layerType, layerName, nf);
      }
      throw std::invalid_argument("setLayerFromParam:not three features");
   }
   *layer = l;
}

void BBFindConfRemapProbe::setOptimalMontage() {
   // Find the best number of rows and columns to use for the montage
//   int const numCategories = displayCategoryIndexEnd - displayCategoryIndexStart + 1;
   int const numDisplayedCategories = targetBBFindConfRemapLayer->getNumDisplayedCategories();
   assert(numDisplayedCategories>0);
   int numRows[numDisplayedCategories];
   float totalSizeX[numDisplayedCategories];
   float totalSizeY[numDisplayedCategories];
   float aspectRatio[numDisplayedCategories];
   float ldgr[numDisplayedCategories]; // log of difference from golden ratio
   float loggoldenratio = logf(0.5f * (1.0f + sqrtf(5.0f)));
   for (int numCol=1; numCol <= numDisplayedCategories ; numCol++) {
      int idx = numCol-1;
      numRows[idx] = (int) ceil((float) numDisplayedCategories/(float) numCol);
      totalSizeX[idx] = numCol * (imageLayer->getLayerLoc()->nxGlobal + 10); // +10 for spacing between images.
      totalSizeY[idx] = numRows[idx] * (imageLayer->getLayerLoc()->nyGlobal + 64 + 10); // +64 for category label
      aspectRatio[idx] = (float) totalSizeX[idx]/(float) totalSizeY[idx];
      ldgr[idx] = fabsf(log(aspectRatio[idx]) - loggoldenratio);
   }
   numMontageColumns = -1;
   float minldfgr = std::numeric_limits<float>::infinity();
   for (int numCol=1; numCol <= numDisplayedCategories ; numCol++) {
      int idx = numCol-1;
      if (ldgr[idx] < minldfgr) {
         minldfgr = ldgr[idx];
         numMontageColumns = numCol;
      }
   }
   assert(numMontageColumns > 0);
   numMontageRows = numRows[numMontageColumns-1];
   while ((numMontageColumns-1)*numMontageRows >= numDisplayedCategories) { numMontageColumns--; }
   while ((numMontageRows-1)*numMontageColumns >= numDisplayedCategories) { numMontageRows--; }
   if (numMontageRows < 2) { numMontageRows = 2; }
}

int BBFindConfRemapProbe::allocateDataStructures() {
   detectionS.resize(getNumValues());

   setOptimalMontage();
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   int const nxGlobal = imageLoc->nxGlobal;
   int const nyGlobal = imageLoc->nyGlobal;
   int const imageLocalSize = imageLoc->nx * imageLoc->ny * 3;
   montageDimX = (nxGlobal + 10) * (numMontageColumns + 2);
   montageDimY = (nyGlobal + 64 + 10) * numMontageRows + 32;
   int const imageMontageSize = montageDimX * montageDimY * 3;
   if (parent->columnId()==0 && drawMontage) {
      montageImage = (unsigned char *) pvCallocError(imageMontageSize, sizeof(*montageImage),
            "%s \"%s\" allocation for heat map montage image\n", getKeyword(), name);
      montageImageComm = (unsigned char *) pvCallocError(imageLocalSize, sizeof(*montageImageComm),
            "%s \"%s\" allocation for MPI communication of heat map montage image\n", getKeyword(), name);
   }
   montageImageLocal = (unsigned char *) pvCallocError(imageLocalSize, sizeof(*montageImageLocal),
         "%s \"%s\" allocation for heat map image in rank %d\n", getKeyword(), name, parent->columnId());
   grayScaleImage = (pvadata_t *) pvCallocError(imageLoc->nx*imageLoc->ny, sizeof(pvadata_t),
         "%s \"%s\" allocation for background image in rank %d\n", getKeyword(), name, parent->columnId());
   return PV_SUCCESS;
}

int BBFindConfRemapProbe::calcValues(double timevalue) {
   std::vector<BBFind> const& boundingboxFinder = targetBBFindConfRemapLayer->getBoundingBoxFinder();
   assert(boundingboxFinder.size()==getNumValues());
   assert(detectionS.size()==getNumValues());
   for (int b=0; b<getNumValues(); b++) {
      detectionS[b].clear();
      BBFind boundingboxFinder = targetBBFindConfRemapLayer->getBoundingBoxFinder()[b]; // TODO: should be able to declare as const
      BBFind::Rectangles const R = boundingboxFinder.getDetections();
      unsigned rsize = R.size();
      for (unsigned d=0; d<rsize; d++) {
         list<BBFind::Rectangle> const * rlist = &R.at(d);
         if (rlist->empty()) {continue;}
         for (list<BBFind::Rectangle>::const_iterator listiter=rlist->begin(); listiter!=rlist->end(); listiter++) {
            BBFind::Rectangle const& r = *listiter;
            LocalizationData L;
            L.feature = targetBBFindConfRemapLayer->getDisplayedCategories()[d]-1;
            L.displayedIndex = (int) d;
            L.left = r.left();
            L.right = r.right();
            L.top = r.top();
            L.bottom = r.bottom();
            L.score = 0.986; // TODO: compute a real score.
            detectionS[b].push_back(L);
         }
      }
   }
   return PV_SUCCESS;
}

int BBFindConfRemapProbe::outputStateWrapper(double timef, double dt){
   int status = PV_SUCCESS;
   if((getTextOutputFlag()||drawMontage) && needUpdate(timef, dt)){
      status = outputState(timef);
   }
   return status;
}

int BBFindConfRemapProbe::outputState(double timevalue) {
   int status = getValues(timevalue); // all processes must call getValues in parallel.
   if (getTextOutputFlag() && getOutputStream()) {
      assert(parent->columnId()==0);
      int const nbatch = parent->getNBatch();
      for (int b=0; b<nbatch; b++) {
         if (nbatch>1) {
            getOutputStream()->printf("  Batch element %d\n", b);
         }
         for(std::vector<LocalizationData>::iterator it = detectionS[b].begin(); it < detectionS[b].end(); it++){
            LocalizationData const& L = *it;
            getOutputStream()->printf("Time %f, ", timevalue);
            if (nbatch>1) { getOutputStream()->printf("batch element %d, ", b); }
            getOutputStream()->printf("\"%s\", score %f, bounding box x=[%d,%d), y=[%d,%d)\n",
                  classNames[L.feature], L.score, L.left, L.right, L.top, L.bottom);
         }
         if (drawMontage) {
            status = makeMontage(b);
         }
      }
   }
   return status;
}

int BBFindConfRemapProbe::makeMontage(int b) {
   assert(drawMontage);
   assert(numMontageRows > 0 && numMontageColumns > 0);
   assert((parent->columnId()==0) == (montageImage!=NULL));
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   PVHalo const * halo = &imageLoc->halo;
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const nf = imageLoc->nf;
   int const N = nx * ny;

   // create grayscale version of image layer for background of heat maps.
   makeGrayScaleImage(b);

   // for each displayed category, copy grayScaleImage to the relevant part of the montage, and impose the upsampled version
   // of the target layer onto it.
   drawHeatMaps(b);

   drawOriginalAndReconstructed();

   if (parent->columnId()!=0) { return PV_SUCCESS; }

   // Draw bounding boxes
   if (boundingBoxLineWidth > 0) {
      assert(getNumValues()==1);
      for (size_t d=0; d<detectionS[b].size(); d++) {
         LocalizationData const * thisBoundingBox = &detectionS[b][d];
         int winningFeature = thisBoundingBox->feature;
         if (winningFeature<0) { continue; }
         int left = thisBoundingBox->left;
         int right = thisBoundingBox->right;
         int top = thisBoundingBox->top;
         int bottom = thisBoundingBox->bottom;
         int winningIndex = thisBoundingBox->displayedIndex;
         assert(winningIndex>=0);
         int montageColumn = kxPos(winningIndex, numMontageColumns, numMontageRows, 1);
         int montageRow = kyPos(winningIndex, numMontageColumns, numMontageRows, 1);
         int xStartInMontage = montageColumn * (imageLoc->nxGlobal+10) + 5 + left;
         int yStartInMontage = montageRow * (imageLoc->nyGlobal+64+10) + 5 + 64 + top;
         int width = (int) (right-left);
         int height = (int) (bottom-top);
         char const bbColor[3] = {'\377', '\0', '\0'}; // red
         for (int y=0; y<boundingBoxLineWidth; y++) {
            int lineStart=kIndex(xStartInMontage, yStartInMontage+y, 0, montageDimX, montageDimY, 3);
            for (int k=0; k<3*width; k++) {
               int f = featureIndex(k,imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart+k] = bbColor[f];
            }
         }
         for (int y=boundingBoxLineWidth; y<height-boundingBoxLineWidth; y++) {
            int lineStart=kIndex(xStartInMontage, yStartInMontage+y, 0, montageDimX, montageDimY, 3);
            for (int k=0; k<3*boundingBoxLineWidth; k++) {
               int f = featureIndex(k,imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart+k] = bbColor[f];
            }
            lineStart=kIndex(xStartInMontage+width-boundingBoxLineWidth, yStartInMontage+y, 0, montageDimX, montageDimY, 3);
            for (int k=0; k<3*boundingBoxLineWidth; k++) {
               int f = featureIndex(k,imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart+k] = bbColor[f];
            }
         }
         for (int y=height-boundingBoxLineWidth; y<height; y++) {
            int lineStart=kIndex(xStartInMontage, yStartInMontage+y, 0, montageDimX, montageDimY, 3);
            for (int k=0; k<3*width; k++) {
               int f = featureIndex(k,imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart+k] = bbColor[f];
            }
         }
      }
   }

   // Add progress information to bottom 32 pixels
   drawProgressInformation();

   // write out montageImage to disk
   writeMontage();

   return PV_SUCCESS;
}

void BBFindConfRemapProbe::makeGrayScaleImage(int b) {
   assert(grayScaleImage);
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   PVHalo const * halo = &imageLoc->halo;
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const nf = imageLoc->nf;
   int const N = nx * ny;
   pvadata_t const * imageActivity = imageLayer->getLayerData()+b*imageLayer->getNumExtended();
   for (int kxy = 0; kxy < N; kxy++) {
      pvadata_t a = (pvadata_t) 0;
      int nExt0 = kIndexExtended(kxy * nf, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      pvadata_t const * imageDataXY = &imageActivity[nExt0];
      for (int f=0; f<nf; f++) {
         a += imageDataXY[f];
      }
      grayScaleImage[kxy] = a/(pvadata_t) nf;
   }
   pvadata_t minValue = std::numeric_limits<pvadata_t>::infinity();
   pvadata_t maxValue = -std::numeric_limits<pvadata_t>::infinity();
   for (int kxy = 0; kxy < N; kxy++) {
      pvadata_t a = grayScaleImage[kxy];
      if (a < minValue) { minValue = a; }
      if (a > maxValue) { maxValue = a; }
   }
   MPI_Allreduce(MPI_IN_PLACE, &minValue, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &maxValue, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
   // Scale grayScaleImage to be between 0 and 1.
   // If maxValue==minValue, make grayScaleImage have a constant 0.5.
   if (maxValue==minValue) {
      for (int kxy = 0; kxy < N; kxy++) {
         grayScaleImage[kxy] = 0.5;
      }
   }
   else {
      pvadata_t scaleFactor = (pvadata_t) 1.0/(maxValue-minValue);
      for (int kxy = 0; kxy < N; kxy++) {
         pvadata_t * pixloc = &grayScaleImage[kxy];
         *pixloc = scaleFactor * (*pixloc - minValue);
      }
   }
}

void BBFindConfRemapProbe::drawHeatMaps(int b) {
   pvadata_t thresholdColor[] = {0.5f, 0.5f, 0.5f}; // rgb color of heat map when activity is at or below detectionThreshold
   pvadata_t heatMapColor[] = {0.0f, 1.0f, 0.0f};   // rgb color of heat map when activity is at or above heatMapMaximum

   PVLayerLoc const * targetLoc = targetLayer->getLayerLoc();
   PVHalo const * targetHalo = &targetLoc->halo;
   size_t numDetections = detectionS[b].size();
   int winningFeature[numDetections];
   int winningIndex[numDetections];
   double boxConfidence[numDetections];
   for (int d=0; d<numDetections; d++) {
      LocalizationData const * box = &detectionS[b].at(d);
      boxConfidence[d] = box->score;
      winningFeature[d] = box->feature;
      winningIndex[d] = box->displayedIndex;
   }

   double maxConfByCategory[targetLoc->nf];
   for (int f=0; f<targetLoc->nf; f++) { maxConfByCategory[f] = -std::numeric_limits<pvadata_t>::infinity(); }
   for (size_t d=0; d<numDetections; d++) {
      int f = winningFeature[d];
      double a = boxConfidence[d];
      double m = maxConfByCategory[f];
      maxConfByCategory[f] = a > m ? a : m;
   }

   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const numDisplayedCategories = targetBBFindConfRemapLayer->getNumDisplayedCategories();
   int const * displayedCategories = targetBBFindConfRemapLayer->getDisplayedCategories();
   for (int idx=0; idx<numDisplayedCategories; idx++) {
      int category = displayedCategories[idx];
      int f = category-1; // category is 1-indexed; f is zero-indexed.
      for (int y=0; y<ny; y++) {
         for (int x=0; x<nx; x++) {
            pvadata_t backgroundLevel = grayScaleImage[x + nx * y];
            int xTarget = (int) ((double) x/imageDilationX);
            int yTarget = (int) ((double) y/imageDilationY);
            int targetIdx = kIndex(xTarget, yTarget, f, targetLoc->nx, targetLoc->ny, targetLoc->nf);
            int targetIdxExt = kIndexExtended(targetIdx, targetLoc->nx, targetLoc->ny, targetLoc->nf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
            pvadata_t heatMapLevel = targetLayer->getLayerData()[targetIdxExt];

            // Only show values if they are the highest category
            for(int idx2=0; idx2<numDisplayedCategories; idx2++) {
               int f2 = displayedCategories[idx2]-1;
               heatMapLevel *= (float) (heatMapLevel >= targetLayer->getLayerData()[targetIdxExt-f+f2]);
            }

            float heatMapThresh = heatMapThreshold[idx];
            float heatMapMax = heatMapMaximum[idx];
            heatMapLevel = (heatMapLevel - heatMapThresh)/(heatMapMax-heatMapThresh);
            heatMapLevel = heatMapLevel < (pvadata_t) 0 ? (pvadata_t) 0 : heatMapLevel > (pvadata_t) 1 ? (pvadata_t) 1 : heatMapLevel;
            int montageIdx = kIndex(x, y, 0, nx, ny, 3);
            for(int rgb=0; rgb<3; rgb++) {
               pvadata_t h = heatMapLevel * heatMapColor[rgb] + (1-heatMapLevel) * thresholdColor[rgb];
               pvadata_t g = imageBlendCoeff * backgroundLevel + (1-imageBlendCoeff) * h;
               assert(g>=(pvadata_t) -0.001 && g <= (pvadata_t) 1.001);
               g = nearbyintf(255*g);
               unsigned char gchar = (unsigned char) g;
               assert(montageIdx>=0 && montageIdx+rgb<nx*ny*3);
               montageImageLocal[montageIdx + rgb] = gchar;
            }
         }
      }
      if (parent->columnId()!=0) {
         MPI_Send(montageImageLocal, nx*ny*3, MPI_UNSIGNED_CHAR, 0, 111, parent->icCommunicator()->communicator());
      }
      else {
         int montageCol = idx % numMontageColumns;
         int montageRow = (idx - montageCol) / numMontageColumns; // Integer arithmetic
         int xStartInMontage = (imageLoc->nxGlobal + 10)*montageCol + 5;
         int yStartInMontage = (imageLoc->nyGlobal + 64 + 10)*montageRow + 64 + 5;
         int const numCommRows = parent->icCommunicator()->numCommRows();
         int const numCommCols = parent->icCommunicator()->numCommColumns();
         for (int rank=0; rank<parent->numberOfColumns(); rank++) {
            if (rank==0) {
               memcpy(montageImageComm, montageImageLocal, nx*ny*3);
            }
            else {
               MPI_Recv(montageImageComm, nx*ny*3, MPI_UNSIGNED_CHAR, rank, 111, parent->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
            }
            int const commRow = rowFromRank(rank, numCommRows, numCommCols);
            int const commCol = columnFromRank(rank, numCommRows, numCommCols);
            for (int y=0; y<ny; y++) {
               int destIdx = kIndex(xStartInMontage+commCol*nx, yStartInMontage+commRow*ny+y, 0, montageDimX, montageDimY, 3);
               int srcIdx = kIndex(0, y, 0, nx, ny, 3);
               memcpy(&montageImage[destIdx], &montageImageComm[srcIdx], nx*3);
            }
         }

         // Draw confidences
         char confidenceText[16];
         if (maxConfByCategory[f]>0.0) {
            int slen = snprintf(confidenceText, 16, "%.1f", 100*maxConfByCategory[f]);
            if (slen >= 16) {
               fflush(stdout);
               fprintf(stderr, "Formatted text for confidence %f of category %d is too long.\n", maxConfByCategory[idx], f);
               exit(EXIT_FAILURE);
            }
         }
         else {
            strncpy(confidenceText, "-", 2);
         }
         drawTextOnMontage("white", "gray", confidenceText, xStartInMontage, yStartInMontage-32, imageLayer->getLayerLoc()->nxGlobal, 32);
      }
   }
}

void BBFindConfRemapProbe::drawOriginalAndReconstructed() {
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   // Draw original image
   int xStart = 5+(2*numMontageColumns+1)*(imageLoc->nxGlobal+10)/2;
   int yStart = 5+64;
   insertImageIntoMontage(xStart, yStart, imageLayer->getLayerData(), imageLoc, true/*extended*/);

   // Draw reconstructed image
   // same xStart, yStart is down one row.
   // I should check that reconLayer and imageLayer have the same dimensions
   yStart += imageLoc->nyGlobal + 64 + 10;
   insertImageIntoMontage(xStart, yStart, reconLayer->getLayerData(), reconLayer->getLayerLoc(), true/*extended*/);
}

void BBFindConfRemapProbe::drawProgressInformation() {
   std::stringstream progress("");
   double elapsed = parent->simulationTime() - parent->getStartTime();
   double finishTime = parent->getStopTime() - parent->getStartTime();
   bool isLastTimeStep = elapsed >= finishTime - parent->getDeltaTimeBase()/2;
   if (!isLastTimeStep) {
      int percentage = (int) nearbyintf(100.0 * elapsed / finishTime);
      progress << "t = " << elapsed << ", finish time = " << finishTime << " (" << percentage << "%%)";
   }
   else {
      progress << "t = " << elapsed << ", completed";
   }
   drawTextOnMontage("black", "white", progress.str().c_str(), 0, montageDimY-32, montageDimX, 32);
}

void BBFindConfRemapProbe::drawTextOnMontage(char const * backgroundColor, char const * textColor, char const * labelText, int xOffset, int yOffset, int width, int height) {
   assert(parent->columnId()==0);
   char * tempfile = strdup("/tmp/Localization_XXXXXX.tif");
   if (tempfile == NULL) {
      fprintf(stderr, "%s \"%s\": drawTextOnMontage failed to create temporary file for text\n", getKeyword(), name);
      exit(EXIT_FAILURE);
   }
   int tempfd = mkstemps(tempfile, 4/*suffixlen*/);
   if (tempfd < 0) {
      fprintf(stderr, "%s \"%s\": drawTextOnMontage failed to create temporary file for writing\n", getKeyword(), name);
      exit(EXIT_FAILURE);
   }
   int status = close(tempfd); //mkstemps opens the file to avoid race between finding unused filename and opening it, but we don't need the file descriptor.
   if (status != 0) {
      fprintf(stderr, "%s \"%s\": drawTextOnMontage failed to close temporory file %s: %s\n", getKeyword(), name, tempfile, strerror(errno));
      exit(EXIT_FAILURE);
   }
   drawTextIntoFile(tempfile, backgroundColor, textColor, labelText, width, height);
   insertFileIntoMontage(tempfile, xOffset, yOffset, width, height);
   status = unlink(tempfile);
   if (status != 0) {
      fprintf(stderr, "%s \"%s\": drawTextOnMontage failed to delete temporary file %s: %s\n", getKeyword(), name, tempfile, strerror(errno));
      exit(EXIT_FAILURE);
   }
   free(tempfile);
}

void BBFindConfRemapProbe::drawTextIntoFile(char const * labelFilename, char const * backgroundColor, char const * textColor, char const * labelText, int width, int height) {
   assert(parent->columnId()==0);
   std::stringstream convertCmd("");
   convertCmd << "convert -depth 8 -background \"" << backgroundColor << "\" -fill \"" << textColor << "\" -size " << width << "x" << height << " -pointsize 24 -gravity center label:\"" << labelText << "\" \"" << labelFilename << "\"";
   int status = system(convertCmd.str().c_str());
   if (status != 0) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error creating label file \"%s\": ImageMagick convert returned %d.\n", getKeyword(), name, labelFilename, WEXITSTATUS(status));
      exit(EXIT_FAILURE);
   }
}

void BBFindConfRemapProbe::insertFileIntoMontage(char const * labelFilename, int xOffset, int yOffset, int xExpectedSize, int yExpectedSize) {
   assert(parent->columnId()==0);
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const nf = imageLoc->nf;
   GDALDataset * dataset = (GDALDataset *) GDALOpen(labelFilename, GA_ReadOnly);
   if (dataset==NULL) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error opening label file \"%s\" for reading.\n", getKeyword(), name, labelFilename);
      exit(EXIT_FAILURE);
   }
   int xLabelSize = dataset->GetRasterXSize();
   int yLabelSize = dataset->GetRasterYSize();
   int labelBands = dataset->GetRasterCount();
   if (xLabelSize != xExpectedSize || yLabelSize != yExpectedSize) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error: label files \"%s\" has dimensions %dx%d (expected %dx%d)\n",
            getKeyword(), name, labelFilename, xLabelSize, yLabelSize, xExpectedSize, yExpectedSize);
      exit(EXIT_FAILURE);
   }
   // same xStart.
   int offsetIdx = kIndex(xOffset, yOffset, 0, montageDimX, montageDimY, 3);
   dataset->RasterIO(GF_Read, 0, 0, xLabelSize, yLabelSize, &montageImage[offsetIdx], xLabelSize, yLabelSize, GDT_Byte, 3/*number of bands*/, NULL, 3/*x-stride*/, 3*montageDimX/*y-stride*/, 1/*band stride*/);
   GDALClose(dataset);
}

void BBFindConfRemapProbe::insertImageIntoMontage(int xStart, int yStart, pvadata_t const * sourceData, PVLayerLoc const * loc, bool extended) {
   pvadata_t minValue = std::numeric_limits<pvadata_t>::infinity();
   pvadata_t maxValue = -std::numeric_limits<pvadata_t>::infinity();
   int const nx = loc->nx;
   int const ny = loc->ny;
   int const nf = loc->nf;
   PVHalo const * halo = &loc->halo;
   int const numImageNeurons = nx*ny*nf;
   if (extended) {
      for (int k=0; k<numImageNeurons; k++) {
         int const kExt = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         pvadata_t a = sourceData[kExt];
         minValue = a < minValue ? a : minValue;
         maxValue = a > maxValue ? a : maxValue;
      }
   }
   else {
      for (int k=0; k<numImageNeurons; k++) {
         pvadata_t a = sourceData[k];
         minValue = a < minValue ? a : minValue;
         maxValue = a > maxValue ? a : maxValue;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &minValue, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &maxValue, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
   if (minValue==maxValue) {
      for (int y=0; y<ny; y++) {
         int lineStart = kIndex(0, y, 0, nx, ny, nf);
         memset(&montageImageLocal[lineStart], 127, (size_t) (nx*nf));
      }
   }
   else {
      pvadata_t scale = (pvadata_t) 1/(maxValue-minValue);
      pvadata_t shift = minValue;
      for (int k=0; k<numImageNeurons; k++) {
         int const kx = kxPos(k, nx, ny, nf);
         int const ky = kyPos(k, nx, ny, nf);
         int const kf = featureIndex(k, nx, ny, nf);
         int const kImageExt = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         pvadata_t a = sourceData[kImageExt];
         a = nearbyintf(255*(a-shift)*scale);
         unsigned char aChar = (unsigned char) (int) a;
         montageImageLocal[k] = aChar;
      }
   }
   if (parent->columnId()!=0) {
      MPI_Send(montageImageLocal, nx*ny*3, MPI_UNSIGNED_CHAR, 0, 111, parent->icCommunicator()->communicator());
   }
   else {
      for (int rank=0; rank<parent->numberOfColumns(); rank++) {
         if (rank!=0) {
            MPI_Recv(montageImageComm, nx*ny*3, MPI_UNSIGNED_CHAR, rank, 111, parent->icCommunicator()->communicator(), MPI_STATUS_IGNORE);
         }
         else {
            memcpy(montageImageComm, montageImageLocal, nx*ny*3);
         }
         int const numCommRows = parent->icCommunicator()->numCommRows();
         int const numCommCols = parent->icCommunicator()->numCommColumns();
         int const commRow = rowFromRank(rank, numCommRows, numCommCols);
         int const commCol = columnFromRank(rank, numCommRows, numCommCols);
         for (int y=0; y<ny; y++) {
            int destIdx = kIndex(xStart+commCol*nx, yStart+commRow*ny+y, 0, montageDimX, montageDimY, 3);
            int srcIdx = kIndex(0, y, 0, nx, ny, 3);
            memcpy(&montageImage[destIdx], &montageImageComm[srcIdx], nx*3);
         }
      }
   }
}

void BBFindConfRemapProbe::writeMontage() {
   std::stringstream montagePathSStream("");
   montagePathSStream << heatMapMontageDir << "/" << outputFilenameBase << "_" << parent->getCurrentStep();
   bool isLastTimeStep = parent->simulationTime() >= parent->getStopTime() - parent->getDeltaTimeBase()/2;
   if (isLastTimeStep) { montagePathSStream << "_final"; }
   montagePathSStream << ".tif";
   char * montagePath = strdup(montagePathSStream.str().c_str()); // not sure why I have to strdup this
   if (montagePath==NULL) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error: unable to create montagePath\n", getKeyword(), name);
      exit(EXIT_FAILURE);
   }
   GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");
   if (driver == NULL) {
      fflush(stdout);
      fprintf(stderr, "GetGDALDriverManager()->GetDriverByName(\"GTiff\") failed.");
      exit(EXIT_FAILURE);
   }
   GDALDataset * dataset = driver->Create(montagePath, montageDimX, montageDimY, 3/*numBands*/, GDT_Byte, NULL);
   if (dataset == NULL) {
      fprintf(stderr, "GDAL failed to open file \"%s\"\n", montagePath);
      exit(EXIT_FAILURE);
   }
   free(montagePath);
   dataset->RasterIO(GF_Write, 0, 0, montageDimX, montageDimY, montageImage, montageDimX, montageDimY, GDT_Byte, 3/*numBands*/, NULL, 3/*x-stride*/, 3*montageDimX/*y-stride*/, 1/*band-stride*/);
   GDALClose(dataset);
}

BBFindConfRemapProbe::~BBFindConfRemapProbe() {
}

PV::BaseObject * createBBFindConfRemapProbe(char const * name, PV::HyPerCol * hc) {
   return hc ? new BBFindConfRemapProbe(name, hc) : nullptr;
}

