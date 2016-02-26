/*
 * LocalizationProbe.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <limits>
#include "LocalizationProbe.hpp"

LocalizationProbe::LocalizationProbe(const char * probeName, PV::HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

LocalizationProbe::LocalizationProbe() {
   initialize_base();
}

int LocalizationProbe::initialize_base() {
   imageLayerName = NULL;
   reconLayerName = NULL;
   classNamesFile = NULL;
   classNames = NULL;
   displayCategoryIndexStart = -1;
   displayCategoryIndexEnd = -1;
   detectionThreshold = 0.0f;
   heatMapMaximum = 1.0f;
   heatMapMontageDir = NULL;
   drawMontage = false;
   displayCommand = NULL;

   outputPeriod = 1.0;
   nextOutputTime = 0.0; // Warning: this does not get checkpointed but it should.  Probes have no checkpointing infrastructure yet.
   imageLayerName = NULL;
   imageLayer = NULL;
   reconLayerName = NULL;
   reconLayer = NULL;

   imageDilationX = 1.0;
   imageDilationY = 1.0;
   numMontageRows = -1;
   numMontageColumns = -1;
   montageDimX = -1;
   montageDimY = -1;
   grayScaleImage = NULL;
   montageImage = NULL;
   imageBlendCoeff = 0.3;

   outputFilenameBase = NULL; // Not used by harness since we don't have a filename to use for the base
   return PV_SUCCESS;
}

int LocalizationProbe::initialize(const char * probeName, PV::HyPerCol * hc) {
   if (hc->icCommunicator()->globalCommSize()!=1) {
      fflush(stdout);
      fprintf(stderr, "LocalizationProbe is not yet MPI capable. Exiting.");
      exit(EXIT_FAILURE);
   }
   outputPeriod = hc->getDeltaTimeBase(); // default outputPeriod is every timestep
   int status = PV::LayerProbe::initialize(probeName, hc);
   PV::InterColComm * icComm = parent->icCommunicator();
   if (status != PV_SUCCESS) { exit(EXIT_FAILURE); }
   outputFilenameBase = strdup("out"); // The harness doesn't provide a filename, so we use this as the file base name for every image.
   return status;
}

int LocalizationProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV::LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_imageLayer(ioFlag);
   ioParam_reconLayer(ioFlag);
   ioParam_detectionThreshold(ioFlag);
   ioParam_heatMapMaximum(ioFlag);
   ioParam_classNamesFile(ioFlag);
   ioParam_outputPeriod(ioFlag);
   ioParam_drawMontage(ioFlag);
   ioParam_displayCategoryIndexStart(ioFlag);
   ioParam_displayCategoryIndexEnd(ioFlag);
   ioParam_heatMapMontageDir(ioFlag);
   ioParam_imageBlendCoeff(ioFlag);
   ioParam_displayCommand(ioFlag);
   return status;
}

void LocalizationProbe::ioParam_imageLayer(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "imageLayer", &imageLayerName);
}

void LocalizationProbe::ioParam_reconLayer(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "reconLayer", &reconLayerName);
}

void LocalizationProbe::ioParam_detectionThreshold(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "detectionThreshold", &detectionThreshold, detectionThreshold);
}

void LocalizationProbe::ioParam_heatMapMaximum(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "heatMapMaximum", &heatMapMaximum, heatMapMaximum);
}

void LocalizationProbe::ioParam_classNamesFile(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "classNamesFile", &classNamesFile, "");
}

void LocalizationProbe::ioParam_outputPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "triggerLayer"));
   if (!triggerLayer) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "outputPeriod", &outputPeriod, outputPeriod, true/*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_drawMontage(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamValue(ioFlag, this->getName(), "drawMontage", &drawMontage, drawMontage, true/*warnIfAbsent*/);
}

void LocalizationProbe::ioParam_displayCategoryIndexStart(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "displayCategoryIndexStart", &displayCategoryIndexStart, -1, true/*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_displayCategoryIndexEnd(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "displayCategoryIndexEnd", &displayCategoryIndexEnd, -1, true/*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_heatMapMontageDir(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamStringRequired(ioFlag, this->getName(), "heatMapMontageDir", &heatMapMontageDir);
   }
}

void LocalizationProbe::ioParam_imageBlendCoeff(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamValue(ioFlag, this->getName(), "imageBlendCoeff", &imageBlendCoeff, imageBlendCoeff/*default value*/, true/*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_displayCommand(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->ioParamString(ioFlag, this->getName(), "displayCommand", &displayCommand, NULL, true/*warnIfAbsent*/);     
   }
}


int LocalizationProbe::initNumValues() {
   return setNumValues(6); // winningFeature,maxActivity,boundingBoxLeft,boundingBoxRight,boundingBoxTop,boundingBoxBottom
}

int LocalizationProbe::communicateInitInfo() {
   int status = PV::LayerProbe::communicateInitInfo();
   assert(targetLayer);
   int const nf = targetLayer->getLayerLoc()->nf;
   if (drawMontage && heatMapMaximum < detectionThreshold) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": heatMapMaximum (%f) cannot be less than detectionThreshold (%f).\n",
               getKeyword(), getName(), heatMapMaximum, detectionThreshold);
         exit(EXIT_FAILURE);
      }
   }
   imageLayer = parent->getLayerFromName(imageLayerName);
   if (imageLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: imageLayer \"%s\" does not refer to a layer in the column.\n",
               getKeyword(), name, imageLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (displayCategoryIndexStart <= 0) {
      displayCategoryIndexStart = 1;
   }
   if (displayCategoryIndexEnd <= 0) {
      displayCategoryIndexEnd = nf;
   }
   imageDilationX = pow(2.0, targetLayer->getXScale() - imageLayer->getXScale());
   imageDilationY = pow(2.0, targetLayer->getYScale() - imageLayer->getYScale());

   if (drawMontage) {
      setOptimalMontage();
   }

   reconLayer = parent->getLayerFromName(reconLayerName);
   if (reconLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: reconLayer \"%s\" does not refer to a layer in the column.\n",
               name, getKeyword(), reconLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   // Check Parameters in LocalizationProbe that are to be passed to Octave.
   if (parent->columnId()==0 && drawMontage) {
      featurefieldwidth = (int) ceilf(log10f((float) (nf+1)));
      classNames = (char **) malloc(nf * sizeof(char *));
      if (classNames == NULL) {
         fprintf(stderr, "%s \"%s\" unable to allocate classNames: %s\n", getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      if (strcmp(classNamesFile,"")) {
         std::ifstream * classNamesStream = new std::ifstream(classNamesFile);
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
         printf("classNamesFile was not set in params file; Class names will be feature indices.\n");
         for (int k=0; k<nf; k++) {
            std::stringstream classNameString("");
            classNameString << "Feature " << k;
            classNames[k] = strdup(classNameString.str().c_str());
            if (classNames[k]==NULL) {
               fprintf(stderr, "%s \"%s\": unable to allocate className %d: %s\n", getKeyword(), name, k, strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
      }

      if (drawMontage) {
         // Make the heatmap montage directory if it doesn't already exist.
         struct stat heatMapMontageStat;
         status = stat(heatMapMontageDir, &heatMapMontageStat);
         if (status!=0 && errno==ENOENT) {
            status = mkdir(heatMapMontageDir, 0775);
            if (status!=0) {
               fflush(stdout);
               fprintf(stderr, "Error: Unable to make heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
               exit(EXIT_FAILURE);
            }
            status = stat(heatMapMontageDir, &heatMapMontageStat);
         }
         if (status!=0) {
            fflush(stdout);
            fprintf(stderr, "Error: Unable to get status of heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
            exit(EXIT_FAILURE);
         }
         if (!(heatMapMontageStat.st_mode & S_IFDIR)) {
            fflush(stdout);
            fprintf(stderr, "Error: Heat map montage \"%s\" is not a directory\n", heatMapMontageDir);
            exit(EXIT_FAILURE);
         }
         // Make the labels directory in heatMapMontageDir if it doesn't already exist
         std::stringstream labelsdirss("");
         labelsdirss << heatMapMontageDir << "/labels";
         char * labelsDir = strdup(labelsdirss.str().c_str());
         if (labelsDir==NULL) {
            fflush(stdout);
            fprintf(stderr, "Errror creating path to heat map montage labels directory: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
         }
         struct stat labelsDirStat;
         status = stat(labelsDir, &labelsDirStat);
         if (status!=0) {
            if (errno==ENOENT) {
               errno = 0;
               status = mkdir(labelsDir, 0775);
            }
            if (status!=0) {
               fflush(stdout);
               fprintf(stderr, "%s \"%s\" error: Unable to verify that labels directory \"%s\" exists: %s\n", getKeyword(), name, labelsDir, strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
         else {
            if (!(labelsDirStat.st_mode & S_IFDIR)) {
               fflush(stdout);
               fprintf(stderr, "%s \"%s\" error: path \"%s\" for the labels is not a directory.\n", getKeyword(), name, labelsDir);
               exit(EXIT_FAILURE);
            }
         }
         // make the labels
         char const * originalLabelName = "original.tif";
         status = makeMontageLabelfile(originalLabelName, "black", "white", "original image");
         if (status != 0) {
            fflush(stdout);
            fprintf(stderr, "%s \"%s\" error creating label file \"%s\".\n", getKeyword(), name, originalLabelName);
            exit(EXIT_FAILURE);
         }

         char const * reconLabelName = "reconstruction.tif";
         status = makeMontageLabelfile(reconLabelName, "black", "white", "reconstruction");
         if (status != 0) {
            fflush(stdout);
            fprintf(stderr, "%s \"%s\" error creating label file \"%s\".\n", getKeyword(), name, reconLabelName);
            exit(EXIT_FAILURE);
         }

         int const nf = targetLayer->getLayerLoc()->nf;
         char labelFilename[PV_PATH_MAX]; // this size is overkill for this situation
         for (int f=0; f<nf; f++) {
            int slen;
            slen = snprintf(labelFilename, PV_PATH_MAX, "gray%0*d.tif", featurefieldwidth, f);
            if (slen>=PV_PATH_MAX) {
               fflush(stdout);
               fprintf(stderr, "%s \"%s\" error: file name for label %d is too long (%d characters versus %d).\n", getKeyword(), name, f, slen, PV_PATH_MAX);
               exit(EXIT_FAILURE);
            }
            status = makeMontageLabelfile(labelFilename, "white", "gray", classNames[f]);
            if (status != 0) {
               fflush(stdout);
               fprintf(stderr, "%s \"%s\" error creating label file \"%s\".\n", getKeyword(), name, reconLabelName);
               exit(EXIT_FAILURE);
            }

            slen = snprintf(labelFilename, PV_PATH_MAX, "blue%0*d.tif", featurefieldwidth, f);
            if (slen>=PV_PATH_MAX) {
               fflush(stdout);
               fprintf(stderr, "%s \"%s\" error: file name for label %d is too long (%d characters versus %d).\n", getKeyword(), name, f, slen, PV_PATH_MAX);
               exit(EXIT_FAILURE);
            }
            status = makeMontageLabelfile(labelFilename, "white", "blue", classNames[f]);
            if (status != 0) {
               exit(EXIT_FAILURE);
            }
         }
      }
   }

   return status;
}

int LocalizationProbe::setOptimalMontage() {
   // Find the best number of rows and columns to use for the montage
   int const numCategories = displayCategoryIndexEnd - displayCategoryIndexStart + 1;
   assert(numCategories>0);
   int numRows[numCategories];
   float totalSizeX[numCategories];
   float totalSizeY[numCategories];
   float aspectRatio[numCategories];
   float ldfgr[numCategories]; // log-distance to golden ratio
   float loggoldenratio = logf(0.5f * (1.0f + sqrtf(5.0f)));
   for (int numCol=1; numCol <= numCategories ; numCol++) {
      int idx = numCol-1;
      numRows[idx] = (int) ceil((float) numCategories/(float) numCol);
      totalSizeX[idx] = numCol * (imageLayer->getLayerLoc()->nx + 10); // +10 for spacing between images.
      totalSizeY[idx] = numRows[idx] * (imageLayer->getLayerLoc()->ny + 64 + 10); // +64 for category label
      aspectRatio[idx] = (float) totalSizeX[idx]/(float) totalSizeY[idx];
      ldfgr[idx] = fabsf(log(aspectRatio[idx]) - loggoldenratio);
   }
   numMontageColumns = -1;
   float minldfgr = std::numeric_limits<float>::infinity();
   for (int numCol=1; numCol <= numCategories ; numCol++) {
      int idx = numCol-1;
      if (ldfgr[idx] < minldfgr) {
         minldfgr = ldfgr[idx];
         numMontageColumns = numCol;
      }
   }
   assert(numMontageColumns > 0);
   numMontageRows = numRows[numMontageColumns-1];
   while ((numMontageColumns-1)*numMontageRows >= numCategories) { numMontageColumns--; }
   while ((numMontageRows-1)*numMontageColumns >= numCategories) { numMontageRows--; }
   if (numMontageRows < 2) { numMontageRows = 2; }
}

char const * LocalizationProbe::getClassName(int k) {
    if (k<0 || k >= targetLayer->getLayerLoc()->nf) { return NULL; }
    else { return classNames[k]; }
}

int LocalizationProbe::allocateDataStructures() {
   int status = PV::LayerProbe::allocateDataStructures();
   if (status != PV_SUCCESS) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\": LocalizationProbe::allocateDataStructures failed.\n", getKeyword(), name);
      exit(EXIT_FAILURE);
   }
   if (drawMontage) {
      assert(imageLayer);
      PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
      int const nx = imageLoc->nx;
      int const ny = imageLoc->ny;
      grayScaleImage = (pvadata_t *) calloc(nx*ny, sizeof(pvadata_t));
      if (grayScaleImage==NULL) {
         fprintf(stderr, "%s \"%s\" error allocating for montage background image: %s\n", getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
      montageDimX = (nx + 10) * (numMontageColumns + 2);
      montageDimY = (ny + 64 + 10) * numMontageRows;
      montageImage = (unsigned char *) calloc((montageDimX) * montageDimY * 3, sizeof(unsigned char));
      if (montageImage==NULL) {
         fprintf(stderr, "%s \"%s\" error allocating for heat map montage image: %s\n", getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }

      int xStart = (2*numMontageColumns+1)*(nx+10)/2; // Integer division
      int yStart = 32+5;
      status = insertLabelIntoMontage("original.tif", xStart, yStart, nx, 32/*yExpectedSize*/);
      if (status != PV_SUCCESS) {
         fflush(stdout);
         fprintf(stderr, "%s \"%s\" error placing the \"original image\" label.\n", getKeyword(), name);
         exit(EXIT_FAILURE);
      }

      // same xStart.
      yStart += ny+64+10; // yStart moves down one panel.
      status = insertLabelIntoMontage("reconstruction.tif", xStart, yStart, nx, 32/*yExpectedSize*/);
      if (status != PV_SUCCESS) {
         fflush(stdout);
         fprintf(stderr, "%s \"%s\" error placing the \"reconstruction\" label.\n", getKeyword(), name);
         exit(EXIT_FAILURE);
      }

      for (int f=displayCategoryIndexStart-1; f<displayCategoryIndexEnd; f++) {
         int fDisplayed = f-displayCategoryIndexStart+1;
         int montageCol=kxPos(fDisplayed, numMontageColumns, numMontageRows, 1);
         int montageRow=kyPos(fDisplayed, numMontageColumns, numMontageRows, 1);
         int xStart = montageCol * (nx+10) + 5;
         int yStart = montageRow * (ny+64+10) + 5;
         char filename[16]; // Should be enough for the labels produced during communicateInitInfo.
         int slen = snprintf(filename, 16, "gray%0*d.tif", featurefieldwidth, f);
         if (slen >= 16) {
            fflush(stdout);
            fprintf(stderr, "%s \"%s\" allocateDataStructures error: featurefieldwidth of %d is too large.\n",
                  getKeyword(), name, featurefieldwidth);
            exit(EXIT_FAILURE);
         }
         status = insertLabelIntoMontage(filename, xStart, yStart, nx, 32/*yExpectedSize*/);
         if (status != PV_SUCCESS) {
            fflush(stdout);
            fprintf(stderr, "%s \"%s\" error placing the label for feature %d.\n", getKeyword(), name, f);
            exit(EXIT_FAILURE);
         }
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::makeMontageLabelfile(char const * labelFilename, char const * backgroundColor, char const * textColor, char const * labelText) {
   int const nx = imageLayer->getLayerLoc()->nx;
   std::stringstream convertCmd("");
   convertCmd << "convert -depth 8 -background \"" << backgroundColor << "\" -fill \"" << textColor << "\" -size " << nx << "x32 -pointsize 24 -gravity center label:\"" << labelText << "\" \"" << heatMapMontageDir << "/labels/" << labelFilename << "\"";
   int status = system(convertCmd.str().c_str());
   if (status != 0) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error creating label file \"%s\": ImageMagick convert returned %d.\n", getKeyword(), name, labelFilename, status);
      exit(EXIT_FAILURE);
   }
   return status;
}

int LocalizationProbe::insertLabelIntoMontage(char const * labelname, int xOffset, int yOffset, int xExpectedSize, int yExpectedSize) {
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const nf = imageLoc->nf;
   std::stringstream labelFilenamesstr("");
   labelFilenamesstr << heatMapMontageDir << "/labels/" << labelname;
   char * labelFilename = strdup(labelFilenamesstr.str().c_str());
   if (labelFilename==NULL) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error creating path for reading reconstruction label.\n", getKeyword(), name);
      return PV_FAILURE;
   }
   GDALDataset * dataset = (GDALDataset *) GDALOpen(labelFilename, GA_ReadOnly);
   if (dataset==NULL) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\" error opening label file \"%s\" for reading.\n", getKeyword(), name, labelFilename);
      return PV_FAILURE;
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
   return PV_SUCCESS;
}

int LocalizationProbe::setOutputFilenameBase(char const * fn) {
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
   return status;
}

bool LocalizationProbe::needUpdate(double timed, double dt) {
   bool updateNeeded = false;
   if (triggerLayer) {
      updateNeeded = LayerProbe::needUpdate(timed, dt);
   }
   else {
      if (timed>=nextOutputTime) {
         nextOutputTime += outputPeriod;
         updateNeeded = true;
      }
      else {
         updateNeeded = false;
      }
   }
   return updateNeeded;
}

int LocalizationProbe::calcValues(double timevalue) {
   int winningFeature, xLocation, yLocation;
   pvadata_t maxActivity;
   findMaxLocation(&winningFeature, &xLocation, &yLocation, &maxActivity);
   double * values = getValuesBuffer();
   if (winningFeature >= 0) {
      assert(xLocation >= 0 && yLocation >= 0);
      values[0] = (double) winningFeature;
      values[1] = maxActivity;
      double * boundingBoxDbl = &values[2];
      if (maxActivity>=detectionThreshold) {
         int boundingBox[4];
         findBoundingBox(winningFeature, xLocation, yLocation, boundingBox);
         boundingBoxDbl[0] = (double) boundingBox[0] * imageDilationX;
         boundingBoxDbl[1] = (double) boundingBox[1] * imageDilationX;
         boundingBoxDbl[2] = (double) boundingBox[2] * imageDilationY;
         boundingBoxDbl[3] = (double) boundingBox[3] * imageDilationY;
      }
      else {
         values[2] = -1.0;
         values[3] = -1.0;
         values[4] = -1.0;
         values[5] = -1.0;
      }
   }
   else {
      assert(xLocation < 0 && yLocation < 0);
      values[0] = -1.0;
      values[1] = 0.0;
      values[2] = -1.0;
      values[3] = -1.0;
      values[4] = -1.0;
      values[5] = -1.0;
   }
   return PV_SUCCESS;
}

int LocalizationProbe::findMaxLocation(int * winningFeature, int * xLocation, int * yLocation, pvadata_t * maxActivity) {
   int const N = targetLayer->getNumNeurons();
   PVLayerLoc const * loc = targetLayer->getLayerLoc();
   PVHalo const * halo = &loc->halo;
   int maxLocation = -1;
   pvadata_t maxVal = -std::numeric_limits<pvadata_t>::infinity();
   for (int n=0; n<N; n++) {
      int nExt = kIndexExtended(n, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
      pvadata_t const a = targetLayer->getLayerData()[nExt];
      if (a>maxVal) {
         maxVal = a;
         maxLocation = n;
      }
   }
   // Need to MPI reduce here
   if (maxLocation>=0) {
      *winningFeature = featureIndex(maxLocation, loc->nx, loc->ny, loc->nf);
      *xLocation = kxPos(maxLocation, loc->nx, loc->ny, loc->nf);
      *yLocation = kyPos(maxLocation, loc->nx, loc->ny, loc->nf);
   }
   else {
      *winningFeature = -1;
      *xLocation = -1;
      *yLocation = -1;
   }
   *maxActivity = maxVal;
   return PV_SUCCESS;
}

int LocalizationProbe::findBoundingBox(int winningFeature, int xLocation, int yLocation, int * boundingBox) {
   if (winningFeature>=0 && xLocation>=0 && yLocation>=0) {
      int lt = xLocation;
      int rt = xLocation;
      int up = yLocation;
      int dn = yLocation;
      int const N = targetLayer->getNumNeurons();
      PVLayerLoc const * loc = targetLayer->getLayerLoc();
      PVHalo const * halo = &loc->halo;
      for (int n=winningFeature; n<N; n+=loc->nf) {
         int nExt = kIndexExtended(n, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
         pvadata_t const a = targetLayer->getLayerData()[nExt];
         if (a>detectionThreshold) {
            int x = kxPos(n, loc->nx, loc->ny, loc->nf);
            int y = kyPos(n, loc->nx, loc->ny, loc->nf);
            if (x<lt) { lt = x; }
            if (x>rt) { rt = x; }
            if (y<up) { up = y; }
            if (y>dn) { dn = y; }
         }
      }
      boundingBox[0] = lt;
      boundingBox[1] = rt+1;
      boundingBox[2] = up;
      boundingBox[3] = dn+1;
   }
   else {
      for (int k=0; k<4; k++) { boundingBox[k] = -1; }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::outputState(double timevalue) {
   getValues(timevalue);
   double * values = getValuesBuffer();
   int winningFeature = (int) values[0];
   double maxActivity = values[1];
   if (winningFeature >= 0) {
      if (maxActivity >= detectionThreshold) {
         fprintf(outputstream->fp, "Time %f, activity %f, bounding box x=[%d,%d), y=[%d,%d): detected %s\n",
               timevalue,
               maxActivity,
               (int) values[2],
               (int) values[3],
               (int) values[4],
               (int) values[5],
               getClassName(winningFeature));
      }
      else {
         fprintf(outputstream->fp, "Time %f, no features detected above threshold %f (highest was %s at %f)\n", timevalue, detectionThreshold, getClassName(winningFeature), maxActivity);
      }
   }
   else {
      fprintf(outputstream->fp, "Time %f, no features detected.\n", timevalue);
      // no activity above threshold
   }
   fflush(outputstream->fp);

   int status = PV_SUCCESS;
   if (drawMontage) {
      status = makeMontage();
   }     
   return status;
}

int LocalizationProbe::makeMontage() {
   assert(drawMontage);
   assert(numMontageRows > 0 && numMontageColumns > 0);
   assert(grayScaleImage);
   assert(montageImage);

   // create grayscale version of image layer for background of heat maps.
   PVLayerLoc const * imageLoc = imageLayer->getLayerLoc();
   PVHalo const * halo = &imageLoc->halo;
   int const nx = imageLoc->nx;
   int const ny = imageLoc->ny;
   int const nf = imageLoc->nf;
   int const N = nx * ny;
   pvadata_t const * imageActivity = imageLayer->getLayerData();
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

   // for each displayed category, copy grayScaleImage to the relevant part of the montage, and impose the upsampled version
   // of the target layer onto it.

   pvadata_t thresholdColor[] = {0.5f, 0.5f, 0.5f}; // rgb color of heat map when activity is at or below detectionThreshold
   pvadata_t heatMapColor[] = {0.0f, 1.0f, 0.0f};   // rgb color of heat map when activity is at or above heatMapMaximum

   PVLayerLoc const * targetLoc = targetLayer->getLayerLoc();
   PVHalo const * targetHalo = &targetLoc->halo;
   int winningFeature = (int) getValuesBuffer()[0];
   double maxConfidence = getValuesBuffer()[1];
   for (int f=displayCategoryIndexStart-1; f<displayCategoryIndexEnd; f++) {
      int montageCol = (f-displayCategoryIndexStart+1) % numMontageColumns;
      int montageRow = (f-displayCategoryIndexStart+1 - montageCol) / numMontageColumns; // Integer arithmetic
      int xStartInMontage = (nx + 10)*montageCol + 5;
      int yStartInMontage = (ny + 64 + 10)*montageRow + 64 + 5;
      for (int y=0; y<ny; y++) {
         for (int x=0; x<nx; x++) {
            pvadata_t backgroundLevel = grayScaleImage[x + nx * y];
            int xTarget = (int) ((double) x/imageDilationX);
            int yTarget = (int) ((double) y/imageDilationY);
            int targetIdx = kIndex(xTarget, yTarget, f, targetLoc->nx, targetLoc->ny, targetLoc->nf);
            int targetIdxExt = kIndexExtended(targetIdx, targetLoc->nx, targetLoc->ny, targetLoc->nf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
            pvadata_t heatMapLevel = targetLayer->getLayerData()[targetIdxExt];
            heatMapLevel = (heatMapLevel - detectionThreshold)/(heatMapMaximum-detectionThreshold);
            heatMapLevel = heatMapLevel < (pvadata_t) 0 ? (pvadata_t) 0 : heatMapLevel > (pvadata_t) 1 ? (pvadata_t) 1 : heatMapLevel;
            int montageIdx = kIndex(xStartInMontage + x, yStartInMontage + y, 0, montageDimX, montageDimY, 3);
            for(int rgb=0; rgb<3; rgb++) { 
               pvadata_t h = heatMapLevel * heatMapColor[rgb] + (1-heatMapLevel) * thresholdColor[rgb];
               pvadata_t g = imageBlendCoeff * backgroundLevel + (1-imageBlendCoeff) * h;
               assert(g>=(pvadata_t) -0.001 && g <= (pvadata_t) 1.001);
               g = nearbyintf(255*g);
               unsigned char gchar = (unsigned char) g; // (g<0.0f ? 0.0f : g>255.0f ? 255.0f : g);
               assert(montageIdx>=0 && montageIdx+rgb<montageDimX*montageDimY*3);
               montageImage[montageIdx + rgb] = gchar;
            }
         }
      }

      // Draw labels and confidences
      pvadata_t maxConfInCategory = -std::numeric_limits<pvadata_t>::infinity();
      PVLayerLoc const * targetLoc = targetLayer->getLayerLoc();
      PVHalo const * targetHalo = &targetLoc->halo;
      int const numNeurons = targetLayer->getNumNeurons();
      int const numFeatures = targetLoc->nf;
      for (int k=f; k<numNeurons; k+=numFeatures) {
         int kExt = kIndexExtended(k, targetLoc->nx, targetLoc->ny, numFeatures, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
         pvadata_t a = targetLayer->getLayerData()[kExt];
         maxConfInCategory = a > maxConfInCategory ? a : maxConfInCategory;
      }
      char confidenceText[32];
      int slen = snprintf(confidenceText, 32, "%.1f%%", 100*maxConfInCategory);
      if (slen >= 32) {
         fflush(stdout);
         fprintf(stderr, "Formatted text for confidence %f of %d is too long.\n", maxConfInCategory, f);
         exit(EXIT_FAILURE);
      }
      char const * textColor = NULL;
      char const * confFile = "confidenceLabel.tif";
      if (f==winningFeature) {
         char labelFilename[PV_PATH_MAX];
         int slen = snprintf(labelFilename, PV_PATH_MAX, "blue%0*d.tif", featurefieldwidth, f);
         assert(slen<PV_PATH_MAX); // it fit when making the labels; it should fit now.
         insertLabelIntoMontage(labelFilename, xStartInMontage, yStartInMontage-64, nx, 32);
         
         textColor = "blue";
      }
      else {
         textColor = "gray";
      }
      makeMontageLabelfile(confFile, "white", textColor, confidenceText);
      insertLabelIntoMontage(confFile, xStartInMontage, yStartInMontage-32, nx, 32);
   }

   // Draw original image
   assert(nf==3); // I've been assuming the image layer has three features.  What if it doesn't?
   int xStart = 5+(2*numMontageColumns+1)*(nx+10)/2;
   int yStart = 5+64;

   minValue = std::numeric_limits<pvadata_t>::infinity();
   maxValue = -std::numeric_limits<pvadata_t>::infinity();
   int const numImageNeurons = imageLayer->getNumNeurons();
   for (int k=0; k<numImageNeurons; k++) {
      int const kExt = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      pvadata_t a = imageLayer->getLayerData()[kExt];
      minValue = a < minValue ? a : minValue;
      maxValue = a > maxValue ? a : maxValue;
   }
   if (minValue==maxValue) {
      for (int y=0; y<ny; y++) {
         int lineStart = kIndex(xStart, yStart+y, 0, montageDimX, montageDimY, nf);
         memset(&montageImage[lineStart], 127, (size_t) (nx*nf));
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
         pvadata_t a = imageLayer->getLayerData()[kImageExt];
         a = nearbyintf(255*(a-shift)*scale);
         unsigned char aChar = (unsigned char) (int) a;
         int const kMontage = kIndex(xStart+kx, yStart+ky, kf, montageDimX, montageDimY, nf);
         montageImage[kMontage] = aChar;
      }
   }

   // Draw reconstructed image
   if (reconLayer) {
      // same xStart, yStart is down one row.
      // I should check that reconLayer and imageLayer have the same dimensions
      yStart += ny + 64 + 10;

      minValue = std::numeric_limits<pvadata_t>::infinity();
      maxValue = -std::numeric_limits<pvadata_t>::infinity();
      for (int k=0; k<numImageNeurons; k++) {
         int const kExt = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         pvadata_t a = reconLayer->getLayerData()[kExt];
         minValue = a < minValue ? a : minValue;
         maxValue = a > maxValue ? a : maxValue;
      }
      if (minValue==maxValue) {
         for (int y=0; y<ny; y++) {
            int lineStart = kIndex(xStart, yStart+y, 0, montageDimX, montageDimY, nf);
            memset(&montageImage[lineStart], 127, (size_t) (nx*nf));
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
            pvadata_t a = reconLayer->getLayerData()[kImageExt];
            a = nearbyintf(255*(a-shift)*scale);
            unsigned char aChar = (unsigned char) (int) a;
            int const kMontage = kIndex(xStart+kx, yStart+ky, kf, montageDimX, montageDimY, nf);
            montageImage[kMontage] = aChar;
         }
      }
   }

   // write out montageImage
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

   // Restore the winning feature's gray label
   if (winningFeature >= displayCategoryIndexStart-1 && winningFeature <= displayCategoryIndexEnd-1) {
      int montageColumn = kxPos(winningFeature - displayCategoryIndexStart + 1, numMontageColumns, numMontageRows, 1);
      int montageRow = kyPos(winningFeature - displayCategoryIndexStart + 1, numMontageColumns, numMontageRows, 1);
      int xStartInMontage = montageColumn * (nx+10) + 5;
      int yStartInMontage = montageRow * (ny+64+10) + 5;
      char labelFilename[PV_PATH_MAX];
      int slen = snprintf(labelFilename, PV_PATH_MAX, "gray%0*d.tif", featurefieldwidth, winningFeature);
      assert(slen<PV_PATH_MAX); // it fit when making the labels; it should fit now.
      insertLabelIntoMontage(labelFilename, xStartInMontage, yStartInMontage, nx, 32);
   }

   return PV_SUCCESS;
}

LocalizationProbe::~LocalizationProbe() {
   free(imageLayerName);
   free(reconLayerName);
   free(classNamesFile);
   free(heatMapMontageDir);
   free(displayCommand);
   free(outputFilenameBase);
   free(grayScaleImage);
   free(montageImage);
}

