/*
 * LocalizationProbe.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: pschultz
 */

#include "LocalizationProbe.hpp"
#include <gdal.h>
#include <gdal_priv.h>
#include <limits>
#include <sstream>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

LocalizationProbe::LocalizationProbe(const char *probeName, PV::HyPerCol *hc) {
   initialize_base();
   initialize(probeName, hc);
}

LocalizationProbe::LocalizationProbe() { initialize_base(); }

int LocalizationProbe::initialize_base() {
   classNamesFile            = NULL;
   classNames                = NULL;
   displayedCategories       = NULL;
   numDisplayedCategories    = 0;
   displayCategoryIndexStart = -1;
   displayCategoryIndexEnd   = -1;
   numDetectionThresholds    = 0;
   detectionThreshold        = NULL;
   heatMapMaximum            = NULL;
   heatMapMontageDir         = NULL;
   minBoundingBoxWidth       = 6;
   minBoundingBoxHeight      = 6;
   drawMontage               = false;
   displayCommand            = NULL;

   outputPeriod   = 1.0;
   nextOutputTime = 0.0; // Warning: this does not get checkpointed but it should.  Probes have no
   // checkpointing infrastructure yet.
   imageLayerName = NULL;
   imageLayer     = NULL;
   reconLayerName = NULL;
   reconLayer     = NULL;

   imageDilationX       = 1.0;
   imageDilationY       = 1.0;
   numMontageRows       = -1;
   numMontageColumns    = -1;
   montageDimX          = -1;
   montageDimY          = -1;
   grayScaleImage       = NULL;
   montageImage         = NULL;
   montageImageLocal    = NULL;
   montageImageComm     = NULL;
   imageBlendCoeff      = 0.3;
   maxDetections        = 100U;
   boundingBoxLineWidth = 5;

   outputFilenameBase =
         NULL; // Not used by harness since we don't have a filename to use for the base
   return PV_SUCCESS;
}

int LocalizationProbe::initialize(const char *probeName, PV::HyPerCol *hc) {
   outputPeriod             = hc->getDeltaTime(); // default outputPeriod is every timestep
   int status               = PV::LayerProbe::initialize(probeName, hc);
   PV::Communicator *icComm = parent->getCommunicator();
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   outputFilenameBase = strdup("out"); // Default file base name for every image.
   return status;
}

int LocalizationProbe::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
   int status = PV::LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_imageLayer(ioFlag);
   ioParam_reconLayer(ioFlag);
   ioParam_displayedCategories(ioFlag);
   ioParam_displayCategoryIndexStart(ioFlag);
   ioParam_displayCategoryIndexEnd(ioFlag);
   ioParam_detectionThreshold(ioFlag);
   ioParam_maxDetections(ioFlag);
   ioParam_classNamesFile(ioFlag);
   ioParam_outputPeriod(ioFlag);
   ioParam_minBoundingBoxWidth(ioFlag);
   ioParam_minBoundingBoxHeight(ioFlag);
   ioParam_drawMontage(ioFlag);
   ioParam_heatMapMaximum(ioFlag);
   ioParam_heatMapMontageDir(ioFlag);
   ioParam_imageBlendCoeff(ioFlag);
   ioParam_boundingBoxLineWidth(ioFlag);
   ioParam_displayCommand(ioFlag);
   return status;
}

void LocalizationProbe::ioParam_imageLayer(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "imageLayer", &imageLayerName);
}

void LocalizationProbe::ioParam_reconLayer(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "reconLayer", &reconLayerName);
}

void LocalizationProbe::ioParam_displayedCategories(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamArray(
         ioFlag,
         this->getName(),
         "displayedCategories",
         &displayedCategories,
         &numDisplayedCategories);
}

void LocalizationProbe::ioParam_displayCategoryIndexStart(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "displayedCategories"));
   if (numDisplayedCategories == 0) {
      this->getParent()->parameters()->ioParamValue(
            ioFlag,
            this->getName(),
            "displayCategoryIndexStart",
            &displayCategoryIndexStart,
            -1,
            true /*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_displayCategoryIndexEnd(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "displayedCategories"));
   if (numDisplayedCategories == 0) {
      this->getParent()->parameters()->ioParamValue(
            ioFlag,
            this->getName(),
            "displayCategoryIndexEnd",
            &displayCategoryIndexEnd,
            -1,
            true /*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_detectionThreshold(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamArray(
         ioFlag, name, "detectionThreshold", &detectionThreshold, &numDetectionThresholds);
}

void LocalizationProbe::ioParam_classNamesFile(enum PV::ParamsIOFlag ioFlag) {
   parameters()->ioParamString(ioFlag, name, "classNamesFile", &classNamesFile, "");
}

void LocalizationProbe::ioParam_outputPeriod(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "triggerLayer"));
   if (!triggerLayer) {
      this->getParent()->parameters()->ioParamValue(
            ioFlag,
            this->getName(),
            "outputPeriod",
            &outputPeriod,
            outputPeriod,
            true /*warnIfAbsent*/);
   }
   if (ioFlag == PV::PARAMS_IO_READ) {
      nextOutputTime = outputPeriod;
   }
}

void LocalizationProbe::ioParam_minBoundingBoxWidth(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "minBoundingBoxWidth",
         &minBoundingBoxWidth,
         minBoundingBoxWidth,
         true /*warnIfAbsent*/);
}

void LocalizationProbe::ioParam_minBoundingBoxHeight(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "minBoundingBoxHeight",
         &minBoundingBoxHeight,
         minBoundingBoxHeight,
         true /*warnIfAbsent*/);
}

void LocalizationProbe::ioParam_drawMontage(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamValue(
         ioFlag, this->getName(), "drawMontage", &drawMontage, drawMontage, true /*warnIfAbsent*/);
}

void LocalizationProbe::ioParam_heatMapMontageDir(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->parameters()->ioParamStringRequired(
            ioFlag, this->getName(), "heatMapMontageDir", &heatMapMontageDir);
   }
}

void LocalizationProbe::ioParam_heatMapMaximum(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      parameters()->ioParamArray(
            ioFlag, name, "heatMapMaximum", &heatMapMaximum, &numHeatMapMaxima);
   }
}

void LocalizationProbe::ioParam_imageBlendCoeff(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->parameters()->ioParamValue(
            ioFlag,
            this->getName(),
            "imageBlendCoeff",
            &imageBlendCoeff,
            imageBlendCoeff /*default value*/,
            true /*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_maxDetections(enum PV::ParamsIOFlag ioFlag) {
   this->getParent()->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "maxDetections",
         &maxDetections,
         maxDetections,
         true /*warnIfAbsent*/);
}

void LocalizationProbe::ioParam_boundingBoxLineWidth(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->parameters()->ioParamValue(
            ioFlag,
            this->getName(),
            "boundingBoxLineWidth",
            &boundingBoxLineWidth,
            boundingBoxLineWidth /*default value*/,
            true /*warnIfAbsent*/);
   }
}

void LocalizationProbe::ioParam_displayCommand(enum PV::ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(this->getName(), "drawMontage"));
   if (drawMontage) {
      this->getParent()->parameters()->ioParamString(
            ioFlag,
            this->getName(),
            "displayCommand",
            &displayCommand,
            NULL,
            true /*warnIfAbsent*/);
   }
}

int LocalizationProbe::initNumValues() {
   return setNumValues(1);
   // numValues is the number of detections.  detections will be a vector of that length.
   // If we add batching to LocalizationProbe, this should change to the batch size.
}

int LocalizationProbe::communicateInitInfo(CommunicateInitInfoMessage const *message) {
   int status = PV::LayerProbe::communicateInitInfo(message);
   assert(targetLayer);
   int const nf = targetLayer->getLayerLoc()->nf;
   imageLayer   = dynamic_cast<HyPerLayer *>(message->lookup(std::string(imageLayerName)));
   if (imageLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: imageLayer \"%s\" does not refer to a layer in the column.\n",
               getDescription_c(),
               imageLayerName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (drawMontage && imageLayer->getLayerLoc()->nf != 3) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: drawMontage requires the image layer have exactly three features.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (numDisplayedCategories == 0) {
      if (displayCategoryIndexStart <= 0) {
         displayCategoryIndexStart = 1;
         if (parent->globalRank() == 0) {
            InfoLog().printf("%s: setting displayCategoryIndexStart to 1.\n", getDescription_c());
         }
      }
      if (displayCategoryIndexEnd <= 0) {
         displayCategoryIndexEnd = nf;
         if (parent->globalRank() == 0) {
            InfoLog().printf(
                  "%s: setting displayCategoryIndexEnd to nf=%d.\n", getDescription_c(), nf);
         }
      }
      if (displayCategoryIndexEnd > nf) {
         if (parent->globalRank() == 0) {
            ErrorLog().printf(
                  "%s: displayCategoryIndexEnd=%d cannot be greater than nf=%d.\n",
                  getDescription_c(),
                  displayCategoryIndexEnd,
                  nf);
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
      numDisplayedCategories = displayCategoryIndexEnd - displayCategoryIndexStart + 1;
      if (parent->globalRank() == 0) {
         InfoLog().flush();
         InfoLog().printf(
               "%s: converting values of displayCategoryIndexStart and displayCategoryIndexEnd to "
               "a displayedCategories array.\n",
               getDescription_c());
      }

      if (numDisplayedCategories <= 0) {
         if (parent->globalRank() == 0) {
            ErrorLog().printf(
                  "%s: displayCategoryIndexStart (%d) cannot be greater than "
                  "displayCategoryIndexEnd (%d).\n",
                  getDescription_c(),
                  displayCategoryIndexStart,
                  displayCategoryIndexEnd);
         }
         MPI_Barrier(getParent()->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
      displayedCategories = (int *)malloc(numDisplayedCategories * sizeof(int));
      for (int idx = 0; idx < numDisplayedCategories; idx++) {
         displayedCategories[idx] = displayCategoryIndexStart + idx;
         assert(displayedCategories[idx] <= displayCategoryIndexEnd);
      }
   }

   if (numDetectionThresholds == 0) {
      detectionThreshold =
            (float *)malloc(sizeof(*detectionThreshold) * (size_t)numDisplayedCategories);
      if (detectionThreshold == NULL) {
         Fatal().printf(
               "%s: Unable to allocate memory for detectionThreshold array: %s\n",
               getDescription_c(),
               strerror(errno));
      }
      for (int k = 0; k < numDisplayedCategories; k++) {
         detectionThreshold[0] = 0.0f;
      }
      numDetectionThresholds = numDisplayedCategories;
   }
   else if (numDetectionThresholds == 1 && numDisplayedCategories > 1) {
      detectionThreshold = (float *)realloc(
            detectionThreshold, sizeof(*detectionThreshold) * (size_t)numDisplayedCategories);
      if (detectionThreshold == NULL) {
         Fatal().printf(
               "%s: Unable to allocate memory for detectionThreshold array: %s\n",
               getDescription_c(),
               strerror(errno));
      }
      float detThresh = detectionThreshold[0];
      for (int k = 1; k < numDisplayedCategories; k++) {
         detectionThreshold[k] = detThresh;
      }
      numDetectionThresholds = numDisplayedCategories;
   }
   else if (numDetectionThresholds != numDisplayedCategories) {
      if (parent->columnId() == 0) {
         Fatal().printf(
               "%s: detectionThreshold array given %d entries, but number of displayed categories "
               "is %d.\n",
               getDescription_c(),
               numDetectionThresholds,
               numDisplayedCategories);
      }
   }

   for (int k = 0; k < numDisplayedCategories; k++) {
      int displayedCategory = displayedCategories[k];
      if (displayedCategory <= 0 || displayedCategory > nf) {
         if (parent->columnId() == 0) {
            fprintf(
                  stderr,
                  "%s: displayedCategories must be between 1 and nf=%d inclusive. (index %d is "
                  "%d)\n",
                  getDescription_c(),
                  nf,
                  k + 1,
                  displayedCategory);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   if (drawMontage) {
      // TODO: abstract out similarities between expanding heatMapMaximum array and expanding
      // detectionThreshold array, and perhaps other arrays in pv-core.
      if (numHeatMapMaxima == 0) {
         heatMapMaximum =
               (float *)malloc(sizeof(*detectionThreshold) * (size_t)numDisplayedCategories);
         if (heatMapMaximum == NULL) {
            Fatal().printf(
                  "%s: Unable to allocate memory for heatMapMaximum array: %s\n",
                  getDescription_c(),
                  strerror(errno));
         }
         for (int k = 0; k < numDisplayedCategories; k++) {
            heatMapMaximum[k] = 1.0f;
         }
         numHeatMapMaxima = numDisplayedCategories;
      }
      else if (numHeatMapMaxima == 1 && numDisplayedCategories > 1) {
         heatMapMaximum = (float *)realloc(
               heatMapMaximum, sizeof(*heatMapMaximum) * (size_t)numDisplayedCategories);
         if (heatMapMaximum == NULL) {
            Fatal().printf(
                  "%s: Unable to allocate memory for heatMapMaximum array: %s\n",
                  getDescription_c(),
                  strerror(errno));
         }
         float heatMapMax = heatMapMaximum[0];
         for (int k = 1; k < numDisplayedCategories; k++) {
            heatMapMaximum[k] = heatMapMax;
         }
         numHeatMapMaxima = numDisplayedCategories;
      }
      else if (numHeatMapMaxima != numDisplayedCategories) {
         if (parent->columnId() == 0) {
            Fatal().printf(
                  "%s: detectionThreshold array given %d entries, but number of displayed "
                  "categories is %d.\n",
                  getDescription_c(),
                  numDetectionThresholds,
                  numDisplayedCategories);
         }
      }
      assert(status == PV_SUCCESS);
      for (int k = 0; k < numDisplayedCategories; k++) {
         if (heatMapMaximum[k] < detectionThreshold[k]) {
            status = PV_FAILURE;
            if (parent->columnId() == 0) {
               ErrorLog() << getDescription() << ": heatMapMaximum entry " << k << " ("
                          << heatMapMaximum[k]
                          << ") cannot be less than corresponding detection threshold entry"
                          << " (" << detectionThreshold[k] << ").\n";
            }
         }
      }
      if (status != PV_SUCCESS) {
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   imageDilationX = pow(2.0, targetLayer->getXScale() - imageLayer->getXScale());
   imageDilationY = pow(2.0, targetLayer->getYScale() - imageLayer->getYScale());

   if (drawMontage) {
      setOptimalMontage();
   }

   reconLayer = dynamic_cast<PV::HyPerLayer *>(message->lookup(std::string(reconLayerName)));
   if (reconLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: reconLayer \"%s\" does not refer to a layer in the column.\n",
               getDescription_c(),
               reconLayerName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (drawMontage && reconLayer->getLayerLoc()->nf != 3) {
      if (parent->columnId() != 0) {
         ErrorLog().printf(
               "%s: drawMontage requires the recon layer have exactly three features.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   // Get the names labeling each feature from the class names file.  Only the root process stores
   // these values.
   if (parent->columnId() == 0) {
      classNames = (char **)malloc(nf * sizeof(char *));
      if (classNames == NULL) {
         Fatal().printf(
               "%s unable to allocate classNames: %s\n", getDescription_c(), strerror(errno));
      }
      if (strcmp(classNamesFile, "")) {
         std::ifstream *classNamesStream = new std::ifstream(classNamesFile);
         if (classNamesStream->fail()) {
            Fatal().printf(
                  "%s: unable to open classNamesFile \"%s\".\n",
                  getDescription_c(),
                  classNamesFile);
         }
         for (int k = 0; k < nf; k++) {
            // Need to clean this section up: handle too-long lines, premature eof, other issues
            char oneclass[1024];
            classNamesStream->getline(oneclass, 1024);
            classNames[k] = strdup(oneclass);
            if (classNames[k] == NULL) {
               ErrorLog().printf(
                     "%s unable to allocate class name %d from \"%s\": %s\n",
                     getDescription_c(),
                     k,
                     classNamesFile,
                     strerror(errno));
               exit(EXIT_FAILURE);
            }
         }
      }
      else {
         WarnLog().printf(
               "classNamesFile was not set in params file; Class names will be feature indices.\n");
         for (int k = 0; k < nf; k++) {
            std::stringstream classNameString("");
            classNameString << "Feature " << k;
            classNames[k] = strdup(classNameString.str().c_str());
            if (classNames[k] == NULL) {
               Fatal().printf(
                     "%s: unable to allocate className %d: %s\n",
                     getDescription_c(),
                     k,
                     strerror(errno));
            }
         }
      }
   }

   // If drawing montages, create montage directory and label files.
   // Under MPI, all process must call ensureDirExists() even though
   // only the root process interacts with the filesystem.
   if (drawMontage) {
      featurefieldwidth = (int)ceilf(log10f((float)(nf + 1)));

      // Make the heat map montage directory if it doesn't already exist
      status = ensureDirExists(parent->getCommunicator(), heatMapMontageDir);
      if (status != PV_SUCCESS) {
         Fatal().printf(
               "Error: Unable to make heat map montage directory \"%s\": %s\n",
               heatMapMontageDir,
               strerror(errno));
      }

      // Make the labels directory in heatMapMontageDir if it doesn't already exist
      std::stringstream labelsdirss("");
      labelsdirss << heatMapMontageDir << "/labels";
      char *labelsDir = strdup(labelsdirss.str().c_str());
      status          = ensureDirExists(parent->getCommunicator(), labelsDir);
      if (status != PV_SUCCESS) {
         Fatal().printf(
               "Error: Unable to make heat map montage labels directory: %s\n", strerror(errno));
      }
      free(labelsDir);

      // make the labels
      if (parent->columnId() == 0) {
         int nxGlobal = imageLayer->getLayerLoc()->nxGlobal;
         std::string originalLabelString("");
         originalLabelString += heatMapMontageDir;
         originalLabelString += "/labels/original.tif";
         status = drawTextIntoFile(
               originalLabelString.c_str(), "black", "white", "Original Image", nxGlobal);
         if (status != 0) {
            Fatal().printf(
                  "%s unable to create label file \"%s\".\n",
                  getDescription_c(),
                  originalLabelString.c_str());
         }

         std::string reconLabelString("");
         reconLabelString += heatMapMontageDir;
         reconLabelString += "/labels/reconstruction.tif";
         status = drawTextIntoFile(
               reconLabelString.c_str(), "black", "white", "Reconstruction", nxGlobal);
         if (status != 0) {
            Fatal().printf(
                  "%s unable to create label file \"%s\".\n",
                  getDescription_c(),
                  reconLabelString.c_str());
         }

         for (int idx = 0; idx < numDisplayedCategories; idx++) {
            int category = displayedCategories[idx];
            int f        = category - 1; // category is one-indexed; f is zero-indexed.
            char labelFilename[PV_PATH_MAX];
            int slen;
            slen = snprintf(
                  labelFilename,
                  PV_PATH_MAX,
                  "%s/labels/gray%0*d.tif",
                  heatMapMontageDir,
                  featurefieldwidth,
                  category);
            if (slen >= PV_PATH_MAX) {
               Fatal().printf(
                     "%s: file name for label %d is too long (%d characters versus %d).\n",
                     getDescription_c(),
                     category,
                     slen,
                     PV_PATH_MAX);
            }
            status = drawTextIntoFile(labelFilename, "white", "gray", classNames[f], nxGlobal);
            if (status != 0) {
               Fatal().printf(
                     "%s: unable to create label file \"%s\".\n",
                     getDescription_c(),
                     labelFilename);
            }
         }
      }
   }

   return status;
}

int LocalizationProbe::setOptimalMontage() {
   // Find the best number of rows and columns to use for the montage
   //   int const numCategories = displayCategoryIndexEnd - displayCategoryIndexStart + 1;
   assert(numDisplayedCategories > 0);
   int numRows[numDisplayedCategories];
   float totalSizeX[numDisplayedCategories];
   float totalSizeY[numDisplayedCategories];
   float aspectRatio[numDisplayedCategories];
   float ldgr[numDisplayedCategories]; // log of difference from golden ratio
   float loggoldenratio = std::log(0.5f * (1.0f + sqrtf(5.0f)));
   for (int numCol = 1; numCol <= numDisplayedCategories; numCol++) {
      int idx      = numCol - 1;
      numRows[idx] = (int)ceil((float)numDisplayedCategories / (float)numCol);
      totalSizeX[idx] =
            numCol * (imageLayer->getLayerLoc()->nxGlobal + 10); // +10 for spacing between images.
      totalSizeY[idx] = numRows[idx]
                        * (imageLayer->getLayerLoc()->nyGlobal + 64 + 10); // +64 for category label
      aspectRatio[idx] = (float)totalSizeX[idx] / (float)totalSizeY[idx];
      ldgr[idx]        = std::abs(std::log(aspectRatio[idx]) - loggoldenratio);
   }
   numMontageColumns = -1;
   float minldfgr    = std::numeric_limits<float>::infinity();
   for (int numCol = 1; numCol <= numDisplayedCategories; numCol++) {
      int idx = numCol - 1;
      if (ldgr[idx] < minldfgr) {
         minldfgr          = ldgr[idx];
         numMontageColumns = numCol;
      }
   }
   assert(numMontageColumns > 0);
   numMontageRows = numRows[numMontageColumns - 1];
   while ((numMontageColumns - 1) * numMontageRows >= numDisplayedCategories) {
      numMontageColumns--;
   }
   while ((numMontageRows - 1) * numMontageColumns >= numDisplayedCategories) {
      numMontageRows--;
   }
   if (numMontageRows < 2) {
      numMontageRows = 2;
   }
   return PV_SUCCESS;
}

char const *LocalizationProbe::getClassName(int k) {
   if (parent->columnId() != 0 || k < 0 || k >= targetLayer->getLayerLoc()->nf) {
      return NULL;
   }
   else {
      return classNames[k];
   }
}

int LocalizationProbe::allocateDataStructures() {
   int status = PV::LayerProbe::allocateDataStructures();
   if (status != PV_SUCCESS) {
      Fatal().printf("%s: allocateDataStructures failed.\n", getDescription_c());
   }
   detections.reserve(maxDetections);
   if (drawMontage) {
      assert(imageLayer);
      PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
      int const nx               = imageLoc->nx;
      int const ny               = imageLoc->ny;
      grayScaleImage             = (float *)calloc(nx * ny, sizeof(float));
      if (grayScaleImage == NULL) {
         Fatal().printf(
               "%s: unable to allocate memory for montage background image: %s\n",
               getDescription_c(),
               strerror(errno));
      }

      montageImageLocal = (unsigned char *)calloc(nx * ny * 3, sizeof(unsigned char));
      if (montageImageLocal == NULL) {
         Fatal().printf(
               "%s: unable to allocate memory for montage background image: %s\n",
               getDescription_c(),
               strerror(errno));
      }

      if (parent->columnId() == 0) {
         montageImageComm = (unsigned char *)calloc(nx * ny * 3, sizeof(unsigned char));
         if (montageImageComm == NULL) {
            Fatal().printf(
                  "%s: unable to allocate memory for MPI communication of heat map montage image: "
                  "%s\n",
                  getDescription_c(),
                  strerror(errno));
         }

         int const nxGlobal = imageLoc->nxGlobal;
         int const nyGlobal = imageLoc->nyGlobal;
         montageDimX        = (nxGlobal + 10) * (numMontageColumns + 2);
         montageDimY        = (nyGlobal + 64 + 10) * numMontageRows + 32;
         montageImage =
               (unsigned char *)calloc(montageDimX * montageDimY * 3, sizeof(unsigned char));
         if (montageImage == NULL) {
            Fatal().printf(
                  "%s: unable to allocate memory for heat map montage image: %s\n",
                  getDescription_c(),
                  strerror(errno));
         }

         int xStart = (2 * numMontageColumns + 1) * (nxGlobal + 10) / 2; // Integer division
         int yStart = 32 + 5;
         std::string originalFileName("");
         originalFileName += heatMapMontageDir;
         originalFileName += "/labels/original.tif";
         status = insertFileIntoMontage(
               originalFileName.c_str(), xStart, yStart, nxGlobal, 32 /*yExpectedSize*/);
         if (status != PV_SUCCESS) {
            Fatal().printf(
                  "%s: unable to place the \"original image\" label.\n", getDescription_c());
         }

         // same xStart.
         yStart += nyGlobal + 64 + 10; // yStart moves down one panel.
         std::string reconFileName("");
         reconFileName += heatMapMontageDir;
         reconFileName += "/labels/reconstruction.tif";
         status = insertFileIntoMontage(
               reconFileName.c_str(), xStart, yStart, nxGlobal, 32 /*yExpectedSize*/);
         if (status != PV_SUCCESS) {
            Fatal().printf(
                  "%s: unable to place the \"reconstruction\" label.\n", getDescription_c());
         }

         for (int idx = 0; idx < numDisplayedCategories; idx++) {
            int category   = displayedCategories[idx];
            int f          = category - 1; // category is one-indexed; f is zero-indexed;
            int montageCol = kxPos(idx, numMontageColumns, numMontageRows, 1);
            int montageRow = kyPos(idx, numMontageColumns, numMontageRows, 1);
            int xStart     = montageCol * (nxGlobal + 10) + 5;
            int yStart     = montageRow * (nyGlobal + 64 + 10) + 5;
            char filename[PV_PATH_MAX];
            int slen = snprintf(
                  filename,
                  PV_PATH_MAX,
                  "%s/labels/gray%0*d.tif",
                  heatMapMontageDir,
                  featurefieldwidth,
                  category);
            if (slen >= PV_PATH_MAX) {
               Fatal().printf(
                     "%s: path to label file for label %d is too large.\n",
                     getDescription_c(),
                     category);
            }
            status =
                  insertFileIntoMontage(filename, xStart, yStart, nxGlobal, 32 /*yExpectedSize*/);
            if (status != PV_SUCCESS) {
               Fatal().printf(
                     "%s: unable to place the label for feature %d.\n", getDescription_c(), f);
            }
         }
      }
   }
   if (getTextOutputFlag()) {
      if (outputStream) {
         PVLayerLoc const *targetLoc = targetLayer->getLayerLoc();
         outputStream->printf(
               "%s, %dx%d with %d categories.\n",
               targetLayer->getDescription_c(),
               targetLoc->nxGlobal,
               targetLoc->nyGlobal,
               targetLoc->nf);
         PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
         outputStream->printf(
               "Image \"%s\", %dx%d with %d features.\n",
               imageLayer->getName(),
               imageLoc->nxGlobal,
               imageLoc->nyGlobal,
               imageLoc->nf);
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::drawTextOnMontage(
      char const *backgroundColor,
      char const *textColor,
      char const *labelText,
      int xOffset,
      int yOffset,
      int width,
      int height) {
   assert(parent->columnId() == 0);
   char *tempfile = strdup("/tmp/Localization_XXXXXX.tif");
   if (tempfile == NULL) {
      Fatal().printf(
            "%s: drawTextOnMontage failed to create temporary file for text\n", getDescription_c());
   }
   int tempfd = mkstemps(tempfile, 4 /*suffixlen*/);
   if (tempfd < 0) {
      Fatal().printf(
            "%s: drawTextOnMontage failed to create temporary file for writing\n",
            getDescription_c());
   }
   int status = close(tempfd); // mkstemps opens the file to avoid race between finding unused
   // filename and opening it, but we don't need the file descriptor.
   if (status != 0) {
      Fatal().printf(
            "%s: drawTextOnMontage failed to close temporory file %s: %s\n",
            getDescription_c(),
            tempfile,
            strerror(errno));
   }
   status = drawTextIntoFile(tempfile, backgroundColor, textColor, labelText, width, height);
   if (status == 0) {
      status = insertFileIntoMontage(tempfile, xOffset, yOffset, width, height);
   }
   status = unlink(tempfile);
   if (status != 0) {
      Fatal().printf(
            "%s: drawTextOnMontage failed to delete temporary file %s: %s\n",
            getDescription_c(),
            tempfile,
            strerror(errno));
   }
   free(tempfile);
   return status;
}

int LocalizationProbe::drawTextIntoFile(
      char const *labelFilename,
      char const *backgroundColor,
      char const *textColor,
      char const *labelText,
      int width,
      int height) {
   assert(parent->columnId() == 0);
   std::stringstream convertCmd("");
   convertCmd << "convert -depth 8 -background \"" << backgroundColor << "\" -fill \"" << textColor
              << "\" -size " << width << "x" << height << " -pointsize 18 -gravity center label:\""
              << labelText << "\" \"" << labelFilename << "\"";
   int status = system(convertCmd.str().c_str());
   if (status != 0) {
      Fatal().printf(
            "%s: unable to create label file \"%s\": ImageMagick convert returned %d.\n",
            getDescription_c(),
            labelFilename,
            WEXITSTATUS(status));
   }
   return status;
}

int LocalizationProbe::insertFileIntoMontage(
      char const *labelFilename,
      int xOffset,
      int yOffset,
      int xExpectedSize,
      int yExpectedSize) {
   assert(parent->columnId() == 0);
   PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
   int const nx               = imageLoc->nx;
   int const ny               = imageLoc->ny;
   int const nf               = imageLoc->nf;
   GDALDataset *dataset       = (GDALDataset *)GDALOpen(labelFilename, GA_ReadOnly);
   if (dataset == NULL) {
      ErrorLog().printf(
            "%s: unable to open label file \"%s\" for reading.\n",
            getDescription_c(),
            labelFilename);
      return PV_FAILURE;
   }
   int xLabelSize = dataset->GetRasterXSize();
   int yLabelSize = dataset->GetRasterYSize();
   int labelBands = dataset->GetRasterCount();
   if (xLabelSize != xExpectedSize || yLabelSize != yExpectedSize) {
      Fatal().printf(
            "%s: label files \"%s\" has dimensions %dx%d (expected %dx%d)\n",
            getDescription_c(),
            labelFilename,
            xLabelSize,
            yLabelSize,
            xExpectedSize,
            yExpectedSize);
   }
   // same xStart.
   int offsetIdx = kIndex(xOffset, yOffset, 0, montageDimX, montageDimY, 3);
   dataset->RasterIO(
         GF_Read,
         0,
         0,
         xLabelSize,
         yLabelSize,
         &montageImage[offsetIdx],
         xLabelSize,
         yLabelSize,
         GDT_Byte,
         3 /*number of bands*/,
         NULL,
         3 /*x-stride*/,
         3 * montageDimX /*y-stride*/,
         1 /*band stride*/);
   GDALClose(dataset);
   return PV_SUCCESS;
}

int LocalizationProbe::insertImageIntoMontage(
      int xStart,
      int yStart,
      float const *sourceData,
      PVLayerLoc const *loc,
      bool extended) {
   float minValue            = std::numeric_limits<float>::infinity();
   float maxValue            = -std::numeric_limits<float>::infinity();
   int const nx              = loc->nx;
   int const ny              = loc->ny;
   int const nf              = loc->nf;
   PVHalo const *halo        = &loc->halo;
   int const numImageNeurons = nx * ny * nf;
   if (extended) {
      for (int k = 0; k < numImageNeurons; k++) {
         int const kExt = kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         float a        = sourceData[kExt];
         minValue       = a < minValue ? a : minValue;
         maxValue       = a > maxValue ? a : maxValue;
      }
   }
   else {
      for (int k = 0; k < numImageNeurons; k++) {
         float a  = sourceData[k];
         minValue = a < minValue ? a : minValue;
         maxValue = a > maxValue ? a : maxValue;
      }
   }
   MPI_Allreduce(
         MPI_IN_PLACE, &minValue, 1, MPI_FLOAT, MPI_MIN, parent->getCommunicator()->communicator());
   MPI_Allreduce(
         MPI_IN_PLACE, &maxValue, 1, MPI_FLOAT, MPI_MAX, parent->getCommunicator()->communicator());
   if (minValue == maxValue) {
      for (int y = 0; y < ny; y++) {
         int lineStart = kIndex(0, y, 0, nx, ny, nf);
         memset(&montageImageLocal[lineStart], 127, (size_t)(nx * nf));
      }
   }
   else {
      float scale = (float)1 / (maxValue - minValue);
      float shift = minValue;
      for (int k = 0; k < numImageNeurons; k++) {
         int const kx = kxPos(k, nx, ny, nf);
         int const ky = kyPos(k, nx, ny, nf);
         int const kf = featureIndex(k, nx, ny, nf);
         int const kImageExt =
               kIndexExtended(k, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         float a              = sourceData[kImageExt];
         a                    = nearbyintf(255 * (a - shift) * scale);
         unsigned char aChar  = (unsigned char)(int)a;
         montageImageLocal[k] = aChar;
      }
   }
   if (parent->columnId() != 0) {
      MPI_Send(
            montageImageLocal,
            nx * ny * 3,
            MPI_UNSIGNED_CHAR,
            0,
            111,
            parent->getCommunicator()->communicator());
   }
   else {
      for (int rank = 0; rank < parent->numberOfColumns(); rank++) {
         if (rank != 0) {
            MPI_Recv(
                  montageImageComm,
                  nx * ny * 3,
                  MPI_UNSIGNED_CHAR,
                  rank,
                  111,
                  parent->getCommunicator()->communicator(),
                  MPI_STATUS_IGNORE);
         }
         else {
            memcpy(montageImageComm, montageImageLocal, nx * ny * 3);
         }
         int const numCommRows = parent->getCommunicator()->numCommRows();
         int const numCommCols = parent->getCommunicator()->numCommColumns();
         int const commRow     = rowFromRank(rank, numCommRows, numCommCols);
         int const commCol     = columnFromRank(rank, numCommRows, numCommCols);
         for (int y = 0; y < ny; y++) {
            int destIdx = kIndex(
                  xStart + commCol * nx, yStart + commRow * ny + y, 0, montageDimX, montageDimY, 3);
            int srcIdx = kIndex(0, y, 0, nx, ny, 3);
            memcpy(&montageImage[destIdx], &montageImageComm[srcIdx], nx * 3);
         }
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::setOutputFilenameBase(char const *fn) {
   free(outputFilenameBase);
   int status = PV_SUCCESS;
   std::string fnString(fn);
   size_t lastSlash = fnString.rfind("/");
   if (lastSlash != std::string::npos) {
      fnString.erase(0, lastSlash + 1);
   }

   size_t lastDot = fnString.rfind(".");
   if (lastDot != std::string::npos) {
      fnString.erase(lastDot);
   }
   if (fnString.empty()) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s setOutputFilenameBase: string \"%s\" is empty after removing directory and "
               "extension.\n",
               getDescription_c(),
               fn);
      }
      status             = PV_FAILURE;
      outputFilenameBase = NULL;
   }
   else {
      outputFilenameBase = strdup(fnString.c_str());
      if (outputFilenameBase == NULL) {
         Fatal().printf(
               "%s setOutputFilenameBase failed for filename \"%s\": %s\n",
               getDescription_c(),
               fn,
               strerror(errno));
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
      if (timed >= nextOutputTime) {
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
   int const N           = targetLayer->getNumNeurons();
   PVLayerLoc const *loc = targetLayer->getLayerLoc();
   PVHalo const *halo    = &loc->halo;
   float *targetRes      = (float *)malloc(sizeof(float) * N);
   if (targetRes == NULL) {
      Fatal().printf("%s unable to allocate buffer for calcValues\n", getDescription_c());
   }
   for (int n = 0; n < N; n++) {
      int nExt =
            kIndexExtended(n, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
      targetRes[n] = targetLayer->getLayerData()[nExt];
   }
   float minDetectionThreshold = std::numeric_limits<float>::infinity();
   for (int k = 0; k < numDisplayedCategories; k++) {
      float a               = detectionThreshold[k];
      minDetectionThreshold = a < minDetectionThreshold ? a : minDetectionThreshold;
   }
   detections.clear();
   unsigned detectionIndex = 0;

   while (detectionIndex < maxDetections) {
      int winningFeature, winningIndex, xLocation, yLocation;
      float activity;
      findMaxLocation(
            &winningFeature, &winningIndex, &xLocation, &yLocation, &activity, targetRes, loc);
      assert(winningFeature == displayedCategories[winningIndex] - 1);
      if (winningFeature >= 0 && activity >= minDetectionThreshold) {
         assert(xLocation >= 0 && yLocation >= 0);
         int boundingBox[4];
         findBoundingBox(
               winningFeature, winningIndex, xLocation, yLocation, targetRes, loc, boundingBox);
         double score  = 0.0;
         int numpixels = 0;
         for (int y = boundingBox[2]; y < boundingBox[3]; y++) {
            for (int x = boundingBox[0]; x < boundingBox[1]; x++) {
               int xLoc = x - loc->kx0;
               int yLoc = y - loc->ky0;
               if (xLoc >= 0 && xLoc < loc->nx && yLoc >= 0 && yLoc < loc->ny) {
                  int n    = kIndex(xLoc, yLoc, winningFeature, loc->nx, loc->ny, loc->nf);
                  int nExt = kIndexExtended(
                        n, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
                  targetRes[n] = 0.0f;
                  score += (double)targetLayer->getLayerData()[nExt];
                  numpixels++;
               }
            }
         }
         MPI_Allreduce(
               MPI_IN_PLACE,
               &score,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               parent->getCommunicator()->communicator());
         MPI_Allreduce(
               MPI_IN_PLACE,
               &numpixels,
               1,
               MPI_INT,
               MPI_SUM,
               parent->getCommunicator()->communicator());
         score = score / numpixels;
         if (boundingBox[1] - boundingBox[0] >= minBoundingBoxWidth
             && boundingBox[3] - boundingBox[2] >= minBoundingBoxHeight) {
            LocalizationData newDetection;
            newDetection.feature        = winningFeature;
            newDetection.displayedIndex = winningIndex;
            newDetection.left           = (int)nearbyint(boundingBox[0] * imageDilationX);
            newDetection.right          = (int)nearbyint(boundingBox[1] * imageDilationX);
            newDetection.top            = (int)nearbyint(boundingBox[2] * imageDilationY);
            newDetection.bottom         = (int)nearbyint(boundingBox[3] * imageDilationY);
            newDetection.score          = score;
            detections.push_back(newDetection);
            detectionIndex++;
         }
      }
      else {
         assert(!(winningFeature >= 0 && activity >= minDetectionThreshold));
         break;
      }
   }

   assert(getNumValues() == 1);
   double *values = getValuesBuffer();
   *values        = detectionIndex;
   free(targetRes);
   return PV_SUCCESS;
}

int LocalizationProbe::findMaxLocation(
      int *winningFeature,
      int *winningIndex,
      int *xLocation,
      int *yLocation,
      float *maxActivity,
      float *buffer,
      PVLayerLoc const *loc) {
   int const nxy   = loc->nx * loc->ny;
   int const nf    = loc->nf;
   int maxLocation = -1;
   float maxVal    = -std::numeric_limits<float>::infinity();
   for (int n = 0; n < nxy; n++) {
      for (int idx = 0; idx < numDisplayedCategories; idx++) {
         int const category = displayedCategories[idx];
         int const f        = category - 1; // category is 1-indexed; f is zero-indexed.
         float const a      = buffer[n * nf + f];
         if (a > maxVal) {
            maxVal      = a;
            maxLocation = n * nf + f;
         }
      }
   }

   MPI_Comm comm = parent->getCommunicator()->communicator();

   float maxValGlobal;
   MPI_Allreduce(&maxVal, &maxValGlobal, 1, MPI_FLOAT, MPI_MAX, comm);

   int maxLocGlobal;
   int const nxGlobal = loc->nxGlobal;
   int const nyGlobal = loc->nyGlobal;
   if (maxValGlobal == maxVal) {
      maxLocGlobal = globalIndexFromLocal(maxLocation, *loc);
   }
   else {
      maxLocGlobal =
            nxGlobal * nyGlobal * nf; // should be the same as targetLayer's getNumGlobalNeurons
   }
   MPI_Allreduce(MPI_IN_PLACE, &maxLocGlobal, 1, MPI_INT, MPI_MIN, comm);
   if (maxLocGlobal >= 0) {
      *winningFeature = featureIndex(maxLocGlobal, nxGlobal, nyGlobal, nf);
      *xLocation      = kxPos(maxLocGlobal, nxGlobal, nyGlobal, nf);
      *yLocation      = kyPos(maxLocGlobal, nxGlobal, nyGlobal, nf);
      *winningIndex   = -1;
      for (int k = 0; k < numDisplayedCategories; k++) {
         int category = displayedCategories[k];
         int f        = category - 1;
         if (*winningFeature == f) {
            *winningIndex = k;
            break;
         }
      }
   }
   else {
      *winningFeature = -1;
      *winningIndex   = -1;
      *xLocation      = -1;
      *yLocation      = -1;
   }
   *maxActivity = maxValGlobal;
   return PV_SUCCESS;
}

int LocalizationProbe::findBoundingBox(
      int winningFeature,
      int winningIndex,
      int xLocation,
      int yLocation,
      float const *buffer,
      PVLayerLoc const *loc,
      int *boundingBox) {
   if (winningFeature >= 0 && xLocation >= 0 && yLocation >= 0) {
      bool locationThisProcess =
            (xLocation >= loc->kx0 && xLocation < loc->kx0 + loc->nx && yLocation >= loc->ky0
             && yLocation < loc->ky0 + loc->ny);
      int lt      = xLocation - loc->kx0;
      int rt      = xLocation - loc->kx0;
      int dn      = yLocation - loc->ky0;
      int up      = yLocation - loc->ky0;
      int const N = targetLayer->getNumNeurons();
      for (int n = winningFeature; n < N; n += loc->nf) {
         float const a = buffer[n];
         if (a > detectionThreshold[winningIndex]) {
            int x = kxPos(n, loc->nx, loc->ny, loc->nf);
            int y = kyPos(n, loc->nx, loc->ny, loc->nf);
            if (x < lt) {
               lt = x;
            }
            if (x > rt) {
               rt = x;
            }
            if (y < up) {
               up = y;
            }
            if (y > dn) {
               dn = y;
            }
         }
      }

      // Convert back to global coordinates
      lt += loc->kx0;
      rt += loc->kx0;
      up += loc->ky0;
      dn += loc->ky0;

      // Now we need the maximum of the rt and dn over all processes,
      // and the minimum of the lt and up over all processes.
      // Change the sign of lt and up, then do MPI_MAX, then change the sign back.
      boundingBox[0] = -lt;
      boundingBox[1] = rt + 1;
      boundingBox[2] = -up;
      boundingBox[3] = dn + 1;
      MPI_Comm comm  = parent->getCommunicator()->communicator();
      MPI_Allreduce(MPI_IN_PLACE, boundingBox, 4, MPI_INT, MPI_MAX, comm);
      boundingBox[0] = -boundingBox[0];
      boundingBox[2] = -boundingBox[2];
   }
   else {
      for (int k = 0; k < 4; k++) {
         boundingBox[k] = -1;
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::outputStateWrapper(double timef, double dt) {
   int status  = PV_SUCCESS;
   mStepNumber = (long int)std::nearbyintf(timef / dt);
   if ((getTextOutputFlag() || drawMontage) && needUpdate(timef, dt)) {
      status = outputState(timef);
   }
   return status;
}

int LocalizationProbe::outputState(double timevalue) {
   int status = getValues(timevalue); // all processes must call getValues in parallel.
   if (getTextOutputFlag() && outputStream) {
      assert(parent->columnId() == 0);
      size_t numDetected = detections.size();
      assert(numDetected < maxDetections);
      if (numDetected == 0) {
         outputStream->printf("Time %f, no detections.\n", timevalue);
      }
      for (size_t d = 0; d < numDetected; d++) {
         LocalizationData const *thisDetection = &detections.at(d);
         int winningFeature                    = thisDetection->feature;
         assert(winningFeature >= 0 && winningFeature < targetLayer->getLayerLoc()->nf);
         double score = thisDetection->score;
         outputStream->printf(
               "Time %f, \"%s\", score %f, bounding box x=[%d,%d), y=[%d,%d)\n",
               timevalue,
               getClassName(winningFeature),
               score,
               thisDetection->left,
               thisDetection->right,
               thisDetection->top,
               thisDetection->bottom);
      }
   }
   if (drawMontage) {
      status = makeMontage();
   }
   return status;
}

int LocalizationProbe::makeMontage() {
   assert(drawMontage);
   assert(numMontageRows > 0 && numMontageColumns > 0);
   assert((parent->columnId() == 0) == (montageImage != NULL));
   PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
   PVHalo const *halo         = &imageLoc->halo;
   int const nx               = imageLoc->nx;
   int const ny               = imageLoc->ny;
   int const nf               = imageLoc->nf;
   int const N                = nx * ny;

   // create grayscale version of image layer for background of heat maps.
   makeGrayScaleImage();

   // for each displayed category, copy grayScaleImage to the relevant part of the montage, and
   // impose the upsampled version
   // of the target layer onto it.
   drawHeatMaps();

   drawOriginalAndReconstructed();

   if (parent->columnId() != 0) {
      return PV_SUCCESS;
   }

   // Draw bounding boxes
   if (boundingBoxLineWidth > 0) {
      assert(getNumValues() == 1);
      for (size_t d = 0; d < detections.size(); d++) {
         LocalizationData const *thisBoundingBox = &detections.at(d);
         int winningFeature                      = thisBoundingBox->feature;
         if (winningFeature < 0) {
            continue;
         }
         int left         = thisBoundingBox->left;
         int right        = thisBoundingBox->right;
         int top          = thisBoundingBox->top;
         int bottom       = thisBoundingBox->bottom;
         int winningIndex = thisBoundingBox->displayedIndex;
         assert(winningIndex >= 0);
         int montageColumn     = kxPos(winningIndex, numMontageColumns, numMontageRows, 1);
         int montageRow        = kyPos(winningIndex, numMontageColumns, numMontageRows, 1);
         int xStartInMontage   = montageColumn * (imageLoc->nxGlobal + 10) + 5 + left;
         int yStartInMontage   = montageRow * (imageLoc->nyGlobal + 64 + 10) + 5 + 64 + top;
         int width             = (int)(right - left);
         int height            = (int)(bottom - top);
         char const bbColor[3] = {'\377', '\0', '\0'}; // red
         for (int y = 0; y < boundingBoxLineWidth; y++) {
            int lineStart =
                  kIndex(xStartInMontage, yStartInMontage + y, 0, montageDimX, montageDimY, 3);
            for (int k = 0; k < 3 * width; k++) {
               int f = featureIndex(k, imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart + k] = bbColor[f];
            }
         }
         for (int y = boundingBoxLineWidth; y < height - boundingBoxLineWidth; y++) {
            int lineStart =
                  kIndex(xStartInMontage, yStartInMontage + y, 0, montageDimX, montageDimY, 3);
            for (int k = 0; k < 3 * boundingBoxLineWidth; k++) {
               int f = featureIndex(k, imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart + k] = bbColor[f];
            }
            lineStart = kIndex(
                  xStartInMontage + width - boundingBoxLineWidth,
                  yStartInMontage + y,
                  0,
                  montageDimX,
                  montageDimY,
                  3);
            for (int k = 0; k < 3 * boundingBoxLineWidth; k++) {
               int f = featureIndex(k, imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart + k] = bbColor[f];
            }
         }
         for (int y = height - boundingBoxLineWidth; y < height; y++) {
            int lineStart =
                  kIndex(xStartInMontage, yStartInMontage + y, 0, montageDimX, montageDimY, 3);
            for (int k = 0; k < 3 * width; k++) {
               int f = featureIndex(k, imageLoc->nxGlobal, imageLoc->nyGlobal, 3);
               montageImage[lineStart + k] = bbColor[f];
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

int LocalizationProbe::makeGrayScaleImage() {
   assert(grayScaleImage);
   PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
   PVHalo const *halo         = &imageLoc->halo;
   int const nx               = imageLoc->nx;
   int const ny               = imageLoc->ny;
   int const nf               = imageLoc->nf;
   int const N                = nx * ny;
   float const *imageActivity = imageLayer->getLayerData();
   for (int kxy = 0; kxy < N; kxy++) {
      float a   = (float)0;
      int nExt0 = kIndexExtended(kxy * nf, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      float const *imageDataXY = &imageActivity[nExt0];
      for (int f = 0; f < nf; f++) {
         a += imageDataXY[f];
      }
      grayScaleImage[kxy] = a / (float)nf;
   }
   float minValue = std::numeric_limits<float>::infinity();
   float maxValue = -std::numeric_limits<float>::infinity();
   for (int kxy = 0; kxy < N; kxy++) {
      float a = grayScaleImage[kxy];
      if (a < minValue) {
         minValue = a;
      }
      if (a > maxValue) {
         maxValue = a;
      }
   }
   MPI_Allreduce(
         MPI_IN_PLACE, &minValue, 1, MPI_FLOAT, MPI_MIN, parent->getCommunicator()->communicator());
   MPI_Allreduce(
         MPI_IN_PLACE, &maxValue, 1, MPI_FLOAT, MPI_MAX, parent->getCommunicator()->communicator());
   // Scale grayScaleImage to be between 0 and 1.
   // If maxValue==minValue, make grayScaleImage have a constant 0.5.
   if (maxValue == minValue) {
      for (int kxy = 0; kxy < N; kxy++) {
         grayScaleImage[kxy] = 0.5;
      }
   }
   else {
      float scaleFactor = (float)1.0 / (maxValue - minValue);
      for (int kxy = 0; kxy < N; kxy++) {
         float *pixloc = &grayScaleImage[kxy];
         *pixloc       = scaleFactor * (*pixloc - minValue);
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::drawHeatMaps() {
   float thresholdColor[] = {
         0.5f, 0.5f, 0.5f}; // rgb color of heat map when activity is at or below detectionThreshold
   float heatMapColor[] = {
         0.0f, 1.0f, 0.0f}; // rgb color of heat map when activity is at or above heatMapMaximum

   PVLayerLoc const *targetLoc = targetLayer->getLayerLoc();
   PVHalo const *targetHalo    = &targetLoc->halo;
   size_t numDetections        = detections.size();
   int winningFeature[numDetections];
   int winningIndex[numDetections];
   double boxConfidence[numDetections];
   for (int d = 0; d < numDetections; d++) {
      LocalizationData const *box = &detections.at(d);
      boxConfidence[d]            = box->score;
      winningFeature[d]           = box->feature;
      winningIndex[d]             = box->displayedIndex;
   }

   double maxConfByCategory[targetLoc->nf];
   for (int f = 0; f < targetLoc->nf; f++) {
      maxConfByCategory[f] = -std::numeric_limits<float>::infinity();
   }
   for (size_t d = 0; d < numDetections; d++) {
      int f                = winningFeature[d];
      double a             = boxConfidence[d];
      double m             = maxConfByCategory[f];
      maxConfByCategory[f] = a > m ? a : m;
   }

   PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
   int const nx               = imageLoc->nx;
   int const ny               = imageLoc->ny;
   for (int idx = 0; idx < numDisplayedCategories; idx++) {
      int category = displayedCategories[idx];
      int f        = category - 1; // category is 1-indexed; f is zero-indexed.
      for (int y = 0; y < ny; y++) {
         for (int x = 0; x < nx; x++) {
            float backgroundLevel = grayScaleImage[x + nx * y];
            int xTarget           = (int)((double)x / imageDilationX);
            int yTarget           = (int)((double)y / imageDilationY);
            int targetIdx =
                  kIndex(xTarget, yTarget, f, targetLoc->nx, targetLoc->ny, targetLoc->nf);
            int targetIdxExt = kIndexExtended(
                  targetIdx,
                  targetLoc->nx,
                  targetLoc->ny,
                  targetLoc->nf,
                  targetHalo->lt,
                  targetHalo->rt,
                  targetHalo->dn,
                  targetHalo->up);
            float heatMapLevel = targetLayer->getLayerData()[targetIdxExt];

            // Only show values if they are the highest category
            for (int idx2 = 0; idx2 < numDisplayedCategories; idx2++) {
               int f2 = displayedCategories[idx2] - 1;
               heatMapLevel *=
                     (float)(heatMapLevel >= targetLayer->getLayerData()[targetIdxExt - f + f2]);
            }

            float detThresh  = detectionThreshold[idx];
            float heatMapMax = heatMapMaximum[idx];
            heatMapLevel     = (heatMapLevel - detThresh) / (heatMapMax - detThresh);
            heatMapLevel     = heatMapLevel < (float)0 ? (float)0 : heatMapLevel > (float)1
                                                                      ? (float)1
                                                                      : heatMapLevel;
            int montageIdx = kIndex(x, y, 0, nx, ny, 3);
            for (int rgb = 0; rgb < 3; rgb++) {
               float h =
                     heatMapLevel * heatMapColor[rgb] + (1 - heatMapLevel) * thresholdColor[rgb];
               float g = imageBlendCoeff * backgroundLevel + (1 - imageBlendCoeff) * h;
               assert(g >= (float)-0.001 && g <= (float)1.001);
               g                   = nearbyintf(255 * g);
               unsigned char gchar = (unsigned char)g;
               assert(montageIdx >= 0 && montageIdx + rgb < nx * ny * 3);
               montageImageLocal[montageIdx + rgb] = gchar;
            }
         }
      }
      if (parent->columnId() != 0) {
         MPI_Send(
               montageImageLocal,
               nx * ny * 3,
               MPI_UNSIGNED_CHAR,
               0,
               111,
               parent->getCommunicator()->communicator());
      }
      else {
         int montageCol        = idx % numMontageColumns;
         int montageRow        = (idx - montageCol) / numMontageColumns; // Integer arithmetic
         int xStartInMontage   = (imageLoc->nxGlobal + 10) * montageCol + 5;
         int yStartInMontage   = (imageLoc->nyGlobal + 64 + 10) * montageRow + 64 + 5;
         int const numCommRows = parent->getCommunicator()->numCommRows();
         int const numCommCols = parent->getCommunicator()->numCommColumns();
         for (int rank = 0; rank < parent->numberOfColumns(); rank++) {
            if (rank == 0) {
               memcpy(montageImageComm, montageImageLocal, nx * ny * 3);
            }
            else {
               MPI_Recv(
                     montageImageComm,
                     nx * ny * 3,
                     MPI_UNSIGNED_CHAR,
                     rank,
                     111,
                     parent->getCommunicator()->communicator(),
                     MPI_STATUS_IGNORE);
            }
            int const commRow = rowFromRank(rank, numCommRows, numCommCols);
            int const commCol = columnFromRank(rank, numCommRows, numCommCols);
            for (int y = 0; y < ny; y++) {
               int destIdx = kIndex(
                     xStartInMontage + commCol * nx,
                     yStartInMontage + commRow * ny + y,
                     0,
                     montageDimX,
                     montageDimY,
                     3);
               int srcIdx = kIndex(0, y, 0, nx, ny, 3);
               memcpy(&montageImage[destIdx], &montageImageComm[srcIdx], nx * 3);
            }
         }

         // Draw confidences
         char confidenceText[16];
         if (maxConfByCategory[f] > 0.0) {
            int slen = snprintf(confidenceText, 16, "%.1f", 100 * maxConfByCategory[f]);
            if (slen >= 16) {
               Fatal().printf(
                     "Formatted text for confidence %f of category %d is too long.\n",
                     maxConfByCategory[idx],
                     f);
            }
         }
         else {
            strncpy(confidenceText, "-", 2);
         }
         drawTextOnMontage(
               "white",
               "gray",
               confidenceText,
               xStartInMontage,
               yStartInMontage - 32,
               imageLayer->getLayerLoc()->nxGlobal,
               32);
      }
   }
   return PV_SUCCESS;
}

int LocalizationProbe::drawOriginalAndReconstructed() {
   PVLayerLoc const *imageLoc = imageLayer->getLayerLoc();
   // Draw original image
   int xStart = 5 + (2 * numMontageColumns + 1) * (imageLoc->nxGlobal + 10) / 2;
   int yStart = 5 + 64;
   insertImageIntoMontage(xStart, yStart, imageLayer->getLayerData(), imageLoc, true /*extended*/);

   // Draw reconstructed image
   // same xStart, yStart is down one row.
   // I should check that reconLayer and imageLayer have the same dimensions
   yStart += imageLoc->nyGlobal + 64 + 10;
   insertImageIntoMontage(
         xStart, yStart, reconLayer->getLayerData(), reconLayer->getLayerLoc(), true /*extended*/);
}

int LocalizationProbe::drawProgressInformation() {
   std::stringstream progress("");
   double elapsed      = parent->simulationTime() - parent->getStartTime();
   double finishTime   = parent->getStopTime() - parent->getStartTime();
   bool isLastTimeStep = elapsed >= finishTime - parent->getDeltaTime() / 2;
   // if (!isLastTimeStep) {
   int percentage = (int)nearbyintf(100.0 * elapsed / finishTime);
   progress << "t = " << elapsed << "/" << finishTime << " (" << percentage << "%%)";
   //   progress << "t = " << elapsed << ", finish time = " << finishTime << " (" << percentage <<
   //   "%%)";
   //}
   // else {
   //   progress << "t = " << elapsed << ", completed";
   //}
   drawTextOnMontage(
         "black", "white", progress.str().c_str(), 0, montageDimY - 32, montageDimX, 32);
}

int LocalizationProbe::writeMontage() {
   std::stringstream montagePathSStream("");
   montagePathSStream << heatMapMontageDir << "/" << outputFilenameBase << "_" << mStepNumber;
   bool isLastTimeStep =
         parent->simulationTime() >= parent->getStopTime() - parent->getDeltaTime() / 2;
   if (isLastTimeStep) {
      montagePathSStream << "_final";
   }
   montagePathSStream << ".tif";
   char *montagePath =
         strdup(montagePathSStream.str().c_str()); // not sure why I have to strdup this
   if (montagePath == NULL) {
      Fatal().printf("%s: unable to create montagePath\n", getDescription_c());
   }
   GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
   if (driver == NULL) {
      Fatal().printf("GetGDALDriverManager()->GetDriverByName(\"GTiff\") failed.");
   }
   GDALDataset *dataset =
         driver->Create(montagePath, montageDimX, montageDimY, 3 /*numBands*/, GDT_Byte, NULL);
   if (dataset == NULL) {
      Fatal().printf("GDAL failed to open file \"%s\"\n", montagePath);
   }
   free(montagePath);
   dataset->RasterIO(
         GF_Write,
         0,
         0,
         montageDimX,
         montageDimY,
         montageImage,
         montageDimX,
         montageDimY,
         GDT_Byte,
         3 /*numBands*/,
         NULL,
         3 /*x-stride*/,
         3 * montageDimX /*y-stride*/,
         1 /*band-stride*/);
   GDALClose(dataset);
   return PV_SUCCESS;
}

LocalizationProbe::~LocalizationProbe() {
   free(displayedCategories);
   free(imageLayerName);
   free(reconLayerName);
   free(detectionThreshold);
   free(classNamesFile);
   free(heatMapMaximum);
   free(heatMapMontageDir);
   free(displayCommand);
   free(outputFilenameBase);
   free(grayScaleImage);
   free(montageImage);
   free(montageImageLocal);
   free(montageImageComm);
}

PV::BaseObject *createLocalizationProbe(char const *name, PV::HyPerCol *hc) {
   return hc ? new LocalizationProbe(name, hc) : NULL;
}
