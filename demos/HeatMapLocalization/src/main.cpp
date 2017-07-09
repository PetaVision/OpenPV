#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <list>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <sys/wait.h>

#include "BBFindConfRemapLayer.hpp"
#include "BBFindConfRemapProbe.hpp"
#include "ConvertFromTable.hpp"
#include "LocalizationBBFindProbe.hpp"
#include "LocalizationProbe.hpp"
#include "columns/Communicator.hpp"
#include "columns/PV_Init.hpp"
#include "layers/ImageFromMemoryBuffer.hpp"
#include "utils/PVAlloc.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#define TEXTFILEBUFFERSIZE 1024

std::string getImageFileName(PV::Communicator *icComm);
int setImageLayerMemoryBuffer(
      PV::Communicator *icComm,
      char const *imageFile,
      PV::ImageFromMemoryBuffer *imageLayer,
      uint8_t **imageBufferPtr,
      size_t *imageBufferSizePtr);
int runWithHarness(PV::HyPerCol *hc, int frameInterval);
int runWithoutHarness(PV::HyPerCol *hc);

class FrameServer {
  public:
   FrameServer() {
      mTmpDir = strdup("/tmp/OpenPVFrameServer-XXXXXX");
      if (mkdtemp(mTmpDir) == NULL) {
         Fatal() << "FrameServer unable to create temporary directory: " << strerror(errno);
         pvExitFailure("");
      }
      mListOfFrames.clear();
   }

   void setFrameRate(int fr) { mFrameRate = fr; }

   virtual ~FrameServer() {
      clearFrames();
      rmdir(mTmpDir);
      free(mTmpDir);
   }

   size_t getNumFrames() const { return mListOfFrames.size(); }

   char const *retrieveFrame() {
      if (mCurrentFrame == mListOfFrames.end()) {
         return nullptr;
      }
      std::string str = *mCurrentFrame;
      char const *s   = (*mCurrentFrame).c_str();
      mCurrentFrame++;
      return s;
   }

   void rewind() { mCurrentFrame = mListOfFrames.begin(); }

   void feedVideoToDragonsGapingMaw(char const *path) {
      pvAssert(mTmpDir);
      std::string cmdString("avconv -threads auto -r ");
      cmdString += std::to_string(mFrameRate);
      cmdString += " -i ";
      cmdString += path;
      cmdString += " -r 1 ";
      cmdString += mTmpDir;
      cmdString += "/";
      cmdString += mFilenamePrefix;
      cmdString += "%0";
      cmdString += std::to_string(strlen(mFilenamePattern));
      cmdString += "d";
      cmdString += mFilenameSuffix;
      int cmdStatus = system(cmdString.c_str());
      if (cmdStatus) {
         Fatal() << "FrameServer: command \"" << cmdString << "\" returned "
                 << WEXITSTATUS(cmdStatus) << "." << std::endl;
         exit(EXIT_FAILURE);
      }
      DIR *dirPtr = opendir(mTmpDir);
      if (dirPtr == NULL) {
         Fatal() << "FrameServer::feedVideoToDragonsGapingMaw unable to open directory \""
                 << mTmpDir << "\": " << strerror(errno);
         pvExitFailure("");
      }
      for (struct dirent *entry = readdir(dirPtr); entry; entry = readdir(dirPtr)) {
         char *filename = entry->d_name;
         if (isFrame(filename)) {
            std::string s(mTmpDir);
            s += "/";
            s += filename;
            mListOfFrames.push_back(s);
         }
      }
      mListOfFrames.sort();
      mCurrentFrame = mListOfFrames.begin();
   }

   void clearFrames() {
      for (auto &s : mListOfFrames) {
         if (unlink(s.c_str())) {
            Fatal() << "FrameServer::clearFrames unable to delete \"" << s
                    << "\": " << strerror(errno);
         }
      }
      mListOfFrames.clear();
      mCurrentFrame = mListOfFrames.begin();
   }

  private:
   bool isFrame(char const *dirEntry) {
      size_t lenPrefix  = strlen(mFilenamePrefix);
      size_t lenPattern = strlen(mFilenamePattern);
      size_t lenSuffix  = strlen(mFilenameSuffix);
      if (dirEntry == NULL) {
         return false;
      }
      if (strlen(dirEntry) != lenPrefix + lenPattern + lenSuffix) {
         return false;
      }
      if (strncmp(dirEntry, mFilenamePrefix, lenPrefix)) {
         return false;
      }
      bool alldigits = true;
      for (size_t n = 0; n < strlen(mFilenamePattern); n++) {
         if (!isdigit(dirEntry[lenPrefix + n])) {
            alldigits = false;
            break;
         }
      }
      if (!alldigits) {
         return false;
      }
      if (strncmp(&dirEntry[lenPrefix + lenPattern], mFilenameSuffix, lenSuffix)) {
         return false;
      }
      return true;
   }

   // Member variables
  private:
   bool mRunWithoutHarnessFlag = false;
   int mFrameRate              = 30;
   char *mTmpDir               = nullptr;
   std::list<std::string> mListOfFrames;
   std::list<std::string>::iterator mCurrentFrame = mListOfFrames.begin();
   char const *mFilenamePrefix                    = "frame";
   char const *mFilenamePattern                   = "XXXX";
   char const *mFilenameSuffix                    = ".png";
}; // end FrameServer

// TODO: since this doesn't get attached to the HyPerCol, creating this via PV_Init::build provides
// no way to delete it when finished.
class HarnessObject : public PV::BaseObject {
  public:
   HarnessObject(char const *name, PV::HyPerCol *hc) { BaseObject::initialize(name, hc); }
   ~HarnessObject() {}
};

PV::BaseObject *createHarnessObject(char const *name, PV::HyPerCol *hc) {
   return hc ? new HarnessObject(name, hc) : nullptr;
}

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV::PV_Init pv_init(&argc, &argv, true /*allowUnrecognizedArguments*/);
   // Build the column from the params file
   pv_init.registerKeyword("BBFindConfRemapLayer", PV::Factory::create<BBFindConfRemapLayer>);
   pv_init.registerKeyword("BBFindConfRemapProbe", PV::Factory::create<BBFindConfRemapProbe>);
   pv_init.registerKeyword("ConvertFromTable", PV::Factory::create<ConvertFromTable>);
   pv_init.registerKeyword("LocalizationBBFindProbe", PV::Factory::create<LocalizationBBFindProbe>);
   pv_init.registerKeyword("LocalizationProbe", PV::Factory::create<LocalizationProbe>);
   pv_init.registerKeyword("Harness", PV::Factory::create<HarnessObject>);
   PV::HyPerCol *hc = new HyPerCol(&pv_init);
   pvAssert(hc->getStartTime() == hc->simulationTime());

   bool runWithoutHarnessFlag = true;
   int frameInterval          = 1;

   PV::PVParams *params = hc->parameters();
   int numGroups        = params->numberOfGroups();
   bool foundHarness    = false;
   for (int g = 0; g < numGroups; g++) {
      if (!strcmp(params->groupKeywordFromIndex(g), "Harness")) {
         runWithoutHarnessFlag =
               params->value(
                     params->groupNameFromIndex(g), "runWithoutHarness", runWithoutHarnessFlag)
               != 0;
         if (!runWithoutHarnessFlag) {
            frameInterval =
                  params->valueInt(params->groupNameFromIndex(g), "frameInterval", frameInterval);
         }
         foundHarness = true;
         break;
      }
   }
   if (!foundHarness && hc->columnId() == 0) {
      runWithoutHarnessFlag = false;
      InfoLog() << "Params file does not have a group with keyword Harness.  PetaVision will run "
                   "the params file as a normal run.\n";
   }

   status = runWithoutHarnessFlag ? runWithoutHarness(hc) : runWithHarness(hc, frameInterval);

   delete hc;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int runWithHarness(PV::HyPerCol *hc, int frameInterval) {
   char const *progName     = hc->getPV_InitObj()->getProgramName();
   double startTime         = hc->getStartTime();
   double stopTime          = hc->getStopTime();
   double dt                = hc->getDeltaTime();
   double displayPeriod     = stopTime - startTime;
   const int rank           = hc->columnId();
   PV::Communicator *icComm = hc->getCommunicator();

   int layerNx, layerNy, layerNf;
   int imageNx, imageNy, imageNf;
   int bufferNx, bufferNy, bufferNf;
   size_t imageBufferSize;
   uint8_t *imageBuffer;

   PV::ImageFromMemoryBuffer *imageLayer = NULL;
   for (int k = 0; k < hc->numberOfLayers(); k++) {
      PV::HyPerLayer *l                           = hc->getLayer(k);
      PV::ImageFromMemoryBuffer *img_buffer_layer = dynamic_cast<PV::ImageFromMemoryBuffer *>(l);
      if (img_buffer_layer) {
         if (imageLayer != NULL) {
            if (hc->columnId() == 0) {
               ErrorLog().printf(
                     "%s error: More than one ImageFromMemoryBuffer (\"%s\" and \"%s\").\n",
                     progName,
                     imageLayer->getName(),
                     img_buffer_layer->getName());
            }
            MPI_Barrier(icComm->communicator());
            exit(EXIT_FAILURE);
         }
         else {
            imageLayer = img_buffer_layer;
         }
      }
   }
   LocalizationProbe *localizationProbe = NULL;
   for (int k = 0; k < hc->numberOfBaseProbes(); k++) {
      PV::BaseProbe *p                      = hc->getBaseProbe(k);
      LocalizationProbe *localization_probe = dynamic_cast<LocalizationProbe *>(p);
      if (localization_probe) {
         if (localizationProbe != NULL) {
            if (hc->columnId() == 0) {
               ErrorLog().printf(
                     "%s error: More than one LocalizationProbe (\"%s\" and \"%s\").\n",
                     progName,
                     localizationProbe->getName(),
                     localization_probe->getName());
            }
            MPI_Barrier(icComm->communicator());
            exit(EXIT_FAILURE);
         }
         else {
            localizationProbe = localization_probe;
         }
      }
   }

   int status = PV_SUCCESS;
   if (rank == 0) {
      std::string errorString("");
      if (imageLayer == NULL) {
         errorString += progName;
         errorString += " error: When running in the harness, params file must have exactly one "
                        "ImageFromMemoryBuffer layer.";
         status = PV_FAILURE;
      }
      if (localizationProbe == NULL) {
         if (!errorString.empty()) {
            errorString += "\n";
         }
         errorString += progName;
         errorString += " error: When running in the harness, params file must have exactly one "
                        "LocalizationProbe.";
         status = PV_FAILURE;
      }
      if (status != PV_SUCCESS) {
         pvExitFailure(errorString.c_str());
      }
   }
   if (rank == 0) {
      layerNx = imageLayer->getLayerLoc()->nxGlobal;
      layerNy = imageLayer->getLayerLoc()->nyGlobal;
      layerNf = imageLayer->getLayerLoc()->nf;

      imageNx = layerNx;
      imageNy = layerNy;
      imageNf = 3;

      bufferNx = layerNx;
      bufferNy = layerNy;
      bufferNf = imageNf;

      imageBufferSize = (size_t)(bufferNx * bufferNy * bufferNf);
      imageBuffer     = NULL;
      GDALAllRegister();
   }

   FrameServer frameServer;
   frameServer.setFrameRate(frameInterval);
   std::string videoFile = getImageFileName(icComm);
   while (!videoFile.empty()) {
      char frameNumberStr[5];
      unsigned int numFrames = 0U;
      if (rank == 0) {
         frameServer.feedVideoToDragonsGapingMaw(videoFile.c_str());
         numFrames = frameServer.getNumFrames();
      }
      MPI_Bcast(&numFrames, 1, MPI_UNSIGNED, 0, icComm->communicator());
      for (unsigned f = 0; f < numFrames; f++) {
         startTime = hc->simulationTime();
         stopTime  = startTime + displayPeriod;
         localizationProbe->setOutputFilenameBase(videoFile.c_str());
         std::string filenameBase(localizationProbe->getOutputFilenameBase());
         filenameBase += "_frame";
         int sizeCheck = snprintf(frameNumberStr, (size_t)5, "%04u", f);
         if (sizeCheck >= 5) {
            Fatal() << "Number of frames exceeds 10000.  Exiting.";
            pvExitFailure("");
         }
         filenameBase += frameNumberStr;
         if (f == numFrames - 1) {
            filenameBase += "_last";
         }
         localizationProbe->setOutputFilenameBase(filenameBase.c_str());
         char const *framePath;
         if (rank == 0) {
            framePath = frameServer.retrieveFrame();
            pvAssert(framePath);
         }
         setImageLayerMemoryBuffer(icComm, framePath, imageLayer, &imageBuffer, &imageBufferSize);
         status = hc->run(startTime, stopTime, dt);
         if (status != PV_SUCCESS) {
            if (rank == 0) {
               Fatal() << "Run failed at t=" << hc->simulationTime() << " (startTime was "
                       << startTime << "; stopTime was " << stopTime << ").";
            }
         }
      }
      if (rank == 0) {
         frameServer.clearFrames();
      }
      videoFile = getImageFileName(icComm);
   }
   return status;
}

int runWithoutHarness(PV::HyPerCol *hc) { return hc->run(); }

std::string getImageFileName(PV::Communicator *icComm) {
   // All processes call this routine.  Calling routine is responsible for freeing the returned
   // string.
   char buffer[TEXTFILEBUFFERSIZE];
   int rank = icComm->commRank();
   if (rank == 0) {
      bool found = false;
      while (!found) {
         fprintf(stdout, "Enter filename: ");
         fflush(stdout);
         char *result = fgets(buffer, TEXTFILEBUFFERSIZE, stdin);
         if (result == NULL) {
            break;
         }

         // Ignore lines containing only whitespace
         for (char const *c = result; *c; c++) {
            if (!isspace(*c)) {
               found = true;
               break;
            }
         }
      }
      if (found) {
         size_t len = strlen(buffer);
         assert(len > 0);
         if (buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
         }
      }
      else {
         buffer[0] = '\0';
      }
   }
   MPI_Bcast(
         buffer, TEXTFILEBUFFERSIZE /*count*/, MPI_CHAR, 0 /*rootprocess*/, icComm->communicator());
   std::string filename = PV::expandLeadingTilde(buffer);
   return filename;
}

int setImageLayerMemoryBuffer(
      PV::Communicator *icComm,
      char const *imageFile,
      PV::ImageFromMemoryBuffer *imageLayer,
      uint8_t **imageBufferPtr,
      size_t *imageBufferSizePtr) {
   // Under MPI, only the root process (rank==0) uses imageFile, imageBufferPtr, or
   // imageBufferSizePtr, but nonroot processes need to call it as well,
   // because the imageBuffer is scattered to all processes during the call to
   // ImageFromMemoryBuffer::setMemoryBuffer().
   int layerNx = imageLayer->getLayerLoc()->nxGlobal;
   int layerNy = imageLayer->getLayerLoc()->nyGlobal;
   int layerNf = imageLayer->getLayerLoc()->nf;
   int imageNx, imageNy, imageNf;
   const uint8_t zeroVal = (uint8_t)0;
   const uint8_t oneVal  = (uint8_t)255;
   int xStride, yStride, bandStride;
   int rank = icComm->commRank();
   if (rank == 0) {
      // Doubleplusungood: much code duplication from PV::Image::readImage
      bool usingTempFile = false;
      char *path         = NULL;
      if (strstr(imageFile, "://") != NULL) {
         InfoLog().printf("Image from URL \"%s\"\n", imageFile);
         usingTempFile          = true;
         std::string pathstring = "/tmp/temp.XXXXXX";
         const char *ext        = strrchr(imageFile, '.');
         if (ext) {
            pathstring += ext;
         }
         path = strdup(pathstring.c_str());
         int fid;
         fid = mkstemps(path, strlen(ext));
         if (fid < 0) {
            Fatal().printf("Cannot create temp image file for image \"%s\".\n", imageFile);
         }
         close(fid);
         std::string systemstring;
         if (strstr(imageFile, "s3://") != NULL) {
            systemstring = "aws s3 cp \'";
            systemstring += imageFile;
            systemstring += "\' ";
            systemstring += path;
         }
         else { // URLs other than s3://
            systemstring = "wget -O ";
            systemstring += path;
            systemstring += " \'";
            systemstring += imageFile;
            systemstring += "\'";
         }

         int const numAttempts = MAX_FILESYSTEMCALL_TRIES;
         for (int attemptNum = 0; attemptNum < numAttempts; attemptNum++) {
            int status = system(systemstring.c_str());
            if (status != 0) {
               if (attemptNum == numAttempts - 1) {
                  Fatal().printf(
                        "download command \"%s\" failed: %s.  Exiting\n",
                        systemstring.c_str(),
                        strerror(errno));
               }
               else {
                  WarnLog().printf(
                        "download command \"%s\" failed: %s.  Retrying %d out of %d.\n",
                        systemstring.c_str(),
                        strerror(errno),
                        attemptNum + 1,
                        numAttempts);
                  sleep(1);
               }
            }
            else {
               break;
            }
         }
      }
      else {
         InfoLog().printf("Image from file \"%s\"\n", imageFile);
         path = strdup(imageFile);
      }
      GDALDataset *gdalDataset = (GDALDataset *)GDALOpen(path, GA_ReadOnly);
      if (gdalDataset == NULL) {
         Fatal().printf(
               "setImageLayerMemoryBuffer: GDALOpen failed for image \"%s\".\n", imageFile);
      }
      imageNx                   = gdalDataset->GetRasterXSize();
      imageNy                   = gdalDataset->GetRasterYSize();
      imageNf                   = GDALGetRasterCount(gdalDataset);
      size_t newImageBufferSize = (size_t)imageNx * (size_t)imageNy * (size_t)imageNf;
      uint8_t *imageBuffer      = *imageBufferPtr;
      size_t imageBufferSize    = *imageBufferSizePtr;
      if (imageBuffer == NULL || newImageBufferSize != imageBufferSize) {
         imageBufferSize = newImageBufferSize;
         imageBuffer     = (uint8_t *)realloc(imageBuffer, imageBufferSize * sizeof(uint8_t));
         if (imageBuffer == NULL) {
            Fatal().printf(
                  "setImageLayerMemoryBuffer: Unable to create image buffer of size %d-by-%d-by-%d "
                  "for image \"%s\": %s\n",
                  imageNx,
                  imageNy,
                  imageNf,
                  imageFile,
                  strerror(errno));
         }
      }

      bool isBinary = true;
      for (int iBand = 0; iBand < imageNf; iBand++) {
         GDALRasterBandH hBand = GDALGetRasterBand(gdalDataset, iBand + 1);
         char **metadata       = GDALGetMetadata(hBand, "Image_Structure");
         if (CSLCount(metadata) > 0) {
            bool found = false;
            for (int i = 0; metadata[i] != NULL; i++) {
               if (strcmp(metadata[i], "NBITS=1") == 0) {
                  found = true;
                  isBinary &= true;
                  break;
               }
            }
            if (!found) {
               isBinary &= false;
            }
         }
         else {
            isBinary = false;
         }
         GDALDataType dataType = gdalDataset->GetRasterBand(iBand + 1)->GetRasterDataType();
         // Why are we using both GDALGetRasterBand and GDALDataset::GetRasterBand?
         if (dataType != GDT_Byte) {
            Fatal().printf(
                  "setImageLayerMemoryBuffer: Image file \"%s\", band %d, is not GDT_Byte type.\n",
                  imageFile,
                  iBand + 1);
         }
      }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (size_t n = 0; n < imageBufferSize; n++) {
         imageBuffer[n] = oneVal;
      }

      xStride    = imageNf;
      yStride    = imageNf * imageNx;
      bandStride = 1;
      gdalDataset->RasterIO(
            GF_Read,
            0 /*xOffset*/,
            0 /*yOffset*/,
            imageNx,
            imageNy,
            imageBuffer,
            imageNx,
            imageNy,
            GDT_Byte,
            imageNf,
            NULL,
            xStride * sizeof(uint8_t),
            yStride * sizeof(uint8_t),
            bandStride * sizeof(uint8_t));

      GDALClose(gdalDataset);
      if (usingTempFile) {
         int rmstatus = remove(path);
         if (rmstatus) {
            Fatal().printf("remove(\"%s\") failed.  Exiting.\n", path);
         }
      }
      free(path);

      *imageBufferPtr     = imageBuffer;
      *imageBufferSizePtr = imageBufferSize;

      int imageDims[3];
      imageDims[0] = imageNx;
      imageDims[1] = imageNy;
      imageDims[2] = imageNf;
      MPI_Bcast(imageDims, 3, MPI_INT, 0, icComm->communicator());
   }
   else {
      int imageDims[3];
      MPI_Bcast(imageDims, 3, MPI_INT, 0, icComm->communicator());
      imageNx    = imageDims[0];
      imageNy    = imageDims[1];
      imageNf    = imageDims[2];
      xStride    = imageNf;
      yStride    = imageNf * imageNx;
      bandStride = 1;
   }
   imageLayer->setMemoryBuffer(
         *imageBufferPtr, imageNy, imageNx, imageNf, xStride, yStride, bandStride, zeroVal, oneVal);
   return PV_SUCCESS;
}
