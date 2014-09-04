/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"

#ifdef PV_USE_MPI
#include <mpi.h>
#endif
#include <assert.h>
#include <string.h>
#include <iostream>

#ifdef PV_USE_GDAL
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif // PV_USE_GDAL
#include <gdal.h>



namespace PV {

Image::Image() {
   initialize_base();
}

Image::Image(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

Image::~Image() {
   free(filename); // It is not an error to pass NULL to free().
   filename = NULL;
   if(frameStartBuf){
      free(frameStartBuf);
      frameStartBuf = NULL;
   }
   if(countBuf){
      free(countBuf);
      countBuf = NULL;
   }
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   delete randState; randState = NULL;

   if(writePosition){
      if (getParent()->icCommunicator()->commRank()==0 && fp_pos != NULL && fp_pos->isfile) {
         PV_fclose(fp_pos);
      }
   }
   if(offsetAnchor){
      free(offsetAnchor);
   }
   free(writeImagesExtension);
}

int Image::initialize_base() {
   numChannels = 0;
   mpi_datatypes = NULL;
   data = NULL;
   filename = NULL;
   imageData = NULL;
   useImageBCflag = false;
   autoResizeFlag = false;
   writeImages = false;
   writeImagesExtension = NULL;
   inverseFlag = false;
   normalizeLuminanceFlag = false;
   normalizeStdDev = true;
   offsets[0] = 0;
   offsets[1] = 0;
   offsetAnchor = (char*) "tl";
   jitterFlag = false;
   jitterType = RANDOM_WALK;
   timeSinceLastJitter = 0;
   jitterRefractoryPeriod = 0;
   stepSize = 0;
   persistenceProb = 0.0;
   recurrenceProb = 1.0;
   biasChangeTime = FLT_MAX;
   writePosition = 0;
   fp_pos = NULL;
   biases[0]   = getOffsetX();
   biases[1]   = getOffsetY();
   frameNumber = 0;
   randState = NULL;
   posstream = NULL;
   pvpFileTime = 0;
   frameStartBuf = NULL;
   countBuf = NULL;
   return PV_SUCCESS;
}

int Image::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);

   needFrameSizesForSpiking = true;

   // Much of the functionality that was previously here has been moved to either read-methods, communicateInitInfo, or allocateDataStructures

   this->lastUpdateTime = parent->getStartTime();

   PVParams * params = parent->parameters();

   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      assert(!params->presentAndNotBeenRead(name, "offsetX"));
      assert(!params->presentAndNotBeenRead(name, "offsetY"));
      assert(!params->presentAndNotBeenRead(name, "offsetAnchor"));
      biases[0] = getOffsetX();
      biases[1] = getOffsetY();
   }

   return status;
}

int Image::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);

   ioParam_imagePath(ioFlag);
   ioParam_offsetAnchor(ioFlag);
   ioParam_offsets(ioFlag);
   ioParam_writeImages(ioFlag);
   ioParam_writeImagesExtension(ioFlag);

   ioParam_useImageBCflag(ioFlag);
   ioParam_autoResizeFlag(ioFlag);
   ioParam_inverseFlag(ioFlag);
   ioParam_normalizeLuminanceFlag(ioFlag);
   ioParam_normalizeStdDev(ioFlag);

   ioParam_frameNumber(ioFlag);

   // Although Image itself does not use jitter, both Movie and Patterns do, so jitterFlag is read in Image.
   ioParam_jitterFlag(ioFlag);
   ioParam_jitterType(ioFlag);
   ioParam_jitterRefractoryPeriod(ioFlag);
   ioParam_stepSize(ioFlag);
   ioParam_persistenceProb(ioFlag);
   ioParam_recurrenceProb(ioFlag);
   ioParam_biasChangeTime(ioFlag);
   ioParam_biasConstraintMethod(ioFlag);
   ioParam_offsetConstraintMethod(ioFlag);
   ioParam_writePosition(ioFlag);
   ioParam_useParamsImage(ioFlag);

   return status;
}

void Image::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "imagePath", &filename);
}

int Image::ioParam_offsets(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "offsetX", &offsets[0], offsets[0]);
   parent->ioParamValue(ioFlag, name, "offsetY", &offsets[1], offsets[1]);

   return PV_SUCCESS;
}

void Image::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
}

void Image::ioParam_writeImages(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeImages", &writeImages, writeImages);
}

void Image::ioParam_writeImagesExtension(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeImages"));
   if (writeImages) {
      parent->ioParamString(ioFlag, name, "writeImagesExtension", &writeImagesExtension, "tif");
   }
}

void Image::ioParam_useImageBCflag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "useImageBCflag", &useImageBCflag, useImageBCflag);
}

void Image::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "autoResizeFlag", &autoResizeFlag, autoResizeFlag);
}

void Image::ioParam_inverseFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "inverseFlag", &inverseFlag, inverseFlag);
}

void Image::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "normalizeLuminanceFlag", &normalizeLuminanceFlag, normalizeLuminanceFlag);
}

void Image::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeLuminanceFlag"));
   if (normalizeLuminanceFlag) {
     parent->ioParamValue(ioFlag, name, "normalizeStdDev", &normalizeStdDev, normalizeStdDev);
   }
}

void Image::ioParam_frameNumber(enum ParamsIOFlag ioFlag) {
   if (filename!=NULL && getFileType(filename)==PVP_FILE_TYPE) {
      parent->ioParamValue(ioFlag, name, "frameNumber", &frameNumber, frameNumber);
   }
}

void Image::ioParam_jitterFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "jitterFlag", &jitterFlag, jitterFlag);
}

void Image::ioParam_jitterType(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "jitterType", &jitterType, jitterType);
   }
}

void Image::ioParam_jitterRefractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "jitterRefractoryPeriod", &jitterRefractoryPeriod, jitterRefractoryPeriod);
   }
}

void Image::ioParam_stepSize(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "stepSize", &stepSize, stepSize);
   }
}

void Image::ioParam_persistenceProb(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "persistenceProb", &persistenceProb, persistenceProb);
   }
}

void Image::ioParam_recurrenceProb(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "recurrenceProb", &recurrenceProb, recurrenceProb);
   }
}


void Image::ioParam_biasChangeTime(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "biasChangeTime", &biasChangeTime, biasChangeTime);
      if (ioFlag == PARAMS_IO_READ) {
         if (biasChangeTime < 0) {
            biasChangeTime = FLT_MAX;
         }
         nextBiasChange = parent->getStartTime() + biasChangeTime;
      }
   }
}

void Image::ioParam_biasConstraintMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "biasConstraintMethod", &biasConstraintMethod, biasConstraintMethod);
      if (ioFlag == PARAMS_IO_READ && (biasConstraintMethod <0 || biasConstraintMethod >3)) {
         fprintf(stderr, "%s \"%s\": biasConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n",
               parent->parameters()->groupKeywordFromName(getName()), getName());
         exit(EXIT_FAILURE);
      }
   }
}

void Image::ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "offsetConstraintMethod", &offsetConstraintMethod, 0/*default*/);
      if (ioFlag == PARAMS_IO_READ && (offsetConstraintMethod <0 || offsetConstraintMethod >3) ) {
         fprintf(stderr, "Image layer \"%s\": offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getName());
         exit(EXIT_FAILURE);
      }
   }
}

void Image::ioParam_writePosition(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "writePosition", &writePosition, writePosition);
   }
}

void Image::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   assert(this->initVObject == NULL);
   return;
}

void Image::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", false/*correct value*/);
   }
}

void Image::ioParam_useParamsImage(enum ParamsIOFlag ioFlag) {
   // Deprecate in favor of HyPerLayer's initializeFromCheckpointFlag?
   if (parent->getCheckpointReadFlag()) {
      parent->ioParamValue(ioFlag, name, "useParamsImage", &useParamsImage, false/*default value*/, true/*warnIfAbsent*/);
      if (useParamsImage && ioFlag==PARAMS_IO_READ && parent->columnId()==0) {
         // useParamsImage was deprecated July 21, 2014
         fprintf(stderr, " *** Image \"%s\" warning: parameter useParamsImage is deprecated.\n", getName());
         fprintf(stderr, " *** Instead, set HyPerCol's initializeFromCheckpointDir to the checkpoint directory,\n");
         fprintf(stderr, " ***     HyPerCol's defaultInitializeFromCheckpointFlag to true,\n");
         fprintf(stderr, " ***     and parameter initializeFromCheckpointFlag of \"%s\" to false.\n", getName());
      }
   }
}

int Image::scatterImageFile(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber, bool autoResizeFlag)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return scatterImageFilePVP(filename, xOffset, yOffset, comm, loc, buf, frameNumber);
   }
   return scatterImageFileGDAL(filename, xOffset, yOffset, comm, loc, buf, autoResizeFlag);
}

int Image::scatterImageFilePVP(const char * filename, int xOffset, int yOffset,
                        PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber)
{
   // Read a PVP file and scatter it to the multiple processes.
   int status = PV_SUCCESS;
   int i;
   int rootproc = 0;
   int rank = comm->commRank();
   PV_Stream * pvstream;
   if(rank==rootproc){
       pvstream = PV::pvp_open_read_file(filename, comm);
   }
   else{
       pvstream = NULL;
   }
   long length = 0;
   int numParams = NUM_BIN_PARAMS;
   int params[numParams];
   PV::pvp_read_header(pvstream, comm, params, &numParams);
   if (frameNumber < 0 || frameNumber >= params[INDEX_NBANDS]) {
      if (rank==rootproc) {
         fprintf(stderr, "scatterImageFilePVP error: requested frameNumber %d but file \"%s\" only has frames numbered 0 through %d.\n", frameNumber, filename, params[INDEX_NBANDS]-1);
      }
      return PV_FAILURE;
   }

   if (rank==rootproc) {
      PVLayerLoc fileloc;
      int headerSize = params[INDEX_HEADER_SIZE];
      fileloc.nx = params[INDEX_NX];
      fileloc.ny = params[INDEX_NY];
      fileloc.nf = params[INDEX_NF];
      // fileloc.nb = params[INDEX_NB];
      fileloc.nxGlobal = params[INDEX_NX_GLOBAL];
      fileloc.nyGlobal = params[INDEX_NY_GLOBAL];
      fileloc.kx0 = params[INDEX_KX0];
      fileloc.ky0 = params[INDEX_KY0];
      int nxProcs = params[INDEX_NX_PROCS];
      int nyProcs = params[INDEX_NY_PROCS];
      int recordsize = params[INDEX_RECORD_SIZE];
      int datasize = params[INDEX_DATA_SIZE];
      if (fileloc.nx != fileloc.nxGlobal || fileloc.ny != fileloc.nyGlobal ||
          nxProcs != 1 || nyProcs != 1 ||
          fileloc.kx0 != 0 || fileloc.ky0 != 0) {
          fprintf(stderr, "File \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
          abort();
      }



      bool spiking = false;
      double timed = 0.0;
      int filetype = params[INDEX_FILE_TYPE];
      int framesize;
      long framepos;
      unsigned len;
      char tempPosFilename [PV_PATH_MAX];
      const char * posFilename;

      switch (filetype) {
      case PVP_FILE_TYPE:
         break;

      case PVP_ACT_FILE_TYPE:
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:

         //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
         //Only need to do this once

         //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
         //Only need to do this once
         if (needFrameSizesForSpiking) {
            std::cout << "Calculating file positions\n";
            frameStartBuf = (long *) calloc(params[INDEX_NBANDS],sizeof(long));
            if (frameStartBuf==NULL) {
               fprintf(stderr, "scatterImageFilePVP unable to allocate memory for frameStart.\n");
               status = PV_FAILURE;
               abort();
            }
            countBuf = (int *) calloc(params[INDEX_NBANDS],sizeof(int));
            if (countBuf==NULL) {
               fprintf(stderr, "scatterImageFilePVP unable to allocate memory for frameLength.\n");
               status = PV_FAILURE;
               abort();
            }

            //Fseek past the header and first timestamp
            PV::PV_fseek(pvstream, (long)8 + (long)headerSize, SEEK_SET);
            int percent = 0;
            for (i = 0; i<params[INDEX_NBANDS]; i++) {
               int newpercent = 100*(float(i)/params[INDEX_NBANDS]);
               if(percent != newpercent){
                  percent = newpercent;
                  std::cout << "\r" << percent << "% Done";
                  std::cout.flush();
               }
               //First byte position should always be 92
               if (i == 0) {
                  frameStartBuf[i] = (long)92;
               }
               //Read in the number of active neurons for that frame and calculate byte position
               else {
                  PV::PV_fread(&countBuf[i-1], sizeof(int), 1, pvstream);
                  frameStartBuf[i] = frameStartBuf[i-1] + (long)countBuf[i-1]*(long)datasize + (long)12;
                  PV::PV_fseek(pvstream, frameStartBuf[i] - (long)4, SEEK_SET);
               }
            }
            std::cout << "\r" << percent << "% Done\n";
            std::cout.flush();
            //We still need the last count
            PV::PV_fread(&countBuf[i-1], sizeof(int), 1, pvstream);

            //So we don't have to calculate frameStart and count again
            needFrameSizesForSpiking = false;
         }
         framepos = (long)frameStartBuf[frameNumber];
         length = countBuf[frameNumber];
         PV::PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
         PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         std::cout << "Reading file time " << timed << " on time " << parent->simulationTime() << "\n";
         unsigned int dropLength;
         PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
         assert(dropLength == length);
         status = PV_SUCCESS;
         break;
               
            ////See if position file exists
            //strcpy(tempPosFilename, filename);
            //len = strlen(filename);
            ////Check suffix of filename
            //if(strcmp(&filename[len-4], ".pvp") != 0){
            //   fprintf(stderr, "Filename %s must end in \".pvp\".\n", filename);
            //   status = PV_FAILURE;
            //   abort();
            //}
            ////Change suffix to pos
            //tempPosFilename[len-2] = 'o';
            //tempPosFilename[len-1] = 's';
            //posFilename = tempPosFilename;

            ////If file doesn't exist, write file
            //if(access(posFilename, F_OK) == -1){
               ////Fseek past the header and first timestamp
               //PV::PV_fseek(pvstream, (long)8 + (long)headerSize, SEEK_SET);

               //int percent = 0;
               //long frameStart;
               //long count;
               //posstream = PV_fopen(posFilename, "w");
               //assert(posstream);
               //for (i = 0; i<params[INDEX_NBANDS]; i++) {
               //   int newpercent = 100*(float(i)/params[INDEX_NBANDS]);
               //   if(percent != newpercent){
               //      percent = newpercent;
               //      std::cout << "\r" << percent << "% Done";
               //      std::cout.flush();
               //   }
               //   //First byte position should always be 92
               //   if (i == 0) {
               //      frameStart = (long)92;
               //   }
               //   //Read in the number of active neurons for that frame and calculate byte position
               //   else {
               //      PV::PV_fread(&count, sizeof(unsigned int), 1, pvstream);
               //      frameStart = frameStart + (long)count*(long)datasize + (long)12;
               //      PV::PV_fseek(pvstream, frameStart - (long)4, SEEK_SET);
               //   }
               //   //Write to file
               //   status = (PV_fwrite(&frameStart, sizeof(long), 1, posstream) != 1);
               //}
               //std::cout << "\r" << percent << "% Done\n";
               //std::cout.flush();

               //PV_fclose(posstream);
            //}
            //posstream = PV_fopen(posFilename, "r");
            //assert(posstream);
            ////So we don't have to calculate frameStart and count again
            //needFrameSizesForSpiking = false;
         //}
         //At this point, posstream should be pointing to something
         //assert(posstream);
         //Calculate based on frame number where to read from posstream
         //PV::PV_fseek(posstream, frameNumber * sizeof(long), SEEK_SET);
         //PV::PV_fread(&framepos, sizeof(long), 1, posstream);
         ////Calculate where in position file to look at fileposition 
         //PV::PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
         //PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         //PV::PV_fread(&length, sizeof(unsigned int), 1, pvstream);
         //std::cout << "length: " << length << "\n";
         //status = PV_SUCCESS;
         //break;
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         length = 0; //Don't need to compute this for nonspiking files.
         framesize = recordsize*datasize*nxProcs*nyProcs+8;
         framepos = (long)framesize * (long)frameNumber + (long)headerSize;
         //ONLY READING TIME INFO HERE
         status = PV::PV_fseek(pvstream, framepos, SEEK_SET);
         if (status != 0) {
            fprintf(stderr, "scatterImageFilePVP error: Unable to seek to start of frame %d in \"%s\": %s\n", frameNumber, filename, strerror(errno));
            status = PV_FAILURE;
         }
         if (status == PV_SUCCESS) {
            size_t numread = PV::PV_fread(&timed, sizeof(double), (size_t) 1, pvstream);
            if (numread != (size_t) 1) {
               fprintf(stderr, "scatterImageFilePVP error: Unable to read timestamp from frame %d of file \"%s\":", frameNumber, filename);
               if (feof(pvstream->fp)) { fprintf(stderr, " end-of-file."); }
               if (ferror(pvstream->fp)) { fprintf(stderr, " fread error."); }
               fprintf(stderr, "\n");
               status = PV_FAILURE;
            }
         }
         break;
      case PVP_WGT_FILE_TYPE:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": file is a weight file, not an image file.\n", filename);
         break;
      case PVP_KERNEL_FILE_TYPE:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": file is a weight file, not an image file.\n", filename);
         break;

      default:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": filetype %d is unrecognized.\n", filename ,filetype);
         status = PV_FAILURE;
         break;
      }
      if (status != PV_SUCCESS) {
         exit(EXIT_FAILURE);
      }

      pvpFileTime = timed;
      std::cout << "Reading pvpFileTime " << pvpFileTime << " with offset (" << xOffset << "," << yOffset << ")\n";

      scatterActivity(pvstream, comm, rootproc, buf, loc, false, &fileloc, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      // buf is a nonextended layer.  Image layers copy the extended buffer data into buf by calling Image::copyToInteriorBuffer
      PV::PV_fclose(pvstream); pvstream = NULL;
   }
   else {
      if ((params[INDEX_FILE_TYPE] == PVP_ACT_FILE_TYPE) || (params[INDEX_FILE_TYPE] == PVP_ACT_SPARSEVALUES_FILE_TYPE)) {
         scatterActivity(pvstream, comm, rootproc, buf, loc, false, NULL, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      }

      else if (params[INDEX_FILE_TYPE] == PVP_NONSPIKING_ACT_FILE_TYPE) {
         scatterActivity(pvstream, comm, rootproc, buf, loc, false, NULL, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      }

   }
   return status;
}















int Image::scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf, bool autoResizeFlag)
{
   int status = 0;

#ifdef PV_USE_GDAL
   // const int maxBands = 3;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   int numTotal; // will be nx*ny*bandsInFile;

#ifdef PV_USE_MPI
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int src = 0;
      const int tag = 13;

      MPI_Bcast(&numTotal, 1, MPI_INT, 0, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatterImageFileGDAL: received from 0, total number of bytes in buffer is %d\n", numTotal);
#endif // DEBUG_OUTPUT
      MPI_Recv(buf, numTotal, MPI_FLOAT, src, tag, mpi_comm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
      int nf=numTotal/(nx*ny);
      assert( nf*nx*ny == numTotal );
      fprintf(stderr, "[%2d]: scatterImageFileGDAL: received from 0, nx==%d ny==%d nf==%d size==%d\n",
              comm->commRank(), nx, ny, nf, numTotal);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
   }
   else {
      GDALAllRegister();

      GDALDataset * dataset = PV_GDALOpen(filename); // (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
      if (dataset==NULL) return 1; // PV_GDALOpen prints an error message.
      int xImageSize = dataset->GetRasterXSize();
      int yImageSize = dataset->GetRasterYSize();
      const int bandsInFile = dataset->GetRasterCount();

      numTotal = nx * ny * bandsInFile;
#ifdef PV_USE_MPI
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatterImageFileGDAL: broadcast from 0, total number of bytes in buffer is %d\n", numTotal);
#endif // DEBUG_OUTPUT
      MPI_Bcast(&numTotal, 1, MPI_INT, 0, mpi_comm);
#endif // PV_USE_MPI

      int xTotalSize = nx * nxProcs;
      int yTotalSize = ny * nyProcs;

      // if false, PetaVision will NOT automatically resize your images, so you had better
      // choose the right offsets and sizes.
      if (!autoResizeFlag){
         if (xOffset + xTotalSize > xImageSize || yOffset + yTotalSize > yImageSize) {
            fprintf(stderr, "[ 0]: scatterImageFile: image size too small, "
                  "xTotalSize==%d xImageSize==%d yTotalSize==%d yImageSize==%d xOffset==%d yOffset==%d\n",
                  xTotalSize, xImageSize, yTotalSize, yImageSize, xOffset, yOffset);
            fprintf(stderr, "[ 0]: xSize==%d ySize==%d nxProcs==%d nyProcs==%d\n",
                  nx, ny, nxProcs, nyProcs);
            GDALClose(dataset);
            return -1;
         }
      }
      // if nf > bands of image, it will copy the gray image to each
      //band of layer
      assert(numBands == 1 || numBands == bandsInFile || (numBands > 1 && bandsInFile == 1));




#ifdef PV_USE_MPI
      int dest = -1;
      const int tag = 13;

      using std::min;
      using std::max;
      for( dest = 1; dest < nyProcs*nxProcs; dest++ ) {
         int col = columnFromRank(dest,nyProcs,nxProcs);
         int row = rowFromRank(dest,nyProcs,nxProcs);
         int kx = nx * col;
         int ky = ny * row;

         //? For the auto resize flag, PV checks which side (x or y) is the shortest, relative to the
         //? hypercolumn size specified.  Then it determines the largest chunk it can possibly take
         //? from the image with the correct aspect ratio determined by hypercolumn.  It then
         //? determines the offset needed in the long dimension to center the cropped image,
         //? and reads in that portion of the image.  The offset can optionally be translated by
         //? offset{X,Y} specified in the params file (values can be positive or negative).
         //? If the specified offset takes the cropped image outside the image file, it uses the
         //? largest-magnitude offset that stays within the image file's borders.

         if (autoResizeFlag){

             if (xImageSize/(double)xTotalSize < yImageSize/(double)yTotalSize){
                int new_y = int(round(ny*xImageSize/(double)xTotalSize));
                int y_off = int(round((yImageSize - new_y*nyProcs)/2.0));

                int jitter_y = max(min(y_off,yOffset),-y_off);

                kx = xImageSize/nxProcs * col;
                ky = new_y * row;

                //fprintf(stderr, "kx = %d, ky = %d, nx = %d, new_y = %d", kx, ky, xImageSize/nxProcs, new_y);

                dataset->RasterIO(GF_Read, kx, ky + y_off + jitter_y, xImageSize/nxProcs, new_y, buf, nx, ny,
                                  GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
                                  bandsInFile*nx*sizeof(float), sizeof(float));
             }
             else{
                int new_x = int(round(nx*yImageSize/(double)yTotalSize));
                int x_off = int(round((xImageSize - new_x*nxProcs)/2.0));

                int jitter_x = max(min(x_off,xOffset),-x_off);

                kx = new_x * col;
                ky = yImageSize/nyProcs * row;

                //fprintf(stderr, "kx = %d, ky = %d, new_x = %d, ny = %d, x_off = %d", kx, ky, new_x, yImageSize/nyProcs, x_off);
                dataset->RasterIO(GF_Read, kx + x_off + jitter_x, ky, new_x, yImageSize/nyProcs, buf, nx, ny,
                                  GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
                                  bandsInFile*nx*sizeof(float),sizeof(float));
             }
          }
         else {

            //fprintf(stderr, "just checking");
             dataset->RasterIO(GF_Read, kx+xOffset, ky+yOffset, nx, ny, buf,
                               nx, ny, GDT_Float32, bandsInFile, NULL,
                               bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));

         }
#ifdef DEBUG_OUTPUT
fprintf(stderr, "[%2d]: scatterImageFileGDAL: sending to %d xSize==%d"
      " ySize==%d bandsInFile==%d size==%d total(over all procs)==%d\n",
      comm->commRank(), dest, nx, ny, bandsInFile, numTotal,
      nx*ny*comm->commSize());
#endif // DEBUG_OUTPUT
         MPI_Send(buf, numTotal, MPI_FLOAT, dest, tag, mpi_comm);
      }
#endif // PV_USE_MPI

      // get local image portion

      //? same logic as before, except this time we know that the row and column are 0

      if (autoResizeFlag){

         if (xImageSize/(double)xTotalSize < yImageSize/(double)yTotalSize){
            int new_y = int(round(ny*xImageSize/(double)xTotalSize));
            int y_off = int(round((yImageSize - new_y*nyProcs)/2.0));

            int offset_y = max(min(y_off,yOffset),-y_off);

            //fprintf(stderr, "kx = %d, ky = %d, nx = %d, new_y = %d", 0, 0, xImageSize/nxProcs, new_y);
            dataset->RasterIO(GF_Read, 0, y_off + offset_y, xImageSize/nxProcs, new_y, buf, nx, ny,
                              GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
                              bandsInFile*nx*sizeof(float), sizeof(float));
         }
         else{
            int new_x = int(round(nx*yImageSize/(double)yTotalSize));
            int x_off = int(round((xImageSize - new_x*nxProcs)/2.0));

            int offset_x = max(min(x_off,xOffset),-x_off);

            //fprintf(stderr, "xImageSize = %d, xTotalSize = %d, yImageSize = %d, yTotalSize = %d", xImageSize, xTotalSize, yImageSize, yTotalSize);
            //fprintf(stderr, "kx = %d, ky = %d, new_x = %d, ny = %d",
            //0, 0, new_x, yImageSize/nyProcs);
            dataset->RasterIO(GF_Read, x_off + offset_x, 0, new_x, yImageSize/nyProcs, buf, nx, ny,
                              GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
                              bandsInFile*nx*sizeof(float),sizeof(float));
         }
      }
      else {

         //fprintf(stderr,"just checking");
          dataset->RasterIO(GF_Read, xOffset, yOffset, nx, ny, buf, nx, ny,
                            GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));
      }

      GDALClose(dataset);
   }
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   if (status == 0) {
      // Workaround for gdal problem with binary images.
      // If the all values are zero or 1, assume its a binary image and keep the values the same.
      // If other values appear, divide by 255 to scale to [0,1]
      bool isgrayscale = false;
      for (int n=0; n<numTotal; n++) {
         if (buf[n] != 0.0 && buf[n] != 1.0) {
            isgrayscale = true;
            break;
         }
      }
      if (isgrayscale) {
         float fac = 1.0f / 255.0f;  // normalize to 1.0
         for( int n=0; n<numTotal; n++ ) {
            buf[n] *= fac;
         }
      }
   }
   return status;
}

int Image::communicateInitInfo() {
   return HyPerLayer::communicateInitInfo();
}

int Image::requireChannel(int channelNeeded, int * numChannelsResult) {
   if (parent->columnId()==0) {
      fprintf(stderr, "%s \"%s\" cannot be a post-synaptic layer.\n",
            parent->parameters()->groupKeywordFromName(name), name);
   }
   *numChannelsResult = 0;
   return PV_FAILURE;
}

int Image::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   if (jitterFlag) {
      status = initRandState();
   }

   data = clayer->activity->data;

   if(filename != NULL) {
      GDALColorInterp * colorbandtypes = NULL;
      status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if( getLayerLoc()->nf != imageLoc.nf && getLayerLoc()->nf != 1) {
         fprintf(stderr, "Image %s: file %s has %d features but the layer has %d features.  Exiting.\n",
               name, filename, imageLoc.nf, getLayerLoc()->nf);
         exit(PV_FAILURE);
      }
      status = readImage(filename, getOffsetX(), getOffsetY(), colorbandtypes);
      assert(status == PV_SUCCESS);
      free(colorbandtypes); colorbandtypes = NULL;
   }
   else {
      this->imageLoc = * getLayerLoc();
   }

   // Open the file recording jitter positions.
   // This is in allocateDataStructures in case a subclass does something weird with the offsets, causing
   // the initial offsets to be unknown until the allocateDataStructures phase
   if(jitterFlag && writePosition){
      // Note: biasX and biasY are used only to calculate offsetX and offsetY;
      //       offsetX and offsetY are used only by readImage;
      //       readImage only uses the offsets in the zero-rank process
      // Therefore, the other ranks do not need to have their offsets stored.
      // In fact, it would be reasonable for the nonzero ranks not to compute biases and offsets at all,
      // but I chose not to fill the code with even more if(rank==0) statements.
      if( parent->icCommunicator()->commRank()==0 ) {
         char file_name[PV_PATH_MAX];

         int nchars = snprintf(file_name, PV_PATH_MAX, "%s/%s_jitter.txt", parent->getOutputPath(), getName());
         if (nchars >= PV_PATH_MAX) {
            fprintf(stderr, "Path for jitter positions \"%s/%s_jitter.txt is too long.\n", parent->getOutputPath(), getName());
            abort();
         }
         printf("Image layer \"%s\" will write jitter positions to %s\n",getName(), file_name);
         fp_pos = PV_fopen(file_name,"w");
         if(fp_pos == NULL) {
            fprintf(stderr, "Image \"%s\" unable to open file \"%s\" for writing jitter positions.\n", getName(), file_name);
            abort();
         }
         fprintf(fp_pos->fp,"Layer \"%s\", t=%f, bias x=%d y=%d, offset x=%d y=%d\n",getName(),parent->simulationTime(),biases[0],biases[1],
               getOffsetX(),getOffsetY());
      }
   }

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

   // exchange border information
   exchange();

   return status;
}

// TODO: checkpointWrite and checkpointRead need to handle nextBiasChange

int Image::initRandState() {
   assert(randState==NULL);
   randState = new Random(parent, 1);
   if (randState==NULL) {
      fprintf(stderr, "%s \"%s\" error in rank %d process: unable to create object of class Random.\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int Image::allocateV() {
   clayer->V = NULL;
   return PV_SUCCESS;
}

int Image::initializeV() {
   assert(getV()==NULL);
   return PV_SUCCESS;
}

int Image::initializeActivity() {
   return PV_SUCCESS;
}

#ifdef PV_USE_OPENCL
// no need for threads for now for image
//
int Image::initializeThreadBuffers(const char * kernelName)
{
   return CL_SUCCESS;
}

// no need for threads for now for image
//
int Image::initializeThreadKernels(const char * kernelName)
{
   return CL_SUCCESS;
}
#endif

/**
 * return some useful information about the image
 */
int Image::tag()
{
   return 0;
}

int Image::recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int neighbor)
{
   // this should never be called as an image shouldn't have an incoming connection
   recvsyn_timer->start();
   recvsyn_timer->stop();
   return 0;
}

double Image::getDeltaUpdateTime(){
   if(jitterFlag){
      return 1;
   }
   else{
      return -1; //Never update
   }
}

//Now handeled in HyPerLayer needUpdate, with getDeltaUpdateTime
//bool Image::needUpdate(double time, double dt){
//   //Image should never need an update unless jittered
//   if(jitterFlag){
//      return true;
//   }
//   else{
//      return false;
//   }
//}

/**
 * update the image buffers
 */
int Image::updateState(double time, double dt)
{
   // make sure image is copied to activity buffer
   //
   //update_timer->start();
   //update_timer->stop();
   return 0;
}

int Image::outputState(double time, bool last)
{
   //io_timer->start();
   // this could probably use Marion's update time interval
   // for some classes
   //
   //io_timer->stop();

   return 0;
}

int Image::checkpointRead(const char * cpDir, double * timeptr){
   PVParams * params = parent->parameters();
   if (this->useParamsImage) {
      if (parent->columnId()==0) {
         fprintf(stderr,"Initializing image from params file location ! \n");
      }
      *timeptr = parent->simulationTime(); // fakes the pvp time stamp
   }
   else {
      if (parent->columnId()==0) {
         fprintf(stderr,"Initializing image from checkpoint NOT from params file location! \n");
      }
      HyPerLayer::checkpointRead(cpDir, timeptr);
   }

   return PV_SUCCESS;
}


//! CLEAR IMAGE
/*!
 * this is Image specific.
 */
int Image::clearImage()
{
   // default is to do nothing for now
   // it could, for example, set the data buffer to zero.

   return 0;
}

int Image::readImage(const char * filename)
{
   return readImage(filename, 0, 0, NULL);
}

int Image::readImage(const char * filename, int offsetX, int offsetY, GDALColorInterp * colorbandtypes)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   if(useImageBCflag){ //Expand dimensions to the extended space
      loc->nx = loc->nx + loc->halo.lt + loc->halo.rt;
      loc->ny = loc->ny + loc->halo.dn + loc->halo.up;
   }

   int n = loc->nx * loc->ny * imageLoc.nf;

   // Use number of bands in file instead of in params, to allow for grayscale conversion
   float * buf = new float[n];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(filename, offsetX, offsetY, parent->icCommunicator(), loc, buf, frameNumber, this->autoResizeFlag);
   if (status != PV_SUCCESS) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Image::readImage failed for layer \"%s\"\n", getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   assert(status == PV_SUCCESS);
   if( loc->nf == 1 && imageLoc.nf > 1 ) {
      float * graybuf = convertToGrayScale(buf,loc->nx,loc->ny,imageLoc.nf, colorbandtypes);
      delete[] buf;
      buf = graybuf;
      //Redefine n for grayscale images
      n = loc->nx * loc->ny;
   }
   else if (loc->nf > 1 && imageLoc.nf == 1)
   {
       buf = copyGrayScaletoMultiBands(buf,loc->nx,loc->ny,loc->nf,colorbandtypes);
       n = loc->nx * loc->ny * loc->nf;
       
   }
   
   // now buf is loc->nf by loc->nx by loc->ny

   // if normalizeLuminanceFlag == true then force average luminance to be 0.5
   bool normalize_standard_dev = normalizeStdDev;
   if(normalizeLuminanceFlag){
     if (normalize_standard_dev){
       double image_sum = 0.0f;
       double image_sum2 = 0.0f;
       for (int k=0; k<n; k++) {
         image_sum += buf[k];
         image_sum2 += buf[k]*buf[k];
       }
      double image_ave = image_sum / n;
      double image_ave2 = image_sum2 / n;
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &image_ave, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      image_ave /= parent->icCommunicator()->commSize();
      MPI_Allreduce(MPI_IN_PLACE, &image_ave2, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      image_ave2 /= parent->icCommunicator()->commSize();
#endif
      // set mean to zero
      for (int k=0; k<n; k++) {
	buf[k] -= image_ave;
      }
      // set std dev to 1
      double image_std = sqrt(image_ave2 - image_ave*image_ave);
      for (int k=0; k<n; k++) {
	buf[k] /= image_std;
      }
     }
     else{
      float image_max = -FLT_MAX;
      float image_min = FLT_MAX;
      for (int k=0; k<n; k++) {
         image_max = buf[k] > image_max ? buf[k] : image_max;
         image_min = buf[k] < image_min ? buf[k] : image_min;
      }
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &image_max, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &image_min, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
#endif
      if (image_max > image_min){
	float image_stretch = 1.0f / (image_max - image_min);
	for (int k=0; k<n; k++) {
	  buf[k] -= image_min;
	  buf[k] *= image_stretch;
	}
      }
      else{ // image_max == image_min, set to gray
	//float image_shift = 0.5f - image_ave;
            for (int k=0; k<n; k++) {
	      buf[k] += 0.5f; //image_shift;
            }
      }
     }
   } // normalizeLuminanceFlag

   if( inverseFlag ) {
      for (int k=0; k<n; k++) {
         buf[k] = 1 - buf[k];
      }
   }

   if( status == PV_SUCCESS ) copyFromInteriorBuffer(buf, 1.0f);

   delete[] buf;

   if(useImageBCflag){ //Restore non-extended dimensions
      loc->nx = loc->nx - loc->halo.lt - loc->halo.rt;
      loc->ny = loc->ny - loc->halo.dn - loc->halo.up;
   }

   return status;
}

/**
 *
 * The data buffer lives in the extended space. Here, we only copy the restricted space
 * to the buffer buf. The size of this buffer is the size of the image patch - borders
 * are not included.
 *
 */
int Image::write(const char * filename)
{
   int status = 0;
   const PVLayerLoc * loc = getLayerLoc();

   const int n = loc->nx * loc->ny * loc->nf;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   status = copyToInteriorBuffer(buf, 255.0);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf);

   delete buf;

   return status;
}

int Image::exchange()
{
   return parent->icCommunicator()->exchange(data, mpi_datatypes, getLayerLoc());
}


int Image::copyToInteriorBuffer(unsigned char * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const PVHalo * halo = &loc->halo;

   for(int n=0; n<getNumNeurons(); n++) {
      int n_ex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      buf[n] = (unsigned char) (fac * data[n_ex]);
   }
   return 0;
}

int Image::copyFromInteriorBuffer(float * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   const PVHalo * halo = &loc->halo;

   if(useImageBCflag){
      for(int n=0; n<getNumExtended(); n++) {
         //int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
         data[n] = fac*buf[n];
      }
   }else{
      for(int n=0; n<getNumNeurons(); n++) {
         int n_ex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         data[n_ex] = fac*buf[n];
      }
   }

   return 0;
}

float * Image::convertToGrayScale(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes)
{
   // even though the numBands argument goes last, the routine assumes that
   // the organization of buf is, bands vary fastest, then x, then y.
   if (numBands < 2) return buf;


   const int sxcolor = numBands;
   const int sycolor = numBands*nx;
   const int sb = 1;

   const int sxgray = 1;
   const int sygray = nx;

   float * graybuf = new float[nx*ny];

   float * bandweight = (float *) malloc(numBands*sizeof(float));
   calcBandWeights(numBands, bandweight, colorbandtypes);

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = buf[i*sxcolor + j*sycolor + b*sb];
            val += d*bandweight[b];
         }
         graybuf[i*sxgray + j*sygray] = val;
      }
   }
   free(bandweight);
   return graybuf;
}
    float* Image::copyGrayScaletoMultiBands(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes)
    {
        const int sxcolor = numBands;
        const int sycolor = numBands*nx;
        const int sb = 1;

        const int sxgray = 1;
        const int sygray = nx;

        float * multiBandsBuf = new float[nx*ny*numBands];
        
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int b = 0; b < numBands; b++)
                {
                    multiBandsBuf[i*sxcolor + j*sycolor + b*sb] = buf[i*sxgray + j*sygray];
                }

            }

        }
        delete buf;
        return multiBandsBuf;
                
    }
    
int Image::calcBandWeights(int numBands, float * bandweight, GDALColorInterp * colorbandtypes) {
   int colortype = 0; // 1=grayscale(with or without alpha), return value 2=RGB(with or without alpha), 0=unrecognized
   const GDALColorInterp grayalpha[2] = {GCI_GrayIndex, GCI_AlphaBand};
   const GDALColorInterp rgba[4] = {GCI_RedBand, GCI_GreenBand, GCI_BlueBand, GCI_AlphaBand};
   const float grayalphaweights[2] = {1.0, 0.0};
   const float rgbaweights[4] = {0.30, 0.59, 0.11, 0.0}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
   switch( numBands ) {
   case 1:
      bandweight[0] = 1.0;
      colortype = 1;
      break;
   case 2:
      if ( !memcmp(colorbandtypes, grayalpha, 2*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, grayalphaweights, 2*sizeof(float));
         colortype = 1;
      }
      break;
   case 3:
      if ( !memcmp(colorbandtypes, rgba, 3*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 3*sizeof(float));
         colortype = 2;
      }
      break;
   case 4:
      if ( !memcmp(colorbandtypes, rgba, 4*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 4*sizeof(float));
         colortype = 2;
      }
      break;
   default:
      break;
   }
   if (colortype==0) {
      equalBandWeights(numBands, bandweight);
   }
   return colortype;
}

void Image::equalBandWeights(int numBands, float * bandweight) {
   float w = 1.0/(float) numBands;
   for( int b=0; b<numBands; b++ ) bandweight[b] = w;
}


//Offsets based on an anchor point, so calculate offsets based off a given anchor point
int Image::getOffsetX(){
   //Offset on left
   if(!strcmp(offsetAnchor, "tl") || !strcmp(offsetAnchor, "cl") || !strcmp(offsetAnchor, "bl")){
      return offsets[1];
   }
   //Offset in center
   else if(!strcmp(offsetAnchor, "tc") || !strcmp(offsetAnchor, "cc") || !strcmp(offsetAnchor, "bc")){
      int layerSizeX = getLayerLoc()->nxGlobal;
      return ((imageLoc.nxGlobal/2)-(layerSizeX/2) - 1) + offsets[1];
   }
   //Offset on bottom
   else if(!strcmp(offsetAnchor, "tr") || !strcmp(offsetAnchor, "cr") || !strcmp(offsetAnchor, "br")){
      int layerSizeX = getLayerLoc()->nxGlobal;
      return (imageLoc.nxGlobal - layerSizeX - 1) + offsets[1];
   }
}

int Image::getOffsetY(){
   //Offset on top
   if(!strcmp(offsetAnchor, "tl") || !strcmp(offsetAnchor, "tc") || !strcmp(offsetAnchor, "tr")){
      return offsets[0];
   }
   //Offset in center
   else if(!strcmp(offsetAnchor, "cl") || !strcmp(offsetAnchor, "cc") || !strcmp(offsetAnchor, "cr")){
      int layerSizeY = getLayerLoc()->nyGlobal;
      return ((imageLoc.nyGlobal/2)-(layerSizeY/2) - 1) + offsets[0];
   }
   //Offset on bottom
   else if(!strcmp(offsetAnchor, "bl") || !strcmp(offsetAnchor, "bc") || !strcmp(offsetAnchor, "br")){
      int layerSizeY = getLayerLoc()->nyGlobal;
      return (imageLoc.nyGlobal-layerSizeY-1) + offsets[0];
   }
}



/*
 * jitter() is not called by Image directly, but it is called by
 * its derived classes Patterns and Movie, so it's placed in Image.
 * It returns true if the offsets changed so that a new image needs
 * to be loaded/drawn.
 */
bool Image::jitter() {
   // move bias
   double timed = parent->simulationTime();
   if( timed > parent->getStartTime() && timed >= nextBiasChange ){
      calcNewBiases(stepSize);
      constrainBiases();
      nextBiasChange += biasChangeTime;
   }

   // move offset
   bool needNewImage = calcNewOffsets(stepSize);
   constrainOffsets();

   if(writePosition && parent->icCommunicator()->commRank()==0){
      fprintf(fp_pos->fp,"t=%f, bias x=%d, y=%d, offset x=%d y=%d\n",timed,biases[0],biases[1],getOffsetX(),getOffsetY());
   }
   lastUpdateTime = timed;
   return needNewImage;
}

/**
 * Calculate a bias in x or y here.  Input argument is the step size and the size of the interval of possible values
 * Output is the value of the bias.
 * It can perform a random walk of a fixed stepsize or it can perform a random jump up to a maximum length
 * equal to step.
 */
int Image::calcBias(int current_bias, int step, int sizeLength)
{
   assert(jitterFlag);
   double p;
   int dbias = 0;
   if (jitterType == RANDOM_WALK) {
      p = randState->uniformRandom();
      dbias = p < 0.5 ? step : -step;
   } else if (jitterType == RANDOM_JUMP) {
      p = randState->uniformRandom();
      dbias = (int) floor(p*(double) step) + 1;
      p = randState->uniformRandom();
      if (p < 0.5) dbias = -dbias;
   }
   else {
      assert(0); // Only allowable values of jitterType are RANDOM_WALK and RANDOM_JUMP
   }

   int new_bias = current_bias + dbias;
   new_bias = (new_bias < 0) ? -new_bias : new_bias;
   new_bias = (new_bias > sizeLength) ? sizeLength - (new_bias-sizeLength) : new_bias;
   return new_bias;
}

int Image::calcNewBiases(int stepSize) {
   assert(jitterFlag);
   int step_radius = 0; // distance to step
   switch (jitterType) {
   case RANDOM_WALK:
      step_radius = stepSize;
      break;
   case RANDOM_JUMP:
      step_radius = 1 + (int) floor(randState->uniformRandom() * stepSize);
      break;
   default:
      assert(0); // Only allowable values of jitterType are RANDOM_WALK and RANDOM_JUMP
      break;
   }
   double p = randState->uniformRandom() * 2 * PI; // direction to step
   int dx = (int) floor( step_radius * cos(p));
   int dy = (int) floor( step_radius * sin(p));
   assert(dx != 0 || dy != 0);
   biases[0] += dx;
   biases[1] += dy;
   return PV_SUCCESS;
}

/**
 * Return an offset that moves randomly around position bias
 * Perform a
 * random jump of maximum length equal to step.
 * The routine returns the resulting offset.
 * (The recurrenceProb test has been moved to the calling routine jitter() )
 */
int Image::calcBiasedOffset(int bias, int current_offset, int step, int sizeLength)
{
   assert(jitterFlag); // calcBiasedOffset should only be called when jitterFlag is true
   int new_offset;
   double p = randState->uniformRandom();
   int d_offset = (int) floor(p*(double) step) + 1;
   p = randState->uniformRandom();
   if (p<0.5) d_offset = -d_offset;
   new_offset = current_offset + d_offset;
   new_offset = (new_offset < 0) ? -new_offset : new_offset;
   new_offset = (new_offset > sizeLength) ? sizeLength - (new_offset-sizeLength) : new_offset;

   return new_offset;
}

bool Image::calcNewOffsets(int stepSize)
{
   assert(jitterFlag);

   bool needNewImage = false;
   double p = randState->uniformRandom();
   if (timeSinceLastJitter >= jitterRefractoryPeriod) {
      if (p > recurrenceProb) {
         p = randState->uniformRandom();
         if (p > persistenceProb) {
            needNewImage = true;
           int step_radius = 1 + (int) floor(randState->uniformRandom() * stepSize);
           double p = randState->uniformRandom() * 2 * PI; // direction to step
           int dx = (int) round( step_radius * cos(p));
           int dy = (int) round( step_radius * sin(p));
           assert(dx != 0 || dy != 0);
           offsets[0] += dx;
           offsets[1] += dy;
           timeSinceLastJitter = 0;
         }
      }
      else {
            assert(sizeof(*offsets) == sizeof(*biases));
            memcpy(offsets, biases, 2*sizeof(offsets));
            timeSinceLastJitter = 0;
      }
   }
   timeSinceLastJitter++;
   return needNewImage;
}

bool Image::constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method) {
   bool moved_x = point[0] < min_x || point[0] > max_x;
   bool moved_y = point[1] < min_y || point[1] > max_y;
   if (moved_x) {
      if (min_x >= max_x) {
         fprintf(stderr, "Image::constrainPoint error.  min_x=%d and max_x= %d\n", min_x, max_x);
         abort();
      }
      int size_x = max_x-min_x;
      int new_x = point[0];
      switch (method) {
      case 0: // Ignore
         break;
      case 1: // Mirror
         new_x -= min_x;
         new_x %= (2*(size_x+1));
         if (new_x<0) new_x++;
         new_x = abs(new_x);
         if (new_x>size_x) new_x = 2*size_x+1-new_x;
         new_x += min_x;
         break;
      case 2: // Stick to wall
         if (new_x<min_x) new_x = min_x;
         if (new_x>max_x) new_x = max_x;
         break;
      case 3: // Circular
         new_x -= min_x;
         new_x %= size_x+1;
         if (new_x<0) new_x += size_x+1;
         new_x += min_x;
         break;
      default:
         assert(0);
         break;
      }
      assert(new_x >= min_x && new_x <= max_x);
      point[0] = new_x;
   }
   if (moved_y) {
      if (min_y >= max_y) {
         fprintf(stderr, "Image::constrainPoint error.  min_y=%d and max_y=%d\n", min_y, max_y);
         abort();
      }
      int size_y = max_y-min_y;
      int new_y = point[1];
      switch (method) {
      case 0: // Ignore
         break;
      case 1: // Mirror
         new_y -= min_y;
         new_y %= (2*(size_y+1));
         if (new_y<0) new_y++;
         new_y = abs(new_y);
         if (new_y>=size_y) new_y = 2*size_y+1-new_y;
         new_y += min_y;
         break;
      case 2: // Stick to wall
         if (new_y<min_y) new_y = min_y;
         if (new_y>max_y) new_y = max_y;
         break;
      case 3: // Circular
         new_y -= min_y;
         new_y %= size_y+1;
         if (new_y<0) new_y += size_y+1;
         new_y += min_y;
         break;
      default:
         assert(0);
         break;
      }
      assert(new_y >= min_y && new_y <= max_y);
      point[1] = new_y;
   }
   return moved_x || moved_y;
}

bool Image::constrainBiases() {
   return constrainPoint(biases, 0, imageLoc.nxGlobal - getLayerLoc()->nxGlobal, 0, imageLoc.nyGlobal - getLayerLoc()->nyGlobal - stepSize, biasConstraintMethod);
}

bool Image::constrainOffsets() {
   return constrainPoint(offsets, 0, imageLoc.nxGlobal - getLayerLoc()->nxGlobal, 0, imageLoc.nyGlobal - getLayerLoc()->nyGlobal - stepSize, biasConstraintMethod);
}

} // namespace PV
