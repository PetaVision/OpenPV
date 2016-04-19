/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"

#include "../arch/mpi/mpi.h"
#include <assert.h>
#include <string.h>
#include <iostream>

#ifdef PV_USE_GDAL
#  include <gdal.h>
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif // PV_USE_GDAL

namespace PV {

#ifdef PV_USE_GDAL

Image::Image() {
   initialize_base();
}

Image::Image(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

Image::~Image() {
   //free(imageFilename); // It is not an error to pass NULL to free().
   //imageFilename = NULL;
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   delete randState; randState = NULL;

   if(writePosition){
      if (getParent()->icCommunicator()->commRank()==0 && fp_pos != NULL && fp_pos->isfile) {
         PV_fclose(fp_pos);
      }
   }
   free(writeImagesExtension);
}

int Image::initialize_base() {
   numChannels = 0;
   mpi_datatypes = NULL;
   data = NULL;
   //imageFilename = NULL;
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
   biases[0]   = 0;
   biases[1]   = 0;
   //frameNumber = 0;
   randState = NULL;
   //posstream = NULL;
   //pvpFileTime = 0;
   //frameStartBuf = NULL;
   //countBuf = NULL;
   //biasConstraintMethod = 0; 
   //padValue = 0;
   return PV_SUCCESS;
}

int Image::initialize(const char * name, HyPerCol * hc) {
   int status = BaseInput::initialize(name, hc);
   return status;
}

int Image::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);

   //ioParam_imagePath(ioFlag);

   ioParam_autoResizeFlag(ioFlag);

   return status;
}

//void Image::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
//   parent->ioParamStringRequired(ioFlag, name, "imagePath", &imageFilename);
//}


void Image::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "autoResizeFlag", &autoResizeFlag, autoResizeFlag);
}

//int Image::scatterImageFile(const char * path, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber, bool autoResizeFlag, int batchIdx)
//{
//   //int status = PV_SUCCESS;
//   //int fileType;
//   //if (comm->commRank()==0) {
//   //   fileType = getFileType(filename);
//   //}
//   //MPI_Bcast(&fileType, 1, MPI_INT, 0, comm->communicator());
//   //if (fileType == PVP_FILE_TYPE) {
//   //   status = scatterImageFilePVP(path, xOffset, yOffset, comm, loc, buf, frameNumber, batchIdx);
//   //}
//   //else {
//      status = scatterImageFileGDAL(path, xOffset, yOffset, comm, loc, buf, autoResizeFlag);
//   //}
//   return status;
//}

//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int Image::retrieveData(double timef, double dt)
{
   assert(inputPath);
   int status = PV_SUCCESS;
   for(int b = 0; b < parent->getNBatch(); b++){
      //Using member varibles here
      status = readImage(inputPath, b, offsets[0], offsets[1], offsetAnchor);
      assert(status == PV_SUCCESS);
   }
   return status;
}

double Image::getDeltaUpdateTime(){
   if(jitterFlag){
      return parent->getDeltaTime();
   }
   else{
      return -1; //Never update
   }
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

   const MPI_Comm mpi_comm = comm->communicator();
   GDALDataType dataType;
   char** metadata;
   char isBinary = true;

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int src = 0;
      const int tag = 13;

      MPI_Bcast(&dataType, 1, MPI_INT, 0, mpi_comm);
      MPI_Bcast(&isBinary, 1, MPI_CHAR, 0, mpi_comm);
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
      GDALRasterBand * poBand = dataset->GetRasterBand(1);
      dataType = poBand->GetRasterDataType();
#ifdef PV_USE_MPI
      MPI_Bcast(&dataType, 1, MPI_INT, 0, mpi_comm);
#endif // PV_USE_MPI

      // GDAL defines whether a band is binary, not whether the image as a whole is binary.
      // Set isBinary to false if any band is not binary (metadata doesn't have NBITS=1)
      for(int iBand = 0; iBand < GDALGetRasterCount(dataset); iBand++){
         GDALRasterBandH hBand = GDALGetRasterBand(dataset, iBand+1);
         metadata = GDALGetMetadata(hBand, "Image_Structure");
         if(CSLCount(metadata) > 0){
            bool found = false;
            for(int i = 0; metadata[i] != NULL; i++){
               if(strcmp(metadata[i], "NBITS=1") == 0){
                  found = true;
                  break;
               }
            }
            if(!found){
               isBinary = false;
            }
         }
         else{
            isBinary = false;
         }
      }
#ifdef PV_USE_MPI
      MPI_Bcast(&isBinary, 1, MPI_CHAR, 0, mpi_comm);
#endif // PV_USE_MPI

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

      // if nf > bands of image, it will copy the gray image to each
      //band of layer
      assert(numBands == 1 || numBands == bandsInFile || (numBands > 1 && bandsInFile == 1));

      int dest = -1;
      const int tag = 13;

      float padValue_conv;
      if(dataType == GDT_Byte){
         padValue_conv = padValue * 255.0f;
      }
      else if(dataType == GDT_UInt16){
         padValue_conv = padValue * 65535.0f;
      }
      else{
         std::cout << "Image data type " << GDALGetDataTypeName(dataType) << " not implemented for image rescaling\n";
         exit(-1);
      }

      using std::min;
      using std::max;
      for( dest = nyProcs*nxProcs-1; dest >= 0; dest-- ) {

         //Need to clear buffer before reading, since we're skipping some of the buffer
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int i = 0; i < nx * ny * bandsInFile; i++){ 
            //Fill with padValue
            buf[i] = padValue_conv;
         }


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
         }//End autoResizeFlag
         else {
            //Need to calculate corner image pixels in globalres space
            //offset is in Image space
            int img_left = -xOffset;
            int img_top = -yOffset;
            int img_right = img_left + xImageSize;
            int img_bot = img_top + yImageSize;

            //Check if in bounds
            if(img_left >= kx+nx || img_right <= kx || img_top >= ky+ny || img_bot <= ky){
               //Do mpi_send to keep number of sends correct
               if(dest > 0){
                  MPI_Send(buf, numTotal, MPI_FLOAT, dest, tag, mpi_comm);
               }
               continue;
            }


            //start of the buffer on the left/top side
            int buf_left = img_left > kx ? img_left-kx : 0;
            int buf_top = img_top > ky ? img_top-ky : 0;
            int buf_right = img_right < kx+nx ? img_right-kx : nx;
            int buf_bot = img_bot < ky+ny ? img_bot-ky : ny;

            int buf_xSize = buf_right - buf_left; 
            int buf_ySize = buf_bot - buf_top;
            assert(buf_xSize > 0 && buf_ySize > 0);

            int buf_offset = buf_top * nx * bandsInFile + buf_left * bandsInFile;

            int img_offset_x = kx+xOffset;
            int img_offset_y = ky+yOffset;
            if(img_offset_x < 0){
               img_offset_x = 0;
            }
            if(img_offset_y < 0){
               img_offset_y = 0;
            }

            float* buf_start = buf + buf_offset;

            dataset->RasterIO(GF_Read, img_offset_x, img_offset_y, buf_xSize, buf_ySize, buf_start,
                  buf_xSize,
                  buf_ySize, GDT_Float32, bandsInFile, NULL,
                  bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));

         }
#ifdef DEBUG_OUTPUT
         fprintf(stderr, "[%2d]: scatterImageFileGDAL: sending to %d xSize==%d"
               " ySize==%d bandsInFile==%d size==%d total(over all procs)==%d\n",
               comm->commRank(), dest, nx, ny, bandsInFile, numTotal,
               nx*ny*comm->commSize());
#endif // DEBUG_OUTPUT

#ifdef PV_USE_MPI
         if(dest > 0){
            MPI_Send(buf, numTotal, MPI_FLOAT, dest, tag, mpi_comm);
         }
#endif // PV_USE_MPI
      }

      GDALClose(dataset);

      //Moved to above for loop
      // get local image portion

      //? same logic as before, except this time we know that the row and column are 0

      //if (autoResizeFlag){

      //   if (xImageSize/(double)xTotalSize < yImageSize/(double)yTotalSize){
      //      int new_y = int(round(ny*xImageSize/(double)xTotalSize));
      //      int y_off = int(round((yImageSize - new_y*nyProcs)/2.0));

      //      int offset_y = max(min(y_off,yOffset),-y_off);

      //      //fprintf(stderr, "kx = %d, ky = %d, nx = %d, new_y = %d", 0, 0, xImageSize/nxProcs, new_y);
      //      dataset->RasterIO(GF_Read, 0, y_off + offset_y, xImageSize/nxProcs, new_y, buf, nx, ny,
      //                        GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
      //                        bandsInFile*nx*sizeof(float), sizeof(float));
      //   }
      //   else{
      //      int new_x = int(round(nx*yImageSize/(double)yTotalSize));
      //      int x_off = int(round((xImageSize - new_x*nxProcs)/2.0));

      //      int offset_x = max(min(x_off,xOffset),-x_off);

      //      //fprintf(stderr, "xImageSize = %d, xTotalSize = %d, yImageSize = %d, yTotalSize = %d", xImageSize, xTotalSize, yImageSize, yTotalSize);
      //      //fprintf(stderr, "kx = %d, ky = %d, new_x = %d, ny = %d",
      //      //0, 0, new_x, yImageSize/nyProcs);
      //      dataset->RasterIO(GF_Read, x_off + offset_x, 0, new_x, yImageSize/nyProcs, buf, nx, ny,
      //                        GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float),
      //                        bandsInFile*nx*sizeof(float),sizeof(float));
      //   }
      //}
      //else {

      //   //fprintf(stderr,"just checking");
      //    dataset->RasterIO(GF_Read, xOffset, yOffset, nx, ny, buf, nx, ny,
      //                      GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));
      //}

      //GDALClose(dataset);
   }
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   if (status == 0) {
      if(!isBinary){
         float fac;
         if(dataType == GDT_Byte){
            fac = 1.0f / 255.0f;  // normalize to 1.0
         }
         else if(dataType == GDT_UInt16){
            fac = 1.0f / 65535.0f;  // normalize to 1.0
         }
         else{
            std::cout << "Image data type " << GDALGetDataTypeName(dataType) << " not implemented for image rescaling\n";
            exit(-1);
         }
         for( int n=0; n<numTotal; n++ ) {
            buf[n] *= fac;
         }
      }
   }
   return status;
}

int Image::communicateInitInfo() {
   int status = BaseInput::communicateInitInfo();
   int fileType = getFileType(inputPath);
   if(fileType == PVP_FILE_TYPE){
      std::cout << "Image/Movie no longer reads PVP files. Use ImagePvp/MoviePvp layer instead.\n";
      exit(-1);
   }
   return status;
}


// TODO: checkpointWrite and checkpointRead need to handle nextBiasChange


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

void Image::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   //Default to -1 in Image
   parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, -1.0);
}

//int Image::outputState(double time, bool last)
//{
//   //io_timer->start();
//   // this could probably use Marion's update time interval
//   // for some classes
//   //
//   //io_timer->stop();
//
//   return 0;
//}

int Image::readImage(const char * filename, int targetBatchIdx, int offsetX, int offsetY, const char* anchor)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   if(useImageBCflag){ //Expand dimensions to the extended space
      loc->nx = loc->nx + loc->halo.lt + loc->halo.rt;
      loc->ny = loc->ny + loc->halo.dn + loc->halo.up;
      //TODO this seems to fix the pvp ext vs res offset if imageBC flag is on, but only for no mpi runs
      //offsetX = offsetX - loc->halo.lt;
      //offsetY = offsetY - loc->halo.up;
   }

   // read the image and scatter the local portions
   char * path = NULL;
   bool usingTempFile = false;
   int numAttempts = 5;

   if (parent->columnId()==0) {
      if (strstr(filename, "://") != NULL) {
         usingTempFile = true;
         std::string pathstring = parent->getOutputPath();
         pathstring += "/temp.XXXXXX";
         const char * ext = strrchr(filename, '.');
         if (ext) { pathstring += ext; }
         path = strdup(pathstring.c_str());
         int fid;
         fid=mkstemps(path, strlen(ext));
         if (fid<0) {
            fprintf(stderr,"Cannot create temp image file.\n");
            exit(EXIT_FAILURE);
         }
         close(fid);
         std::string systemstring;
         if (strstr(filename, "s3://") != NULL) {
            systemstring = "aws s3 cp \'";
            systemstring += filename;
            systemstring += "\' ";
            systemstring += path;
         }
         else { // URLs other than s3://
            systemstring = "wget -O ";
            systemstring += path;
            systemstring += " \'";
            systemstring += filename;
            systemstring += "\'";
         }
         
         for(int attemptNum = 0; attemptNum < numAttempts; attemptNum++){
            int status = system(systemstring.c_str());
            if(status != 0){
               if(attemptNum == numAttempts - 1){
                  fprintf(stderr, "download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
                  exit(EXIT_FAILURE);
               }
               else{
                  fprintf(stderr, "download command \"%s\" failed: %s.  Retrying %d out of %d.\n", systemstring.c_str(), strerror(errno), attemptNum+1, numAttempts);
                  sleep(1);
               }
            }
            else{
               break;
            }
         }
      }
      else {
         path = strdup(filename);
      }
   }
   GDALColorInterp * colorbandtypes = NULL;
   status = getImageInfoGDAL(path, parent->icCommunicator(), &imageLoc, &colorbandtypes);
   if(status != 0) {
      fprintf(stderr, "Movie: Unable to get image info for \"%s\"\n", filename);
      abort();
   }

   //See if we are padding
   int n = loc->nx * loc->ny * imageLoc.nf;

   // Use number of bands in file instead of in params, to allow for grayscale conversion
   float * buf = new float[n];
   assert(buf != NULL);

   int aOffsetX = getOffsetX(anchor, offsetX);
   int aOffsetY = getOffsetY(anchor, offsetY);

   status = scatterImageFileGDAL(path, aOffsetX, aOffsetY, parent->icCommunicator(), loc, buf, this->autoResizeFlag);
   if (status != PV_SUCCESS) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Image::readImage failed for layer \"%s\"\n", getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (usingTempFile) {
      int rmstatus = remove(path);
      if (rmstatus) {
         fprintf(stderr, "remove(\"%s\") failed.  Exiting.\n", path);
         exit(EXIT_FAILURE);
      }
   }
   free(path);

   assert(status == PV_SUCCESS);
   if( loc->nf == 1 && imageLoc.nf > 1 ) {
      buf = convertToGrayScale(buf,loc->nx,loc->ny,imageLoc.nf, colorbandtypes);
      //Redefine n for grayscale images
      n = loc->nx * loc->ny;
   }
   else if (loc->nf > 1 && imageLoc.nf == 1)
   {
       buf = copyGrayScaletoMultiBands(buf,loc->nx,loc->ny,loc->nf,colorbandtypes);
       n = loc->nx * loc->ny * loc->nf;
       
   }
   free(colorbandtypes); colorbandtypes = NULL;

   //This copies the buffer to activity buffer
   if( status == PV_SUCCESS ) copyFromInteriorBuffer(buf, targetBatchIdx, 1.0f);

   delete[] buf;

   if(useImageBCflag){ //Restore non-extended dimensions
      loc->nx = loc->nx - loc->halo.lt - loc->halo.rt;
      loc->ny = loc->ny - loc->halo.dn - loc->halo.up;
   }

   return status;
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
   delete[] buf;
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
   delete[] buf;
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

#else // PV_USE_GDAL
Image::Image(const char * name, HyPerCol * hc) {
   if (hc->columnId()==0) {
      fprintf(stderr, "Image class requires compiling with PV_USE_GDAL set\n");
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
Image::Image() {}
#endif // PV_USE_GDAL

BaseObject * createImage(char const * name, HyPerCol * hc) {
   return hc ? new Image(name, hc) : NULL;
}

} // namespace PV
