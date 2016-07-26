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
   data = NULL;
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
   randState = NULL;
   return PV_SUCCESS;
}

int Image::initialize(const char * name, HyPerCol * hc) {
   int status = BaseInput::initialize(name, hc);
   return status;
}

int Image::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);
   // Image has no additional parameters.
   return status;
}

//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int Image::retrieveData(double timef, double dt, int batchIndex)
{
   assert(inputPath);
   int status = PV_SUCCESS;
   //Using member varibles here
   status = readImage(inputPath);
   if(status != PV_SUCCESS) {
      pvErrorNoExit().printf("%s: readImage failed at t=%f for batchIndex %d\n", getDescription_c(), timef, batchIndex);
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

int Image::readImageFileGDAL(char const * filename, PVLayerLoc const * loc) {
   assert(parent->columnId()==0);

   GDALAllRegister();

   GDALDataset * dataset = PV_GDALOpen(filename);
   if (dataset==NULL) { return PV_FAILURE; } // PV_GDALOpen prints an error message.
   GDALRasterBand * poBand = dataset->GetRasterBand(1);
   GDALDataType dataType = poBand->GetRasterDataType();

   // GDAL defines whether a band is binary, not whether the image as a whole is binary.
   // Set isBinary to false if any band is not binary (metadata doesn't have NBITS=1)
   bool isBinary = true;
   for(int iBand = 0; iBand < GDALGetRasterCount(dataset); iBand++){
      GDALRasterBandH hBand = GDALGetRasterBand(dataset, iBand+1);
      char ** metadata = GDALGetMetadata(hBand, "Image_Structure");
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

   int const xImageSize = imageLoc.nxGlobal;
   int const yImageSize = imageLoc.nyGlobal;
   int const bandsInFile = imageLoc.nf;

   int numTotal = xImageSize * yImageSize * bandsInFile;

   dataset->RasterIO(GF_Read, 0, 0, xImageSize, yImageSize, imageData,
         xImageSize, yImageSize, GDT_Float32, bandsInFile, NULL,
         bandsInFile*sizeof(float), bandsInFile*xImageSize*sizeof(float), sizeof(float));


   GDALClose(dataset);
   if(!isBinary){
      pvadata_t fac;
      if(dataType == GDT_Byte){
         fac = 1.0f / 255.0f;  // normalize to 1.0
      }
      else if(dataType == GDT_UInt16){
         fac = 1.0f / 65535.0f;  // normalize to 1.0
      }
      else{
         pvError().printf("Image data type %s in file \"%s\" is not implemented.\n", GDALGetDataTypeName(dataType), filename);
      }
      for( int n=0; n<numTotal; n++ ) {
         imageData[n] *= fac;
      }
   }
   return EXIT_SUCCESS;
}

#ifdef INACTIVE // Commented out April 19, 2016.  Might prove useful to restore the option to resize using GDAL.
int Image::scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf, bool autoResizeFlag)
{
   int status = 0;

#ifdef PV_USE_GDAL
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   int numTotal; // will be nx*ny*bandsInFile

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
      pvDebug().printf("[%2d]: scatterImageFileGDAL: received from 0, total number of bytes in buffer is %d\n", numTotal);
#endif // DEBUG_OUTPUT
      MPI_Recv(buf, numTotal, MPI_FLOAT, src, tag, mpi_comm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
      int nf=numTotal/(nx*ny);
      assert( nf*nx*ny == numTotal );
      pvDebug().printf("[%2d]: scatterImageFileGDAL: received from 0, nx==%d ny==%d nf==%d size==%d\n",
              comm->commRank(), nx, ny, nf, numTotal);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
   }
   else {
      GDALAllRegister();

      GDALDataset * dataset = PV_GDALOpen(filename);
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
      pvDebug().printf("[%2d]: scatterImageFileGDAL: broadcast from 0, total number of bytes in buffer is %d\n", numTotal);
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
         pvError() << "Image data type " << GDALGetDataTypeName(dataType) << " not implemented for image rescaling\n";
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
         pvDebug().printf("[%2d]: scatterImageFileGDAL: sending to %d xSize==%d"
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
   }
#else
   pvError().printf(GDAL_CONFIG_ERR_STR);
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
            pvError() << "Image data type " << GDALGetDataTypeName(dataType) << " not implemented for image rescaling\n";
         }
         for( int n=0; n<numTotal; n++ ) {
            buf[n] *= fac;
         }
      }
   }
   return status;
}
#endif // INACTIVE // Commented out April 19, 2016.  Might prove useful to restore the option to resize using GDAL.

int Image::communicateInitInfo() {
   int status = BaseInput::communicateInitInfo();
   int fileType = getFileType(inputPath);
   if(fileType == PVP_FILE_TYPE){
      pvError() << "Image/Movie no longer reads PVP files. Use ImagePvp/MoviePvp layer instead.\n";
   }
   return status;
}


// TODO: checkpointWrite and checkpointRead need to handle nextBiasChange

/**
 * update the image buffers
 */
int Image::updateState(double time, double dt)
{
   return 0;
}

void Image::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   //Default to -1 in Image
   parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, -1.0);
}

int Image::readImage(const char * filename)
{
   assert(parent->columnId()==0); // readImage is called by retrieveData, which only the root process calls.  BaseInput::scatterInput does the scattering.
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   char * path = NULL;
   bool usingTempFile = false;

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
         pvError().printf("Cannot create temp image file.\n");
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

      int numAttempts = 5;
      for(int attemptNum = 0; attemptNum < numAttempts; attemptNum++){
         int status = system(systemstring.c_str());
         if(status != 0){
            if(attemptNum == numAttempts - 1){
               pvError().printf("download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
            }
            else{
               pvWarn().printf("download command \"%s\" failed: %s.  Retrying %d out of %d.\n", systemstring.c_str(), strerror(errno), attemptNum+1, numAttempts);
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
   GDALColorInterp * colorbandtypes = NULL;
   status = getImageInfoGDAL(path, &imageLoc, &colorbandtypes);
   calcColorType(imageLoc.nf, colorbandtypes);
   free(colorbandtypes); colorbandtypes = NULL;

   if(status != 0) {
      pvError().printf("Movie: Unable to get image info for \"%s\"\n", filename);
   }

   delete[] imageData;
   imageData = new pvadata_t[imageLoc.nxGlobal*imageLoc.nyGlobal*imageLoc.nf];

   status = readImageFileGDAL(path, &imageLoc);
   if (status != PV_SUCCESS) {
      pvError().printf("Image::readImage failed for %s\n", getDescription_c());
   }

   if (usingTempFile) {
      int rmstatus = remove(path);
      if (rmstatus) {
         pvError().printf("remove(\"%s\") failed.  Exiting.\n", path);
      }
   }
   free(path);

   return status;
}

int Image::calcColorType(int numBands, GDALColorInterp * colorbandtypes) {
   int colortype = 0; // 1=grayscale(with or without alpha), return value 2=RGB(with or without alpha), 0=unrecognized
   const GDALColorInterp grayalpha[2] = {GCI_GrayIndex, GCI_AlphaBand};
   const GDALColorInterp rgba[4] = {GCI_RedBand, GCI_GreenBand, GCI_BlueBand, GCI_AlphaBand};
   const float grayalphaweights[2] = {1.0, 0.0};
   const float rgbaweights[4] = {0.30, 0.59, 0.11, 0.0}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
   switch (numBands) {
   case 1:
   case 2:
      imageColorType = memcmp(colorbandtypes, grayalpha, numBands*sizeof(GDALColorInterp)) ? COLORTYPE_UNRECOGNIZED : COLORTYPE_GRAYSCALE;
      break;
   case 3:
   case 4:
      imageColorType = memcmp(colorbandtypes, rgba, numBands*sizeof(GDALColorInterp)) ? COLORTYPE_UNRECOGNIZED : COLORTYPE_RGB;
      break;
   default:
      imageColorType = COLORTYPE_UNRECOGNIZED;
      break;
   }
   return PV_SUCCESS;
}

#else // PV_USE_GDAL
Image::Image(const char * name, HyPerCol * hc) {
   if (hc->columnId()==0) {
      pvErrorNoExit().printf("Image \"%s\": Image class requires compiling with PV_USE_GDAL set\n", name);
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
Image::Image() {}
int Image::retrieveData(double timef, double dt, int batchIndex) { return PV_FAILURE; }
#endif // PV_USE_GDAL

BaseObject * createImage(char const * name, HyPerCol * hc) {
   return hc ? new Image(name, hc) : NULL;
}

} // namespace PV
