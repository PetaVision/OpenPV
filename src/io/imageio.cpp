#include "imageio.hpp"
#include "io.h"
#include "fileio.hpp"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef PV_USE_GDAL
#  include <gdal.h>
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif // PV_USE_GDAL

#undef DEBUG_OUTPUT

// Are copy{To,From}LocBuffer necessary?  Can't we MPI_Bcast loc with count sizeof(locSize) and type MPI_CHAR?
static void copyToLocBuffer(int buf[], PVLayerLoc * loc)
{
   buf[0] = loc->nx;
   buf[1] = loc->ny;
   buf[2] = loc->nf;
   buf[3] = loc->nbatch;
   buf[4] = loc->nxGlobal;
   buf[5] = loc->nyGlobal;
   buf[6] = loc->kx0;
   buf[7] = loc->ky0;
   buf[8] = loc->halo.lt;
   buf[9] = loc->halo.rt;
   buf[10] = loc->halo.dn;
   buf[11] = loc->halo.up;
}

static void copyFromLocBuffer(int buf[], PVLayerLoc * loc)
{
   loc->nx       = buf[0];
   loc->ny       = buf[1];
   loc->nf       = buf[2];
   loc->nbatch = buf[3];
   loc->nxGlobal = buf[4];
   loc->nyGlobal = buf[5];
   loc->kx0      = buf[6];
   loc->ky0      = buf[7];
   loc->halo.lt  = buf[8];
   loc->halo.rt  = buf[9];
   loc->halo.dn  = buf[10];
   loc->halo.up  = buf[11];
}

int getFileType(const char * filename)
{
   const char * ext = strrchr(filename, '.');
   if (ext && strcmp(ext, ".pvp") == 0) {
      return PVP_FILE_TYPE;
   }
   return 0;
}

#ifdef OBSOLETE // Marked obsolete Jan 29, 2016.  getImageInfo was commented out in the .cpp file some time ago.
///**
// * Calculates location information given processor distribution and the
// * size of the image.
// *
// * @filename the name of the image file (in)
// * @ic the inter-column communicator (in)
// * @loc location information (inout) (loc->nx and loc->ny are out)
// */
//int getImageInfo(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes)
//{
//   int fileType;
//   if (comm->commRank()==0) {
//      fileType = getFileType(filename);
//   }
//   MPI_Bcast(&fileType, 1, MPI_INT, 0, comm->communicator());
//   if (fileType == PVP_FILE_TYPE) {
//      return getImageInfoPVP(filename, comm, loc, colorbandtypes);
//   }
//   return getImageInfoGDAL(filename, comm, loc, colorbandtypes);
//}
#endif // OBSOLETE // Marked obsolete Jan 29, 2016.  getImageInfo was commented out in the .cpp file some time ago.


int getImageInfoPVP(const char * filename, PV::Communicator * comm, PVLayerLoc * loc)
{
   // const int locSize = sizeof(PVLayerLoc) / sizeof(int);
   // int locBuf[locSize];
   int status = 0;

   // LayerLoc should contain 12 ints
   //assert(sizeof(PVLayerLoc) / sizeof(int) == 12);

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   fprintf(stderr, "[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           comm->commRank(), nxProcs, nyProcs, icRow, icCol);
#endif // DEBUG_OUTPUT

   PV_Stream * pvstream = NULL;
   if (comm->commRank()==0) { pvstream = PV::PV_fopen(filename, "rb", false/*verifyWrites*/); }
   int numParams = NUM_PAR_BYTE_PARAMS;
   int params[numParams];
   pvp_read_header(pvstream, comm, params, &numParams);
   PV::PV_fclose(pvstream); pvstream = NULL;

   assert(numParams == NUM_PAR_BYTE_PARAMS);
   //assert(params[INDEX_FILE_TYPE] == PVP_NONSPIKING_ACT_FILE_TYPE);

   const int dataSize = params[INDEX_DATA_SIZE];
   const int dataType = params[INDEX_DATA_TYPE];

   //assert( (dataType == PV_BYTE_TYPE && dataSize == 1) || (dataType == PV_FLOAT_TYPE && dataSize == 4));

   loc->nx       = params[INDEX_NX];
   loc->ny       = params[INDEX_NY];
   loc->nxGlobal = params[INDEX_NX_GLOBAL];
   loc->nyGlobal = params[INDEX_NY_GLOBAL];
   loc->kx0      = params[INDEX_KX0];
   loc->ky0      = params[INDEX_KY0];
   //loc->nb       = params[INDEX_NB];
   loc->nf       = params[INDEX_NF];

   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;

   return status;
}



int gatherImageFile(const char * filename,
                    PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * pvdata_buf, bool verifyWrites){
   unsigned char * char_buf;
   const int numItems = loc->nx * loc->ny * loc->nf;
   char_buf = (unsigned char *) calloc(numItems, sizeof(unsigned char));
   assert( char_buf != NULL );
   pvdata_t max_buf = -1.0e20;
   pvdata_t min_buf = 1.0e20;
   for (int i = 0; i < numItems; i++) {
      max_buf = pvdata_buf[i] > max_buf ? pvdata_buf[i] : max_buf;
      min_buf = pvdata_buf[i] < min_buf ? pvdata_buf[i] : min_buf;
   }
   pvdata_t range_buf = max_buf - min_buf;  // all char_buf == 0
   if (range_buf == 0) {
      range_buf = 1.0;
   }
   for (int i = 0; i < numItems; i++) {
      char_buf[i] = 255 * ( pvdata_buf[i] - min_buf ) / range_buf;
   }
   int status = gatherImageFile(filename, comm, loc, char_buf, verifyWrites);
   free(char_buf);
   return status;
}

int gatherImageFile(const char * filename,
                    PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return gatherImageFilePVP(filename, comm, loc, buf, verifyWrites);
   }
#ifdef PV_USE_GDAL
   return gatherImageFileGDAL(filename, comm, loc, buf, verifyWrites);
#else // PV_USE_GDAL
   if (comm->commRank()==0) {
      fprintf(stderr, "Error reading \"%s\": " GDAL_CONFIG_ERR_STR, filename);
   }
   return PV_FAILURE;
#endif // PV_USE_GDAL
}

int gatherImageFilePVP(const char * filename,
                       PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites)
{
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   PV_Stream * pvstream = NULL;
   if (rank==rootproc) {
      pvstream = PV::PV_fopen(filename, "wb", verifyWrites);
      if (pvstream==NULL) {
         fprintf(stderr, "gatherImageFilePVP error opening \"%s\" for writing.\n", filename);
         abort();
      }
      int params[NUM_PAR_BYTE_PARAMS];
      const int numParams  = NUM_PAR_BYTE_PARAMS;
      const int headerSize = numParams * sizeof(int);
      const int recordSize = loc->nxGlobal * loc->nyGlobal * loc->nf;

      params[INDEX_HEADER_SIZE] = headerSize;
      params[INDEX_NUM_PARAMS]  = numParams;
      params[INDEX_FILE_TYPE]   = PVP_FILE_TYPE;
      params[INDEX_NX]          = loc->nxGlobal;
      params[INDEX_NY]          = loc->nyGlobal;
      params[INDEX_NF]          = loc->nf;
      params[INDEX_NUM_RECORDS] = 1;
      params[INDEX_RECORD_SIZE] = recordSize;
      params[INDEX_DATA_SIZE]   = 1; // sizeof(unsigned char);
      params[INDEX_DATA_TYPE]   = PV_BYTE_TYPE;
      params[INDEX_NX_PROCS]    = 1;
      params[INDEX_NY_PROCS]    = 1;
      params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
      params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
      params[INDEX_KX0]         = 0;
      params[INDEX_KY0]         = 0;
      params[INDEX_NBATCH]          = loc->nbatch; // loc->nb;
      params[INDEX_NBANDS]      = 1;

      int numWrite = PV::PV_fwrite(params, sizeof(int), numParams, pvstream);
      if (numWrite != numParams) {
         fprintf(stderr, "gatherImageFilePVP error writing the header.  fwrite called with %d parameters; %d were written.\n", numParams, numWrite);
         abort();
      }
   }
   status = gatherActivity(pvstream, comm, rootproc, buf, loc, false/*extended*/);
   // buf is a nonextended buffer.  Image layers copy buf into the extended data buffer by calling Image::copyFromInteriorBuffer
   if (rank==rootproc) {
      PV::PV_fclose(pvstream); pvstream=NULL;
   }
   return status;
}

#ifdef PV_USE_GDAL
GDALDataset * PV_GDALOpen(const char * filename)
{
   int gdalopencounts = 0;
   GDALDataset * dataset = NULL;
   while (dataset == NULL) {
      dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
      if (dataset != NULL) break;
      gdalopencounts++;
      if (gdalopencounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (dataset == NULL) {
      fprintf(stderr, "getImageInfoGDAL error opening \"%s\": %s\n", filename,
            strerror(errno));
   }
   return dataset;
}

int getImageInfoGDAL(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes)
{
   int status = PV_SUCCESS;
   int rank = comm->commRank();

   const int locSize = sizeof(PVLayerLoc) / sizeof(int) + 1; // The extra 1 is for the status of the OpenGDAL call
   // LayerLoc should contain 12 ints, so locSize should be 13.
   //assert(locSize == 13);

   int locBuf[locSize];

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           rank, nxProcs, nyProcs, icRow, icCol);
#endif // DEBUG_OUTPUT

   if (rank == 0) {
      GDALAllRegister();

      GDALDataset* dataset = PV_GDALOpen(filename);
      if (dataset==NULL) status = PV_FAILURE; // PV_GDALOpen prints an error message

      if (status==PV_SUCCESS) {
         int xImageSize = dataset->GetRasterXSize();
         int yImageSize = dataset->GetRasterYSize();

         loc->nf = dataset->GetRasterCount();
         if( colorbandtypes ) {
            *colorbandtypes = (GDALColorInterp *) malloc(loc->nf*sizeof(GDALColorInterp));
            if( *colorbandtypes == NULL ) {
               fprintf(stderr, "getImageInfoGDAL: Rank 0 process unable to allocate memory for colorbandtypes\n");
               abort();
            }
            for( int b=0; b<loc->nf; b++) {
               GDALRasterBand * band = dataset->GetRasterBand(b+1);
               (*colorbandtypes)[b] = band->GetColorInterpretation();
            }
         }

         // calculate local layer size

         //SL nx and ny are no longer being used in image loc
         //int nx = xImageSize / nxProcs;
         //int ny = yImageSize / nyProcs;
         //loc->nx = nx;
         //loc->ny = ny;
         loc->nx = -1;
         loc->ny = -1;

         // loc->nb = 0;
         memset(&loc->halo, 0, sizeof(loc->halo));

         //loc->nxGlobal = nxProcs * nx;
         //loc->nyGlobal = nyProcs * ny;
         //Image loc does not have a batch index
         loc->nbatch = 0;
         loc->nxGlobal = xImageSize;
         loc->nyGlobal = yImageSize;
         loc->kx0 = 0;
         loc->ky0 = 0;

         locBuf[0] = PV_SUCCESS;
         copyToLocBuffer(&locBuf[1], loc);

         GDALClose(dataset);
      }
      else {
         locBuf[0] = PV_FAILURE;
         memset(&locBuf[1], 0, (locSize-1)*sizeof(int));
         if (colorbandtypes) *colorbandtypes = NULL;
      }
   }

#ifdef PV_USE_MPI
   // broadcast location information
   MPI_Bcast(locBuf, locSize, MPI_INT, 0, comm->communicator());
   copyFromLocBuffer(&locBuf[1], loc);
#endif // PV_USE_MPI

   status = locBuf[0];

   // fix up layer indices
   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;

#ifdef PV_USE_MPI
   // broadcast colorband type info.  This needs to follow copyFromLocBuffer because
   // it depends on the loc->nf which is set in copyFromLocBuffer.
   // status was MPI_Bcast along with locBuf, there is no danger of some processes calling MPI_Bcast and others not.
   if( colorbandtypes ) {
      if (rank==0) {
         if (status==PV_SUCCESS) {
            MPI_Bcast(*colorbandtypes, loc->nf*sizeof(GDALColorInterp), MPI_BYTE, 0, comm->communicator());
         }
      }
      else {
         if (status==PV_SUCCESS) {
            *colorbandtypes = (GDALColorInterp *) malloc(loc->nf*sizeof(GDALColorInterp));
            if( *colorbandtypes == NULL ) {
               fprintf(stderr, "getImageInfoGDAL: Rank %d process unable to allocate memory for colorbandtypes\n", rank);
               abort();
            }
            MPI_Bcast(*colorbandtypes, loc->nf*sizeof(GDALColorInterp), MPI_BYTE, 0, comm->communicator());
         }
         else {
            *colorbandtypes = NULL;
         }
      }
   }
#endif // PV_USE_MPI

   return status;
}

int gatherImageFileGDAL(const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf, bool verifyWrites)
{
   if (verifyWrites && comm->commRank()==0) {
      fprintf(stderr, "Warning: gatherImageFileGDAL called for \"%s\" with verifyWrites set to true.\n", filename);
      fprintf(stderr, "Readback has not been implemented for this function.\n");
   }

   int status = 0;

   // const int maxBands = 3;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   // assert(numBands <= maxBands);

#ifdef PV_USE_MPI
   const int nxnynf = nx * ny * numBands;

   const int tag = 14;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int dest = 0;

      MPI_Send(buf, nxnynf, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: sent to 0, nx==%d ny==%d nf==%d size==%d\n",
              comm->commRank(), nx, ny, numBands, nxnynf);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI
   }
   else {
      GDALAllRegister();

      GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");
      if (driver == NULL) {
         exit(1);
      }

      int xImageSize = nx * nxProcs;
      int yImageSize = ny * nyProcs;

      GDALDataset * dataset = driver->Create(filename, xImageSize, yImageSize, numBands,
                                             GDT_Byte, NULL);

      if (dataset == NULL) {
          fprintf(stderr, "[%2d]: gather: failed to open file %s\n", comm->commRank(), filename);
      }
      else {
#ifdef DEBUG_OUTPUT
          fprintf(stderr, "[%2d]: gather: opened file %s\n", comm->commRank(), filename);
#endif // DEBUG_OUTPUT
      }

      assert(numBands <= dataset->GetRasterCount()); // Since dataset was created using numBands, what is this assert testing for?

      // write local image portion
      dataset->RasterIO(GF_Write, 0, 0, nx, ny, buf, nx, ny, GDT_Byte,
                        numBands, NULL,numBands, numBands*nx, 1);

#ifdef PV_USE_MPI
      int src = -1;
      unsigned char * icBuf = (unsigned char *) malloc(nxnynf * sizeof(unsigned char));
      assert(icBuf != NULL);
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
            int kx = nx * px;
            int ky = ny * py;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: gather: receiving from %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), src, nx, ny, nxny*numBands,
                    numTotal*comm->commSize());
#endif // DEBUG_OUTPUT
            MPI_Recv(icBuf, nxnynf, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
            dataset->RasterIO(GF_Write, kx, ky, nx, ny, icBuf, nx, ny, GDT_Byte,
                              numBands, NULL, numBands, numBands*nx, 1);
         }
      }
      free(icBuf);
#endif // PV_USE_MPI
      GDALClose(dataset);
   }

   return status;
}
#endif // PV_USE_GDAL

#ifdef OBSOLETE // Marked obsolete Jan 29, 2016.  At some point it was commented out.
int scatterImageFile(const char * filename, int xOffset, int yOffset,
                     PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber, bool autoResizeFlag)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return scatterImageFilePVP(filename, xOffset, yOffset, comm, loc, buf, frameNumber);
   }
   return scatterImageFileGDAL(filename, xOffset, yOffset, comm, loc, buf, autoResizeFlag);
}

int scatterImageFilePVP(const char * filename, int xOffset, int yOffset,
                        PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber)
{
   // Read a PVP file and scatter it to the multiple processes.
   int status = PV_SUCCESS;

   int rootproc = 0;
   int rank = comm->commRank();
   PV_Stream * pvstream;
   if(rank==rootproc){
       pvstream = PV::pvp_open_read_file(filename, comm);
   }
   else{
       pvstream = NULL;
   }
   int numParams = NUM_BIN_PARAMS;
   int params[numParams];
   PV::pvp_read_header(pvstream, comm, params, &numParams);

   if (rank==rootproc) {
      PVLayerLoc fileloc;
      int headerSize = params[INDEX_HEADER_SIZE];
      fileloc.nx = params[INDEX_NX];
      fileloc.ny = params[INDEX_NY];
      fileloc.nf = params[INDEX_NF];
      fileloc.nb = params[INDEX_NB];
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


      //bool spiking = false;
      double timed = 0.0;
      int filetype = params[INDEX_FILE_TYPE];
      int framesize;
      long framepos;
      switch (filetype) {
      case PVP_FILE_TYPE:
         break;
      case PVP_ACT_FILE_TYPE:
         //spiking = true;

         //Where should I save values_start?
         //If i haven't calculated values_start yet:
         long findpos;
         findpos = (long)headerSize;
         int values_start[INDEX_NBANDS];
         PV::PV_fseek(pvstream, framepos, SEEK_SET);
         PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         for (int i = 0; i<INDEX_NBANDS; i++) {
            PV::PV_fseek(pvstream, findpos, SEEK_SET);
         }




         PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": Reading spiking PVP files into an Image layer hasn't been implemented yet.\n", filename);
         abort();
         break;
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         framesize = recordsize*datasize*nxProcs*nyProcs+8;
         framepos = (long)framesize * (long)frameNumber + (long)headerSize;
         //ONLY READING TIME INFO HERE
         PV::PV_fseek(pvstream, framepos, SEEK_SET);
         PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         status = PV_SUCCESS;
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
      scatterActivity(pvstream, comm, rootproc, buf, loc, false, &fileloc, xOffset, yOffset);
      // buf is a nonextended layer.  Image layers copy the extended buffer data into buf by calling Image::copyToInteriorBuffer
      PV::PV_fclose(pvstream); pvstream = NULL;
   }
   else {
      scatterActivity(pvstream, comm, rootproc, buf, loc, false, NULL, xOffset, yOffset);
   }
   return status;
}

int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
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
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatterImageFileGDAL: broadcast from 0, total number of bytes in buffer is %d\n", numTotal);
#endif // DEBUG_OUTPUT
      MPI_Bcast(&numTotal, 1, MPI_INT, 0, mpi_comm);

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
         //? a random number between -offset and +offset (offset specified in params file) or
         //? between the maximum translation possible, whichever is smaller.

         if (autoResizeFlag){
             using std::min;

             if (xImageSize/(double)xTotalSize < yImageSize/(double)yTotalSize){
                int new_y = int(round(ny*xImageSize/(double)xTotalSize));
                int y_off = int(round((yImageSize - new_y*nyProcs)/2.0));

                int jitter_y = 0;
                if (yOffset > 0){
                   srand(time(NULL));
                   jitter_y = rand() % min(y_off*2,yOffset*2) - min(y_off,yOffset);
                }

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

                int jitter_x = 0;
                if (xOffset > 0){
                   srand(time(NULL));
                   jitter_x = rand() % min(x_off*2,xOffset*2) - min(x_off,xOffset);
                }

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
         using std::min;
         
         if (xImageSize/(double)xTotalSize < yImageSize/(double)yTotalSize){
            int new_y = int(round(ny*xImageSize/(double)xTotalSize));
            int y_off = int(round((yImageSize - new_y*nyProcs)/2.0));
            
            int jitter_y = 0;
            if (yOffset > 0){
               srand(time(NULL));
               jitter_y = rand() % min(y_off*2,yOffset*2) - min(y_off,yOffset);
            }

            //fprintf(stderr, "kx = %d, ky = %d, nx = %d, new_y = %d", 0, 0, xImageSize/nxProcs, new_y);
            dataset->RasterIO(GF_Read, 0, y_off + jitter_y, xImageSize/nxProcs, new_y, buf, nx, ny,
                              GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float), 
                              bandsInFile*nx*sizeof(float), sizeof(float));           
         }
         else{
            int new_x = int(round(nx*yImageSize/(double)yTotalSize));
            int x_off = int(round((xImageSize - new_x*nxProcs)/2.0));
            
            int jitter_x = 0;
            if (xOffset > 0){
               srand(time(NULL));
               jitter_x = rand() % min(x_off*2,xOffset*2) - min(x_off,xOffset);
            }
            
            //fprintf(stderr, "xImageSize = %d, xTotalSize = %d, yImageSize = %d, yTotalSize = %d", xImageSize, xTotalSize, yImageSize, yTotalSize);
            //fprintf(stderr, "kx = %d, ky = %d, new_x = %d, ny = %d",
            //0, 0, new_x, yImageSize/nyProcs);
            dataset->RasterIO(GF_Read, x_off + jitter_x, 0, new_x, yImageSize/nyProcs, buf, nx, ny,
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
#endif // OBSOLETE // Marked obsolete Jan 29, 2016.  At some point it was commented out.
