#include "imageio.hpp"
#include "io.h"

#include <assert.h>
#include <string.h>

#ifdef PV_USE_GDAL
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif // PV_USE_GDAL

#undef DEBUG_OUTPUT

static void copyToLocBuffer(int buf[], PVLayerLoc * loc)
{
   buf[0] = loc->nx;
   buf[1] = loc->ny;
   buf[2] = loc->nxGlobal;
   buf[3] = loc->nyGlobal;
   buf[4] = loc->kx0;
   buf[5] = loc->ky0;
   buf[6] = loc->nb;
   buf[7] = loc->nf;
   buf[8] = loc->halo.lt;
   buf[9] = loc->halo.rt;
   buf[10] = loc->halo.dn;
   buf[11] = loc->halo.up;
}

static void copyFromLocBuffer(int buf[], PVLayerLoc * loc)
{
   loc->nx       = buf[0];
   loc->ny       = buf[1];
   loc->nxGlobal = buf[2];
   loc->nyGlobal = buf[3];
   loc->kx0      = buf[4];
   loc->ky0      = buf[5];
   loc->nb       = buf[6];
   loc->nf       = buf[7];
   loc->halo.lt  = buf[8];
   loc->halo.rt  = buf[9];
   loc->halo.dn  = buf[10];
   loc->halo.up  = buf[11];
}

int getFileType(const char * filename)
{
   const char * ext = strrchr(filename, '.');
   if (strcmp(ext, ".pvp") == 0) {
      return PVP_FILE_TYPE;
   }
   return 0;
}

/**
 * Calculates location information given processor distribution and the
 * size of the image.
 *
 * @filename the name of the image file (in)
 * @ic the inter-column communicator (in)
 * @loc location information (inout) (loc->nx and loc->ny are out)
 */
int getImageInfo(const char * filename, PV::Communicator * comm, PVLayerLoc * loc)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return getImageInfoPVP(filename, comm, loc);
   }
   return getImageInfoGDAL(filename, comm, loc);
}

int getImageInfoPVP(const char * filename, PV::Communicator * comm, PVLayerLoc * loc)
{
   const int locSize = sizeof(PVLayerLoc) / sizeof(int);
   int locBuf[locSize];
   int status = 0;

   // LayerLoc should contain 8 ints
   assert(locSize == 8);

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   fprintf(stderr, "[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           comm->commRank(), nxProcs, nyProcs, icRow, icCol);
#endif

   if (comm->commRank() == 0) {
      int numParams, params[NUM_PAR_BYTE_PARAMS];

      FILE * fp = fopen(filename, "rb");
      assert(fp != NULL);

      numParams = pv_read_binary_params(fp, NUM_PAR_BYTE_PARAMS, params);
      fclose(fp);

      assert(numParams == NUM_PAR_BYTE_PARAMS);
      assert(params[INDEX_FILE_TYPE] == PVP_FILE_TYPE);

      const int dataSize = params[INDEX_DATA_SIZE];
      const int dataType = params[INDEX_DATA_TYPE];
//      const int nxProcs  = params[INDEX_NX_PROCS];
//      const int nyProcs  = params[INDEX_NY_PROCS];

      loc->nx       = params[INDEX_NX];
      loc->ny       = params[INDEX_NY];
      loc->nxGlobal = params[INDEX_NX_GLOBAL];
      loc->nyGlobal = params[INDEX_NY_GLOBAL];
      loc->kx0      = params[INDEX_KX0];
      loc->ky0      = params[INDEX_KY0];
      loc->nb       = params[INDEX_NB];
      loc->nf       = params[INDEX_NF];

      assert(dataSize == 1);
      assert(dataType == PV_BYTE_TYPE);

      copyToLocBuffer(locBuf, loc);
   }

#ifdef PV_USE_MPI
   // broadcast location information
   // TODO - IMPORTANT - WHICH OF THESE IS CORRECT
   //MPI_Bcast(locBuf, 1+locSize, MPI_INT, 0, comm->communicator());
   MPI_Bcast(locBuf, locSize, MPI_INT, 0, comm->communicator());
#endif

   copyFromLocBuffer(locBuf, loc);

   // fix up layer indices
   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;

   return status;
}

int getImageInfoGDAL(const char * filename, PV::Communicator * comm, PVLayerLoc * loc)
{
   int status = 0;

#ifdef PV_USE_GDAL
   const int locSize = sizeof(PVLayerLoc) / sizeof(int);
   int locBuf[locSize];

   // LayerLoc should contain 12 ints
   assert(locSize == 12);

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           comm->commRank(), nxProcs, nyProcs, icRow, icCol);
#endif

   if (comm->commRank() == 0) {
      GDALAllRegister();

      GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
      if (dataset == NULL) return 1;

      int xImageSize = dataset->GetRasterXSize();
      int yImageSize = dataset->GetRasterYSize();

      loc->nf = dataset->GetRasterCount();

      // calculate local layer size

      int nx = xImageSize / nxProcs;
      int ny = yImageSize / nyProcs;

      loc->nx = nx;
      loc->ny = ny;

      loc->nxGlobal = nxProcs * nx;
      loc->nyGlobal = nyProcs * ny;

      copyToLocBuffer(locBuf, loc);

      GDALClose(dataset);
   }

#ifdef PV_USE_MPI
   // broadcast location information
   MPI_Bcast(locBuf, 1+locSize, MPI_INT, 0, comm->communicator());
#endif

   copyFromLocBuffer(locBuf, loc);

   // fix up layer indices
   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   return status;
}


int gatherImageFile(const char * filename,
                    PV::Communicator * comm, const PVLayerLoc * loc, pvdata_t * pvdata_buf){
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
   int status = gatherImageFile(filename, comm, loc, char_buf);
   free(char_buf);
   return status;
}

int gatherImageFile(const char * filename,
                    PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return gatherImageFilePVP(filename, comm, loc, buf);
   }
   return gatherImageFileGDAL(filename, comm, loc, buf);
}

int gatherImageFilePVP(const char * filename,
                       PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf)
{
   int status = 0;
   // const int maxBands = 3;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   // assert(numBands <= maxBands);

   const int nxny     = nx * ny;
   const int numItems = nxny * numBands;

#ifdef PV_USE_MPI
   const int tag = PVP_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int dest = 0;
      MPI_Send(buf, numItems, MPI_BYTE, dest, tag, mpi_comm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: sent to 0, nx==%d ny==%d size==%d\n",
              comm->commRank(), nx, ny, nx*ny);
#endif
#endif // PV_USE_MPI
   }
   else {
      int params[NUM_PAR_BYTE_PARAMS];

      const int numParams  = NUM_PAR_BYTE_PARAMS;
      const int headerSize = numParams * sizeof(int);
      const int recordSize = numItems * sizeof(unsigned char);

      FILE * fp = fopen(filename, "wb");

      params[INDEX_HEADER_SIZE] = headerSize;
      params[INDEX_NUM_PARAMS]  = numParams;
      params[INDEX_FILE_TYPE]   = PVP_FILE_TYPE;
      params[INDEX_NX]          = loc->nx;
      params[INDEX_NY]          = loc->ny;
      params[INDEX_NF]          = loc->nf;
      params[INDEX_NUM_RECORDS] = nxProcs * nyProcs;
      params[INDEX_RECORD_SIZE] = recordSize;
      params[INDEX_DATA_SIZE]   = sizeof(unsigned char);
      params[INDEX_DATA_TYPE]   = PV_BYTE_TYPE;
      params[INDEX_NX_PROCS]    = nxProcs;
      params[INDEX_NY_PROCS]    = nyProcs;
      params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
      params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
      params[INDEX_KX0]         = loc->kx0;
      params[INDEX_KY0]         = loc->ky0;
      params[INDEX_NB]          = loc->nb;
      params[INDEX_NBANDS]      = loc->nf;

      int numWrite = fwrite(params, sizeof(int), numParams, fp);
      assert(numWrite == numParams);

      // write local image portion
      fseek(fp, (long) headerSize, SEEK_SET);
      numWrite = fwrite(buf, sizeof(unsigned char), numItems, fp);
      assert(numWrite == numItems);

#ifdef PV_USE_MPI
      int src = -1;
      unsigned char * tmp = (unsigned char *) malloc(numItems * sizeof(unsigned char));
      assert(tmp != NULL);
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: gather: receiving from %d nx==%d ny==%d numItems==%d\n",
                    comm->commRank(), src, nx, ny, numItems);
#endif
            MPI_Recv(tmp, numItems, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            long offset = headerSize + src * recordSize;
            fseek(fp, offset, SEEK_SET);
            numWrite = fwrite(tmp, sizeof(unsigned char), numItems, fp);
            assert(numWrite == numItems);
         }
      }
      free(tmp);
#endif // PV_USE_MPI

      status = fclose(fp);
   }

   return status;
}

int gatherImageFileGDAL(const char * filename,
                        PV::Communicator * comm, const PVLayerLoc * loc, unsigned char * buf)
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
   // assert(numBands <= maxBands);

   const int nxny = nx * ny;

#ifdef PV_USE_MPI
   const int tag = 14;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int dest = 0;

      for (int b = 0; b < numBands; b++) {
         MPI_Send(&buf[b*nxny], nxny, MPI_BYTE, dest, tag, mpi_comm);
      }
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: sent to 0, nx==%d ny==%d size==%d\n",
              comm->commRank(), nx, ny, nxny);
#endif
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
#endif
      }

      // GDALRasterBand * band[maxBands];

      assert(numBands <= dataset->GetRasterCount());

      GDALRasterBand ** band = (GDALRasterBand **) malloc( numBands * sizeof(GDALRasterBand *) );

      for (int b = 0; b < numBands; b++) {
         band[b] = dataset->GetRasterBand(b+1);
      }

      // write local image portion
      for (int b = 0; b < numBands; b++) {
         band[b]->RasterIO(GF_Write, 0, 0, nx, ny,
                           &buf[b*nxny], nx, ny, GDT_Byte, 0, 0);
      }

#ifdef PV_USE_MPI
      int src = -1;
      unsigned char * tmp = (unsigned char *) malloc(nxny * sizeof(unsigned char));
      assert(tmp != NULL);
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
#endif
            for (int b = 0; b < numBands; b++) {
               MPI_Recv(tmp, nxny, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
               band[b]->RasterIO(GF_Write, kx, ky, nx, ny,
                                 tmp, nx, ny, GDT_Byte, 0, 0);
            }
         }
      }
      free(tmp);
#endif // PV_USE_MPI
      GDALClose(dataset);
   }
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   return status;
}

int scatterImageFile(const char * filename, int xOffset, int yOffset,
                     PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      // TODO - add xOffset and yOffset parameters
      return scatterImageFilePVP(filename, comm, loc, buf);
   }
   return scatterImageFileGDAL(filename, xOffset, yOffset, comm, loc, buf);
}

int scatterImageFilePVP(const char * filename,
                        PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf)
{
   int status = 0;
   const int maxBands = 3;

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   assert(numBands <= maxBands);

   const int nxny     = nx * ny;
   const int numItems = nxny * numBands;

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int src = 0;
      const int tag = PVP_FILE_TYPE;
      const MPI_Comm mpi_comm = comm->communicator();

      MPI_Recv(buf, numItems, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatter: received from 0, nx==%d ny==%d numItems==%d\n",
              comm->commRank(), nx, ny, numItems);
#endif
#endif // PV_USE_MPI
   }
   else {
      int params[NUM_PAR_BYTE_PARAMS];
      int numParams, numRead, type, nxIn, nyIn, nfIn;

      FILE * fp = pv_open_binary(filename, &numParams, &type, &nxIn, &nyIn, &nfIn);
      assert(fp != NULL);
      assert(numParams == NUM_PAR_BYTE_PARAMS);
      assert(type      == PVP_FILE_TYPE);

      status = pv_read_binary_params(fp, numParams, params);
      assert(status == numParams);

      const size_t headerSize = (size_t) params[INDEX_HEADER_SIZE];

      const int numRecords = params[INDEX_NUM_RECORDS];
      const int dataSize = params[INDEX_DATA_SIZE];
      const int dataType = params[INDEX_DATA_TYPE];
      const int nxProcs  = params[INDEX_NX_PROCS];
      const int nyProcs  = params[INDEX_NY_PROCS];

      loc->nx       = params[INDEX_NX];
      loc->ny       = params[INDEX_NY];
      loc->nxGlobal = params[INDEX_NX_GLOBAL];
      loc->nyGlobal = params[INDEX_NY_GLOBAL];
      loc->kx0      = params[INDEX_KX0];
      loc->ky0      = params[INDEX_KY0];
      loc->nb       = params[INDEX_NB];
      loc->nf       = params[INDEX_NF];

      assert(dataSize == 1);
      assert(dataType == PV_BYTE_TYPE);
      assert(nxProcs == comm->numCommColumns());
      assert(nyProcs == comm->numCommRows());
      assert(numRecords == comm->commSize());

#ifdef PV_USE_MPI
      int dest = -1;
      const int tag = PVP_FILE_TYPE;
      const MPI_Comm mpi_comm = comm->communicator();
      const int recordSize = numItems * sizeof(unsigned char);

      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;

#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: scatter: sending to %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), dest, nx, ny, nx*ny,
                    nx*ny*comm->commSize());
#endif
            long offset = headerSize + dest * recordSize;
            fseek(fp, offset, SEEK_SET);
            numRead = fread(buf, sizeof(unsigned char), numItems, fp);
            assert(numRead == numItems);
            MPI_Send(buf, numItems, MPI_BYTE, dest, tag, mpi_comm);
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      fseek(fp, (long) headerSize, SEEK_SET);
      numRead = fread(buf, sizeof(unsigned char), numItems, fp);
      assert(numRead == numItems);

      status = pv_close_binary(fp);
   }

   return status;
}

int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf)
{
   int status = 0;

#ifdef PV_USE_GDAL
   const int maxBands = 3;

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;
   assert(numBands <= maxBands);

   const int nxny = nx * ny;

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int numTotal = nxny * numBands;

      const int src = 0;
      const int tag = 13;
      const MPI_Comm mpi_comm = comm->communicator();

      for (int b = 0; b < numBands; b++) {
         MPI_Recv(&buf[b*nxny], numTotal, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
      }
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatter: received from 0, nx==%d ny==%d size==%d\n",
              comm->commRank(), nx, ny, numTotal);
#endif
#endif // PV_USE_MPI
   }
   else {
      GDALAllRegister();

      GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);

      int xImageSize = dataset->GetRasterXSize();
      int yImageSize = dataset->GetRasterYSize();

      int xTotalSize = nx * nxProcs;
      int yTotalSize = ny * nyProcs;

      if (xOffset + xTotalSize > xImageSize || yOffset + yTotalSize > yImageSize) {
         fprintf(stderr, "[ 0]: scatterImageFile: image size too small, "
                 "xTotalSize==%d xImageSize==%d yTotalSize==%d yImageSize==%d xOffset==%d yOffset==%d\n",
                 xTotalSize, xImageSize, yTotalSize, yImageSize, xOffset, yOffset);
         fprintf(stderr, "[ 0]: xSize==%d ySize==%d nxProcs==%d nyProcs==%d\n",
                 nx, ny, nxProcs, nyProcs);
         GDALClose(dataset);
         return -1;
      }

      GDALRasterBand * band[maxBands];

      assert(numBands <= dataset->GetRasterCount());

      for (int b = 0; b < numBands; b++) {
         band[b] = dataset->GetRasterBand(b+1);
      }

#ifdef PV_USE_MPI
      int dest = -1;
      const int tag = 13;
      const MPI_Comm mpi_comm = comm->communicator();

      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++dest == 0) continue;
            int kx = nx * px;
            int ky = ny * py;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: scatter: sending to %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), dest, nx, ny, nx*ny,
                    nx*ny*comm->commSize());
#endif
            for (int b = 0; b < numBands; b++) {
               band[b]->RasterIO(GF_Read, kx, ky, nx, ny,
                                 &buf[b*nxny], nx, ny, GDT_Byte, 0, 0);
               MPI_Send(&buf[b*nxny], nx*ny, MPI_BYTE, dest, tag, mpi_comm);
            }
         }
      }
#endif // PV_USE_MPI

      // get local image portion
      for (int b = 0; b < numBands; b++) {
         band[b]->RasterIO(GF_Read, xOffset, yOffset, nx, ny,
                           &buf[b*nxny], nx, ny, GDT_Byte, 0, 0);
      }
      GDALClose(dataset);
   }
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   return status;
}

int scatterImageBlocks(const char* filename,
                       PV::Communicator * comm, PVLayerLoc * loc, float * buf)
{
   int status = 0;
#ifdef UNIMPLEMENTED

   const MPI_Comm icComm = comm->communicator();

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();
   const int icCol  = comm->commColumn();
   const int icRow  = comm->commRow();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nxGlobal = loc->nxGlobal;
   const int nyBorder = loc->nyBorder;

   const int xSize = nx + 2 * nxGlobal;
   const int ySize = ny + 2 * nyBorder;

   if (icRank > 0) {
   }
   else {
      int nxBlocks, nyBlocks, nxBlockSize, nyBlockSize;
      int ixBlock, iyBlock;

      GDALAllRegister();

      GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
      GDALRasterBand * band = dataset->GetRasterBand(1);

      CPLAssert(band->GetRasterDataType() == GDT_Byte);

      band->GetBlockSize(&nxBlockSize, &nyBlockSize);
      nxBlocks = (band->GetXSize() + nxBlockSize - 1) / nxBlockSize;
      nyBlocks = (band->GetYSize() + nyBlockSize - 1) / nyBlockSize;

      GByte * data = (GByte *) CPLMalloc(nxBlockSize * nyBlockSize);

      fprintf(stderr, "[ 0]: nxBlockSize==%d nyBlockSize==%d"
              " nxBlocks==%d nyBlocks==%d\n",
              nxBlockSize, nyBlockSize, nxBlocks, nyBlocks);

      for (iyBlock = 0; iyBlock < nyBlocks; iyBlock++) {
         for (ixBlock = 0; ixBlock < nxBlocks; ixBlock++) {
            int nxValid, nyValid;
            band->ReadBlock(ixBlock, ixBlock, data);
         }
      }
   }
#endif

   return status;
}

/**
 * gather relevant portions of srcBuf on root process from all others
 *    NOTE: dstBuf is np times larger than srcBuf on root process,
 *          dstBuf may be NULL if not root process
 */
int gather(PV::Communicator * comm, const PVLayerLoc * loc,
           unsigned char * dstBuf, unsigned char * srcBuf)
{
   // TODO - fix this to work for features
   assert(loc->nf == 1);

   const int nxProcs = comm->numCommColumns();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int sx  = 1;
   const int sy  = nx;
   const int sxg = 1;
   const int syg = nx * nxProcs;

   const int icRank = comm->commRank();

   if (icRank == 0) {
      // copy local portion on root process
      //
      for (int ky = 0; ky < ny; ky++) {
         for (int kx = 0; kx < nx; kx++) {
            int kxg = kx;
            int kyg = ky;
            dstBuf[kxg*sxg + kyg*syg] = srcBuf[kx*sx + ky*sy];
         }
      }
   }

#ifdef PV_USE_MPI
   const int tag = 44;
   const int nxny = nx*ny;
   const MPI_Comm mpi_comm = comm->communicator();
   const int nyProcs = comm->numCommRows();

   if (icRank > 0) {
      const int dest = 0;
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: sending to %d nx==%d ny==%d\n",
              comm->commRank(), dest, nx, ny);
#endif
      // everyone sends to root process
      //
      MPI_Send(srcBuf, nxny, MPI_BYTE, dest, tag, mpi_comm);
   }
   else if (nxProcs * nyProcs > 1) {
      int src = -1;
      unsigned char * tmp = (unsigned char *) malloc(nxny * sizeof(unsigned char));
      assert(tmp != NULL);

      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
            int kxg0 = nx * px;
            int kyg0 = ny * py;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: gather: receiving from %d nx==%d"
                    " ny==%d kxg0==%d kyg0==%d size==%d\n",
                    comm->commRank(), src, nx, ny, kxg0, kyg0, nxny);
#endif
            MPI_Recv(tmp, nxny, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);

            // copy tmp buffer from remote src into proper location in global dst
            //
            for (int ky = 0; ky < ny; ky++) {
               for (int kx = 0; kx < nx; kx++) {
                  int kxg = kxg0 + kx;
                  int kyg = kyg0 + ky;
                  dstBuf[kxg*sxg + kyg*syg] = srcBuf[kx*sx + ky*sy];
               }
            }
         }
      }
      free(tmp);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: finished\n", comm->commRank());
#endif
   }
#endif // PV_USE_MPI

   return 0;
}

/**
 * scatter relevant portions of buf from root process to all others
 *    NOTE: buf is np times larger on root process
 */
int scatter(PV::Communicator * comm, PVLayerLoc * loc, unsigned char * buf)
{
   return -1;
}

/**
 * gather relevant portions of buf on root process from all others
 *    NOTE: buf is np times larger on root process
 */
int gather(PV::Communicator * comm, const PVLayerLoc * loc, float * buf)
{
   return -1;
}

/**
 * scatter relevant portions of buf from root process to all others
 *    NOTE: buf is np times larger on root process
 */
int scatter(PV::Communicator * comm, PVLayerLoc * loc, float * buf)
{
   return -1;
}

#ifdef PV_USE_GDAL
int writeWithBorders(const char * filename, const PVLayerLoc * loc, float * buf)
{
   int X = loc->nx + 2*loc->nb;
   int Y = loc->ny + 2*loc->nb;
   int B = loc->nf;

   GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");
   GDALDataset* layer_file = driver->Create(filename, X, Y, B, GDT_Byte, NULL);

   // TODO - add multiple raster bands
   GDALRasterBand * band = layer_file->GetRasterBand(1);

   band->RasterIO(GF_Write, 0, 0, X, Y, buf, X, Y, GDT_Float32, 0, 0);

   GDALClose(layer_file);

   return 0;
}
#endif // PV_USE_GDAL



