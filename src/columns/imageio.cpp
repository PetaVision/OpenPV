#include "Communicator.hpp"

#include <assert.h>
#include <gdal_priv.h>
#include <mpi.h>
#include <ogr_spatialref.h>

#undef DEBUG_OUTPUT

static void copyToLocBuffer(int buf[], LayerLoc * loc)
{
	buf[0] = loc->nx;
	buf[1] = loc->ny;
	buf[2] = loc->nxGlobal;
	buf[3] = loc->nyGlobal;
	buf[4] = loc->kx0;
	buf[5] = loc->ky0;
	buf[6] = loc->nPad;
	buf[7] = loc->nBands;
}

static void copyFromLocBuffer(int buf[], LayerLoc * loc)
{
	loc->nx       = buf[0];
	loc->ny       = buf[1];
	loc->nxGlobal = buf[2];
	loc->nyGlobal = buf[3];
	loc->kx0      = buf[4];
	loc->ky0      = buf[5];
	loc->nPad     = buf[6];
	loc->nBands   = buf[7];
}

/**
 * Calculates location information given processor distribution and the
 * size of the image.  
 *
 * @filename the name of the image file (in)
 * @ic the inter-column communicator (in)
 * @loc location information (inout) (loc->nx and loc->ny are out)
 */
int getImageInfo(const char* filename, PV::Communicator * comm, LayerLoc * loc)
{
   const int locSize = sizeof(LayerLoc) / sizeof(int);
   int locBuf[locSize];
   int err = 0;

   // LayerLoc should contain 8 ints
   assert(locSize == 8);

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

      loc->nBands = dataset->GetRasterCount();

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

   // broadcast location information
   MPI_Bcast(locBuf, 1+locSize, MPI_INT, 0, comm->communicator());

   copyFromLocBuffer(locBuf, loc);

   // fix up layer indices
   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;

   return err;
}

/**
 * @filename
 */
int scatterImageFile(const char * filename,
                     PV::Communicator * comm, LayerLoc * loc, float * buf)
{
   int err = 0;
   const int tag = 13;

   const MPI_Comm icComm = comm->communicator();

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   if (icRank > 0) {
      const int src = 0;
      MPI_Recv(buf, nx*ny, MPI_FLOAT, src, tag, icComm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: scatter: received from 0, nx==%d ny==%d size==%d\n",
              comm->commRank(), nx, ny, nx*ny);
#endif
   }
   else {
      GDALAllRegister();

      GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);

      int xImageSize = dataset->GetRasterXSize();
      int yImageSize = dataset->GetRasterYSize();

      int xTotalSize = nx * nxProcs;
      int yTotalSize = ny * nyProcs;

      if (xTotalSize > xImageSize || yTotalSize > yImageSize) {
         fprintf(stderr, "[ 0]: scatterImageFile: image size too small, "
                 "xTotalSize==%d xImageSize==%d yTotalSize==%d yImageSize==%d\n",
                 xTotalSize, xImageSize, yTotalSize, yImageSize);
         fprintf(stderr, "[ 0]: xSize==%d ySize==%d nxProcs==%d nyProcs==%d\n",
                 nx, ny, nxProcs, nyProcs);
         GDALClose(dataset);
         return -1;
      }

      // TODO - decide what to do about multiband images (color)
      // include each band
      GDALRasterBand * band = dataset->GetRasterBand(1);

      int dest = -1;
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
            band->RasterIO(GF_Read, kx, ky, nx, ny, 
                           buf, nx, ny, GDT_Float32, 0, 0);
            MPI_Send(buf, nx*ny, MPI_FLOAT, dest, tag, icComm);
         }
      }

      // get local image portion
      band->RasterIO(GF_Read, 0, 0, nx, ny,
                     buf, nx, ny, GDT_Float32, 0, 0);
      GDALClose(dataset);
   }

   return err;
}


/**
 * @filename
 */
int gatherImageFile(const char * filename,
                    PV::Communicator * comm, LayerLoc * loc, float * buf)
{
   int err = 0;
   const int tag = 14;

   const MPI_Comm icComm = comm->communicator();

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   const int nx = loc->nx;
   const int ny = loc->ny;

   //TODO - color
   const int numBands = loc->nBands;

   if (icRank > 0) {
      const int dest = 0;
      MPI_Send(buf, nx*ny, MPI_FLOAT, dest, tag, icComm);
#ifdef DEBUG_OUTPUT
      fprintf(stderr, "[%2d]: gather: sent to 0, nx==%d ny==%d size==%d\n",
              comm->commRank(), nx, ny, nx*ny);
#endif
   }
   else {
//#include "cpl_string.h"

      GDALAllRegister();

      char ** metadata;

      GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");

      if( driver == NULL )
          exit( 1 );

      metadata = driver->GetMetadata();
      if( CSLFetchBoolean( metadata, GDAL_DCAP_CREATE, FALSE ) ) {
          // printf("Driver %s supports Create() method.\n", "GTiff");
      }

      GDALDataset * dataset;
      char ** options = NULL;

      int xImageSize = nx * nxProcs;
      int yImageSize = ny * nyProcs;

      dataset = driver->Create(filename, xImageSize, yImageSize, numBands,
                                 GDT_Byte, options);

      if (dataset == NULL) {
          fprintf(stderr, "[%2d]: gather: failed to open file %s\n", comm->commRank(), filename);
      }
      else {
#ifdef DEBUG_OUTPUT
          fprintf(stderr, "[%2d]: gather: opened file %s\n", comm->commRank(), filename);
#endif
      }

      double adfGeoTransform[6] = { 444720, 30, 0, 3751320, 0, -30 };
      OGRSpatialReference oSRS;
      char *pszSRS_WKT = NULL;

      dataset->SetGeoTransform( adfGeoTransform );

      oSRS.SetUTM( 11, TRUE );
      oSRS.SetWellKnownGeogCS( "NAD27" );
      oSRS.exportToWkt( &pszSRS_WKT );
      dataset->SetProjection( pszSRS_WKT );
      CPLFree( pszSRS_WKT );

      // TODO - decide what to do about multiband images (color)
      // include each band
      GDALRasterBand * band = dataset->GetRasterBand(1);

      // write local image portion
      band->RasterIO(GF_Write, 0, 0, nx, ny,
                     buf, nx, ny, GDT_Float32, 0, 0);

      int src = -1;
      for (int py = 0; py < nyProcs; py++) {
         for (int px = 0; px < nxProcs; px++) {
            if (++src == 0) continue;
            int kx = nx * px;
            int ky = ny * py;
#ifdef DEBUG_OUTPUT
            fprintf(stderr, "[%2d]: gather: receiving from %d xSize==%d"
                    " ySize==%d size==%d total==%d\n",
                    comm->commRank(), src, nx, ny, nx*ny,
                    nx*ny*comm->commSize());
#endif
            MPI_Recv(buf, nx*ny, MPI_FLOAT, src, tag, icComm, MPI_STATUS_IGNORE);
            band->RasterIO(GF_Write, kx, ky, nx, ny,
                           buf, nx, ny, GDT_Float32, 0, 0);
         }
      }
      GDALClose(dataset);
   }

   return err;
}

int scatterImageBlocks(const char* filename,
                       PV::Communicator * comm, LayerLoc * loc, float * buf)
{
   int err = 0;
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

   return err;
}

/**
 * gather relevant portions of buf on root process from all others
 *    NOTE: buf is np times larger on root process
 */
int gather (PV::Communicator * comm, LayerLoc * loc, float * buf)
{
   return -1;
}

/**
 * scatter relevant portions of buf from root process to all others
 *    NOTE: buf is np times larger on root process
 */
int scatter(PV::Communicator * comm, LayerLoc * loc, float * buf)
{
   return -1;
}

int writeWithBorders(const char * filename, LayerLoc * loc, float * buf)
{
   int X = loc->nx + 2 * loc->nPad;
   int Y = loc->ny + 2 * loc->nPad;
   int B = loc->nBands;

   GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");
   GDALDataset* layer_file = driver->Create(filename, X, Y, B, GDT_Byte, NULL);

   // TODO - add multiple raster bands
   GDALRasterBand * band = layer_file->GetRasterBand(1);

   band->RasterIO(GF_Write, 0, 0, X, Y, buf, X, Y, GDT_Float32, 0, 0);

   GDALClose(layer_file);

   return 0;
}

int image_test_main(int argc, char * argv[])
{
   int err = 0;
   LayerLoc loc;

   const char * filename = "/Users/rasmussn/Codes/PANN/"
                           "world.topo.200408.3x21600x21600.C2.jpg";
   
   PV::Communicator * comm = new PV::Communicator(&argc, &argv);

   loc.nx = 256;
   loc.ny = 256;
   loc.nPad = 16;

   getImageInfo(filename, comm, &loc);

   printf("[%d]: nx==%d ny==%d nxGlobal==%d nyGlobal==%d kx0==%d ky0==%d\n",
          comm->commRank(), (int)loc.nx, (int)loc.ny,
          (int)loc.nxGlobal, (int)loc.nyGlobal, (int)loc.kx0, (int)loc.ky0);

   int xSize = loc.nx  + 2 * loc.nPad;
   int ySize = loc.ny  + 2 * loc.nPad;

   float * buf = new float[xSize*ySize];

   //   err = scatterImageFile(filename, ic, &loc, buf);
   //   if (err) exit(err);

   err = scatterImageBlocks(filename, comm, &loc, buf);
   if (err) exit(err);

   delete buf;
   delete comm;

   return err;
}
