#include "imageio.hpp"
#include "io.h"
#include "fileio.hpp"

#include <assert.h>
#include <string.h>

#ifdef PV_USE_GDAL
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
int getImageInfo(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return getImageInfoPVP(filename, comm, loc, colorbandtypes);
   }
   return getImageInfoGDAL(filename, comm, loc, colorbandtypes);
}

int getImageInfoPVP(const char * filename, PV::Communicator * comm, PVLayerLoc * loc, GDALColorInterp ** colorbandtypes)
{
   const int locSize = sizeof(PVLayerLoc) / sizeof(int);
   int locBuf[locSize];
   int status = 0;

   // LayerLoc should contain 12 ints
   assert(locSize == 12);

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   fprintf(stderr, "[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           comm->commRank(), nxProcs, nyProcs, icRow, icCol);
#endif // DEBUG_OUTPUT

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

//      assert(dataSize == 1);
      assert( (dataType == PV_BYTE_TYPE && dataSize == 1) || (dataType == PV_FLOAT_TYPE && dataSize == 4));


      copyToLocBuffer(locBuf, loc);
   }

#ifdef PV_USE_MPI
   // broadcast location information
   MPI_Bcast(locBuf, locSize, MPI_INT, 0, comm->communicator());
#endif // PV_USE_MPI

   copyFromLocBuffer(locBuf, loc);

   // fix up layer indices
   loc->kx0 = loc->nx * icCol;
   loc->ky0 = loc->ny * icRow;

   return status;
}

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

#ifdef PV_USE_GDAL
   const int locSize = sizeof(PVLayerLoc) / sizeof(int) + 1; // The extra 1 is for the status of the OpenGDAL call
   // LayerLoc should contain 12 ints, so locSize should be 13.
   assert(locSize == 13);

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

         int nx = xImageSize / nxProcs;
         int ny = yImageSize / nyProcs;

         loc->nx = nx;
         loc->ny = ny;

         loc->nxGlobal = nxProcs * nx;
         loc->nyGlobal = nyProcs * ny;

         locBuf[0] = PV_SUCCESS;
         copyToLocBuffer(&locBuf[1], loc);

         GDALClose(dataset);
      }
      else {
         locBuf[0] = PV_FAILURE;
         memset(&locBuf[1], 0, locSize*sizeof(int));
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
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   FILE * fp = NULL;
   if (rank==rootproc) {
      fp = fopen(filename, "wb");
      if (fp==NULL) {
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
      params[INDEX_NB]          = loc->nb;
      params[INDEX_NBANDS]      = 1;

      int numWrite = PV::PV_fwrite(params, sizeof(int), numParams, fp);
      if (numWrite != numParams) {
         fprintf(stderr, "gatherImageFilePVP error writing the header.  fwrite called with %d parameters; %d were written.\n", numParams, numWrite);
         abort();
      }
   }
   status = gatherActivity(fp, comm, rootproc, buf, loc, false/*extended*/);
   // buf is a nonextended buffer.  Image layers copy buf into the extended data buffer by calling Image::copyFromInteriorBuffer
   if (rank==rootproc) {
      fclose(fp); fp=NULL;
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

#ifdef PV_USE_MPI
   const int nxnynf = nx * ny * numBands;

   const int tag = 14;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   if (icRank > 0) {
#ifdef PV_USE_MPI
      const int dest = 0;

      MPI_Send(buf, nxnynf, MPI_BYTE, dest, tag, mpi_comm);
      // for (int b = 0; b < numBands; b++) {
      //   MPI_Send(&buf[b*nxny], nxny, MPI_BYTE, dest, tag, mpi_comm);
      // }
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
#else
   fprintf(stderr, GDAL_CONFIG_ERR_STR);
   exit(1);
#endif // PV_USE_GDAL

   return status;
}

int scatterImageFile(const char * filename, int xOffset, int yOffset,
                     PV::Communicator * comm, const PVLayerLoc * loc, float * buf)
{
   if (getFileType(filename) == PVP_FILE_TYPE) {
      return scatterImageFilePVP(filename, xOffset, yOffset, comm, loc, buf);
   }
   return scatterImageFileGDAL(filename, xOffset, yOffset, comm, loc, buf);
}

int scatterImageFilePVP(const char * filename, int xOffset, int yOffset,
                        PV::Communicator * comm, const PVLayerLoc * loc, float * buf)
{
   // Read a PVP file and scatter it to the multiple processes.
   int status = PV_SUCCESS;

   int rootproc = 0;
   int rank = comm->commRank();

   FILE * fp = NULL;
   if (rank==rootproc) {
      int numParams = 0;
      int filetype = 0;
      int nx = 0;
      int ny = 0;
      int nf = 0;
      fp = pv_open_binary(filename, &numParams, &filetype, &nx, &ny, &nf);
      if (fp==NULL) {
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\" for reading.\n", filename);
         abort();
      }
      if (numParams < MIN_BIN_PARAMS) {
         fprintf(stderr, "scatterImageFilePVP error in header of \"%s\": number of parameters is too small.\n", filename);
         abort();
      }
      rewind(fp);
      int params[numParams];
      int paramsread = fread(params, sizeof(int), numParams, fp);
      if (paramsread != numParams) {
         fprintf(stderr, "scatterImageFilePVP error reading header of \"%s\".\n", filename);
         abort();
      }
      PVLayerLoc fileloc;
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
      if (fileloc.nx != fileloc.nxGlobal || fileloc.ny != fileloc.nyGlobal ||
          nxProcs != 1 || nyProcs != 1 ||
          fileloc.kx0 != 0 || fileloc.ky0 != 0) {
          fprintf(stderr, "File \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
          abort();
      }
      bool spiking = false;
      double timed = 0.0;
      switch (filetype) {
      case PVP_FILE_TYPE:
         break;
      case PVP_ACT_FILE_TYPE:
         spiking = true;
         fread(&timed, sizeof(double), 1, fp);
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": Reading spiking PVP files into an Image layer hasn't been implemented yet.\n", filename);
         abort();
         break;
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         fread(&timed, sizeof(double), 1, fp);
         status = PV_SUCCESS;
         break;
      case PVP_WGT_FILE_TYPE:
      case PVP_KERNEL_FILE_TYPE:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": file is a weight file, not an image file.\n", filename);
         break;
      default:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": filetype %d is unrecognized.\n", filename ,filetype);
         status = PV_FAILURE;
         break;
      }
      scatterActivity(fp, comm, rootproc, buf, loc, false/*extended*/, &fileloc, xOffset, yOffset);
      // buf is a nonextended layer.  Image layers copy the extended buffer data into buf by calling Image::copyToInteriorBuffer
      fclose(fp); fp = NULL;
   }
   else {
      scatterActivity(fp, comm, rootproc, buf, loc, false/*extended*/, NULL, xOffset, yOffset);
   }
   return status;
}

#ifdef OBSOLETE // Marked obsolete Dec 10, 2012, during reworking of PVP files to be MPI-independent.
int windowFromPVPBuffer(int startx, int starty, int nx, int ny, int * params, float * destbuf, char * pvpbuffer, const char * filename) {
   // Extracts a window from the data portion of a PVP file.
   // The params from the PVP file should be in params[].  The data should already be loaded into pvpbuffer,
   // with the bytes exactly as they appear in the file, not including the header.  The routine accounts for
   // fragmentation of the window across MPI processes.
   // The nx-by-ny portion beginning at row starty, column startx is loaded into destbuf.  If params[INDEX_DATA_TYPE]
   // is byte, the data is converted to float as it is stored into destbuf.  All params[INDEX_NF] features are
   // copied; there is no way to select individual features.
   for( int y=0; y<ny; y++) {
      for( int x=0; x<nx; x++) {
         int idx_in_buf = kIndex(x, y, 0, nx, ny, params[INDEX_NF]);
         int xProc = (startx + x)/params[INDEX_NX];
         int yProc = (starty + y)/params[INDEX_NY];
         int kProc = rankFromRowAndColumn(yProc, xProc, params[INDEX_NY_PROCS], params[INDEX_NX_PROCS]);
         if( kProc < 0 ) {
            fprintf(stderr, "windowFromPVPBuffer: Requested x and y coordinates out of bounds of PVP file %s. x=%d, y=%d, xOffset=%d, yOffset=%d\n", filename, x, y, startx, starty);
            abort();
         }
         int idx_in_proc = kIndex(x + startx - xProc*params[INDEX_NX],
                                  y + starty - yProc*params[INDEX_NY],
                                  0,
                                  params[INDEX_NX], params[INDEX_NY], params[INDEX_NF]);
         long offset = kProc * params[INDEX_RECORD_SIZE] + idx_in_proc*params[INDEX_DATA_SIZE];
         if( params[INDEX_DATA_TYPE] == PV_BYTE_TYPE ) {
            char * filebufstart = &pvpbuffer[offset];
            for( int f=0; f<params[INDEX_NF]; f++ ) {
               destbuf[idx_in_buf+f] = (float) filebufstart[idx_in_proc+f];
               destbuf[idx_in_buf+f] *= 1.0/255.0;
            }
         }
         else if( params[INDEX_DATA_TYPE] == PV_FLOAT_TYPE ) {
            float * srcbuf = (float *) (&pvpbuffer[offset]);
            memcpy(&destbuf[idx_in_buf], srcbuf, sizeof(float)*params[INDEX_NF]);
         }
         else {
            fprintf(stderr, "windowFromPVPBuffer: pvp file %s does not have a well-formed header.\n", filename);
            fprintf(stderr, "Parameters (INDEX_DATA_SIZE,INDEX_DATA_TYPE) must be set to either (1,%d) (compressed) or (4,%d) (uncompressed).\n", PV_BYTE_TYPE, PV_FLOAT_TYPE);
            abort();
         }
      }
   }
   return PV_SUCCESS;
}
#endif // OBSOLETE

int scatterImageFileGDAL(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf)
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

      if (xOffset + xTotalSize > xImageSize || yOffset + yTotalSize > yImageSize) {
         fprintf(stderr, "[ 0]: scatterImageFile: image size too small, "
                 "xTotalSize==%d xImageSize==%d yTotalSize==%d yImageSize==%d xOffset==%d yOffset==%d\n",
                 xTotalSize, xImageSize, yTotalSize, yImageSize, xOffset, yOffset);
         fprintf(stderr, "[ 0]: xSize==%d ySize==%d nxProcs==%d nyProcs==%d\n",
                 nx, ny, nxProcs, nyProcs);
         GDALClose(dataset);
         return -1;
      }

      assert(numBands == 1 || numBands == bandsInFile);

#ifdef PV_USE_MPI
      int dest = -1;
      const int tag = 13;

      for( dest = 1; dest < nyProcs*nxProcs; dest++ ) {
         int kx = nx * columnFromRank(dest, nyProcs, nxProcs);
         int ky = ny * rowFromRank(dest, nyProcs, nxProcs);
         dataset->RasterIO(GF_Read, kx+xOffset, ky+yOffset, nx, ny, buf,
                           nx, ny, GDT_Float32, bandsInFile, NULL,
                           bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));
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
      dataset->RasterIO(GF_Read, xOffset, yOffset, nx, ny, buf, nx, ny,
                        GDT_Float32, bandsInFile, NULL, bandsInFile*sizeof(float), bandsInFile*nx*sizeof(float), sizeof(float));
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

#ifdef OBSOLETE // Marked obsolete Dec 10, 2012.  No one calls either gather or writeWithBorders and they have TODO's that indicate that they're broken.

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
#endif // DEBUG_OUTPUT
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
#endif // DEBUG_OUTPUT
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
#endif // DEBUG_OUTPUT
   }
#endif // PV_USE_MPI

   return 0;
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

#endif // OBSOLETE


