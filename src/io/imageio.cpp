#include "imageio.hpp"
#include "io.hpp"
#include "fileio.hpp"
#include "utils/PVLog.hpp"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#undef DEBUG_OUTPUT

int getFileType(const char * filename)
{
   const char * ext = strrchr(filename, '.');
   if (ext && strcmp(ext, ".pvp") == 0) {
      return PVP_FILE_TYPE;
   }
   return 0;
}

int getImageInfoPVP(const char * filename, PV::Communicator * comm, PVLayerLoc * loc)
{
   int status = 0;

   const int icCol = comm->commColumn();
   const int icRow = comm->commRow();

#ifdef DEBUG_OUTPUT
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   pvDebug().printf("[%2d]: nxProcs==%d nyProcs==%d icRow==%d icCol==%d\n",
           comm->commRank(), nxProcs, nyProcs, icRow, icCol);
#endif // DEBUG_OUTPUT

   PV_Stream * pvstream = NULL;
   if (comm->commRank()==0) { pvstream = PV::PV_fopen(filename, "rb", false/*verifyWrites*/); }
   int numParams = NUM_PAR_BYTE_PARAMS;
   int params[numParams];
   pvp_read_header(pvstream, comm, params, &numParams);
   PV::PV_fclose(pvstream); pvstream = NULL;

   assert(numParams == NUM_PAR_BYTE_PARAMS);

   const int dataSize = params[INDEX_DATA_SIZE];
   const int dataType = params[INDEX_DATA_TYPE];

   loc->nx       = params[INDEX_NX];
   loc->ny       = params[INDEX_NY];
   loc->nxGlobal = params[INDEX_NX_GLOBAL];
   loc->nyGlobal = params[INDEX_NY_GLOBAL];
   loc->kx0      = params[INDEX_KX0];
   loc->ky0      = params[INDEX_KY0];
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
   return gatherImageFilePVP(filename, comm, loc, buf, verifyWrites);
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
         pvError().printf("gatherImageFilePVP error opening \"%s\" for writing.\n", filename);
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
         pvError().printf("gatherImageFilePVP error writing the header.  fwrite called with %d parameters; %d were written.\n", numParams, numWrite);
      }
   }
   status = gatherActivity(pvstream, comm, rootproc, buf, loc, false/*extended*/);
   // buf is a nonextended buffer.  Image layers copy buf into the extended data buffer by calling Image::copyFromInteriorBuffer
   if (rank==rootproc) {
      PV::PV_fclose(pvstream); pvstream=NULL;
   }
   return status;
}


