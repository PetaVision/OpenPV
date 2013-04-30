/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "io.h"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI
#include "../include/PVLayerLoc.h"
#include "../columns/Communicator.hpp"
#include "../arch/opencl/pv_uint4.h"

#include <unistd.h>

namespace PV {

void timeToParams(double time, void * params);
double timeFromParams(void * params);

size_t pv_sizeof(int datatype);

PV_Stream * PV_fopen(const char * path, const char * mode);
long int PV_ftell(PV_Stream * pvstream);
int PV_fseek(PV_Stream * pvstream, long int offset, int whence);
size_t PV_fwrite(const void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream);
int PV_fclose(PV_Stream * pvstream);
PV_Stream * PV_stdout();

PV_Stream * pvp_open_read_file(const char * filename, Communicator * comm);

PV_Stream * pvp_open_write_file(const char * filename, Communicator * comm, bool append);

int pvp_close_file(PV_Stream * pvstream, Communicator * comm);

int pvp_read_header(PV_Stream * pvstream, Communicator * comm, int * params, int * numParams);
int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams);
void read_header_err(const char * filename, Communicator * comm, int returned_num_params, int * params);
int pvp_write_header(PV_Stream * pvstream, Communicator * comm, int * params, int numParams);

// The pvp_write_header below will go away in favor of the pvp_write_header above.
int pvp_write_header(PV_Stream * pvstream, Communicator * comm, double time, const PVLayerLoc * loc,
                     int filetype, int datatype, int numbands,
                     bool extended, bool contiguous, unsigned int numParams, size_t localSize);

int * pvp_set_file_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_activity_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_weight_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches);
int * pvp_set_nonspiking_act_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands);
int * pvp_set_kernel_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches);
int * alloc_params(int numParams);
int set_weight_params(int * params, int nxp, int nyp, int nfp, float min, float max, int numPatches);

int pvp_read_time(PV_Stream * pvstream, Communicator * comm, int root_process, double * timed);

int writeActivity(PV_Stream * pvstream, Communicator * comm, double time, PVLayer * l);

int writeActivitySparse(PV_Stream * pvstream, Communicator * comm, double time, PVLayer * l);

int readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numArbors, int numPatches, const char * filename,
                Communicator * comm, double * timed, const PVLayerLoc * loc, bool * shmget_owner = NULL, bool shmget_flag = false);

int writeWeights(const char * filename, Communicator * comm, double timed, bool append,
                 const PVLayerLoc * loc, int nxp, int nyp, int nfp, float minVal, float maxVal,
                 PVPatch *** patches, pvdata_t ** dataStart, int numPatches, int numArbors, bool compress=true, int file_type=PVP_WGT_FILE_TYPE);

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);

int writeRandState(const char * filename, Communicator * comm, uint4 * randState, const PVLayerLoc * loc);

int readRandState(const char * filename, Communicator * comm, uint4 * randState, const PVLayerLoc * loc);

template <typename T> int gatherActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended) {
   // In MPI when this process is called, all processes must call it.
   // Only the root process uses the file pointer.
   int status = PV_SUCCESS;

   int numLocalNeurons = layerLoc->nx * layerLoc->ny * layerLoc->nf;

   int xLineStart = 0;
   int yLineStart = 0;
   int xBufSize = layerLoc->nx;
   int yBufSize = layerLoc->ny;
   int nb = 0;
   if (extended) {
      nb = layerLoc->nb;
      xLineStart = nb;
      yLineStart = nb;
      xBufSize += 2*nb;
      yBufSize += 2*nb;
   }
   int linesize = layerLoc->nx*layerLoc->nf; // All values across x and f for a specific y are contiguous; do a single write for each y.
   size_t datasize = sizeof(T);
   // read into a temporary buffer since buffer may be extended but the file only contains the restricted part.
   T * temp_buffer = (T *) calloc(numLocalNeurons, datasize);
   if (temp_buffer==NULL) {
      fprintf(stderr, "scatterActivity unable to allocate memory for temp_buffer.\n");
      status = PV_FAILURE;
      abort();
   }
#ifdef PV_USE_MPI
   int rank = comm->commRank();
   if (rank==rootproc) {
      if (pvstream == NULL) {
         fprintf(stderr, "gatherActivity error: file pointer on root process is null.\n");
         status = PV_FAILURE;
         abort();
      }
      long startpos = PV_ftell(pvstream);
      if (startpos == -1) {
         fprintf(stderr, "gatherActivity error when getting file position: %s\n", strerror(errno));
         status = PV_FAILURE;
         abort();
      }
      // Write zeroes to make sure the file is big enough since we'll write nonsequentially under MPI.  This may not be necessary.
      int comm_size = comm->commSize();
      for (int r=0; r<comm_size; r++) {
         int numwritten = PV_fwrite(temp_buffer, datasize, numLocalNeurons, pvstream);
         if (numwritten != numLocalNeurons) {
            fprintf(stderr, "gatherActivity error when writing: number of bytes attempted %d, number written %d\n", numwritten, numLocalNeurons);
            status = PV_FAILURE;
            abort();
         }
      }
      int fseekstatus = PV_fseek(pvstream, startpos, SEEK_SET);
      if (fseekstatus != 0) {
         fprintf(stderr, "gatherActivity error when setting file position: %s\n", strerror(errno));
         status = PV_FAILURE;
         abort();
      }
      for (int r=0; r<comm_size; r++) {
         if (r==rootproc) {
            if (extended) {
               for (int y=0; y<layerLoc->ny; y++) {
                  int k_extended = kIndex(nb, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
                  int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
                  memcpy(&temp_buffer[k_restricted], &buffer[k_extended], datasize*linesize);
               }
            }
            else {
               memcpy(temp_buffer, buffer, (size_t) numLocalNeurons*datasize);
            }
         }
         else {
            MPI_Recv(temp_buffer, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
         }
         // Data to be written is in temp_buffer, which is nonextend.
         for (int y=0; y<layerLoc->ny; y++) {
            int ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int k_local = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            int k_global = kIndex(kx0, y+ky0, 0, layerLoc->nxGlobal, layerLoc->nyGlobal, layerLoc->nf);
            int fseekstatus = PV_fseek(pvstream, startpos + k_global*datasize, SEEK_SET);
            if (fseekstatus == 0) {
               int numwritten = PV_fwrite(&temp_buffer[k_local], datasize, linesize, pvstream);
               if (numwritten != linesize) {
                  fprintf(stderr, "gatherActivity error when writing: number of bytes attempted %d, number written %d\n", numwritten, numLocalNeurons);
                  status = PV_FAILURE;
               }
            }
            else {
               fprintf(stderr, "gatherActivity error when setting file position: %s\n", strerror(errno));
               status = PV_FAILURE;
               abort();
            }
         }
      }
      PV_fseek(pvstream, startpos+numLocalNeurons*datasize*comm_size, SEEK_SET);
   }
   else {
      if (nb>0) {
         // temp_buffer is a restricted buffer, but if extended is true, buffer is an extended buffer.
         for (int y=0; y<layerLoc->ny; y++) {
            int k_extended = kIndex(nb, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
            int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            memcpy(&temp_buffer[k_restricted], &buffer[k_extended], datasize*linesize);
         }
         MPI_Send(temp_buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
      else {
         MPI_Send(buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
   }
#endif // PV_USE_MPI
   free(temp_buffer); temp_buffer = NULL;
   return status;
}

template <typename T> int scatterActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc=NULL, int offsetX=0, int offsetY=0) {
   // In MPI when this process is called, all processes must call it.
   // Only the root process uses the file pointer fp or the file PVLayerLoc fileLoc.
   //
   // layerLoc refers to the PVLayerLoc of the layer being read into.
   // fileLoc refers to the PVLayerLoc of the file being read from.  They do not have to be the same.
   // The position (0,0) of the layer corresponds to (offsetX, offsetY) of the file.
   // fileLoc and layerLoc do not have to have the same nxGlobal or nyGlobal, but they must have the same nf.

   // Potential improvements:
   // Detect when you can do a single read of the whole block instead of layerLoc->ny smaller reads of one line each
   // If nb=0, don't need to allocate a temporary buffer; can just read into buffer.
   int status = PV_SUCCESS;

   int numLocalNeurons = layerLoc->nx * layerLoc->ny * layerLoc->nf;

   int xLineStart = 0;
   int yLineStart = 0;
   int xBufSize = layerLoc->nx;
   int yBufSize = layerLoc->ny;
   int nb = 0;
   if (extended) {
      nb = layerLoc->nb;
      xLineStart = nb;
      yLineStart = nb;
      xBufSize += 2*nb;
      yBufSize += 2*nb;
   }
   int linesize = layerLoc->nx * layerLoc->nf;
   size_t datasize = sizeof(T);
   // read into a temporary buffer since buffer may be extended but the file only contains the restricted part.
   T * temp_buffer = (T *) calloc(numLocalNeurons, datasize);
   if (temp_buffer==NULL) {
      fprintf(stderr, "scatterActivity unable to allocate memory for temp_buffer.\n");
      status = PV_FAILURE;
      abort();
   }

#ifdef PV_USE_MPI
   int rank = comm->commRank();
   if (rank==rootproc) {
      if (pvstream == NULL) {
         fprintf(stderr, "scatterActivity error: file pointer on root process is null.\n");
         status = PV_FAILURE;
         abort();
      }
      long startpos = PV_ftell(pvstream);
      if (startpos == -1) {
         fprintf(stderr, "scatterActivity error when getting file position: %s\n", strerror(errno));
         status = PV_FAILURE;
         abort();
      }
      if (fileLoc==NULL) fileLoc = layerLoc;
      if (fileLoc->nf != layerLoc->nf) {
         fprintf(stderr, "scatterActivity error: layerLoc->nf and fileLoc->nf must be equal (they are %d and %d)\n", layerLoc->nf, fileLoc->nf);
         abort();
      }
      if (offsetX < 0 || offsetX + layerLoc->nxGlobal > fileLoc->nxGlobal ||
            offsetY < 0 || offsetY + layerLoc->nyGlobal > fileLoc->nyGlobal) {
         fprintf(stderr, "scatterActivity error: offset window does not completely fit inside image frame. This case has not been implemented yet.\n");
         abort();
      }
      int comm_size = comm->commSize();
      for (int r=0; r<comm_size; r++) {
         if (r==rootproc) continue; // Need to load root process last, or subsequent processes will clobber temp_buffer.
         for (int y=0; y<layerLoc->ny; y++) {
            int ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
            int k_inmemory = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            int k_infile = kIndex(offsetX+kx0, offsetY+ky0+y, 0, fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
            PV_fseek(pvstream, startpos + k_infile*(long) datasize, SEEK_SET);
            int numread = fread(&temp_buffer[k_inmemory], datasize, linesize, pvstream->fp);
            if (numread != linesize) {
               fprintf(stderr, "scatterActivity error when reading: number of bytes attempted %d, number written %d\n", numread, numLocalNeurons);
               abort();
            }
         }
         MPI_Send(temp_buffer, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
      }
      for (int y=0; y<layerLoc->ny; y++) {
         int ky0 = layerLoc->ny*rowFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         int kx0 = layerLoc->nx*columnFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         int k_inmemory = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
         int k_infile = kIndex(offsetX+kx0, offsetY+ky0+y, 0, fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
         PV_fseek(pvstream, startpos + k_infile*(long) datasize, SEEK_SET);
         int numread = fread(&temp_buffer[k_inmemory], datasize, linesize, pvstream->fp);
         if (numread != linesize) {
            fprintf(stderr, "scatterActivity error when reading: number of bytes attempted %d, number written %d\n", linesize, numread);
            abort();
         }
      }
      PV_fseek(pvstream, startpos+numLocalNeurons*datasize*comm_size, SEEK_SET);
   }
   else {
      MPI_Recv(temp_buffer, sizeof(uint4)*numLocalNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
   }
#else // PV_USE_MPI
   for (int y=0; y<layerLoc->ny; y++) {
      int k_inmemory = kIndex(xLineStart, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
      int k_infile = kIndex(offsetX, offsetY+y, 0, fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
      PV_fseek(pvstream, startpos + k_infile*(long) datasize, SEEK_SET);
      int numread = fread(&temp_buffer[k_inmemory], datasize, linesize, pvstream->fp);
      if (numread != linesize) {
         fprintf(stderr, "scatterActivity error when reading: number of bytes attempted %d, number written %d\n", numread, numLocalNeurons);
         abort();
      }
   }
#endif // PV_USE_MPI
   // At this point, each process has the data, as a restricted layer, in temp_buffer.
   // Each process now copies the data to buffer, which may be extended.
   if (nb>0) {
      for (int y=0; y<layerLoc->ny; y++) {
         int k_extended = kIndex(xLineStart, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
         int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
         memcpy(&buffer[k_extended], &temp_buffer[k_restricted], (size_t)linesize*datasize);
      }
   }
   else {
      memcpy(buffer, temp_buffer, (size_t) numLocalNeurons*datasize);
   }
   free(temp_buffer); temp_buffer = NULL;

   return status;
}

} // namespace PV

#endif /* FILEIO_HPP_ */
