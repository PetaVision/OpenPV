/*
 * fileio.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: Craig Rasmussen
 */

#include "fileio.hpp"
#include "../connections/weight_conversions.hpp"
#include "../layers/HyPerLayer.hpp"

#include <assert.h>
#include <iostream>

#undef DEBUG_OUTPUT

namespace PV {

// timeToParams and timeFromParams use memcpy instead of casting pointers
// because casting pointers violates strict aliasing.


void timeToParams(double time, void * params)
{
   memcpy(params, &time, sizeof(double));
}

double timeFromParams(void * params)
{
   double x;
   memcpy(&x, params, sizeof(double));
   return x;
}

size_t pv_sizeof(int datatype)
{
   if (datatype == PV_INT_TYPE) {
      return sizeof(int);
   }
   if (datatype == PV_FLOAT_TYPE) {
      return sizeof(float);
   }
   if (datatype == PV_BYTE_TYPE) {
      return sizeof(unsigned char);
   }
   if (datatype == PV_SPARSEVALUES_TYPE) {
      return sizeof(indexvaluepair);
   }

   // shouldn't arrive here
   assert(false);
   return 0;
}

/**
 * Returns the size of a patch when read or written. The size includes the size
 * of the (nxp,nyp) patch header.
 */
size_t pv_sizeof_patch(int count, int datatype)
{
   return ( 2*sizeof(unsigned short) + sizeof(unsigned int) + count*pv_sizeof(datatype) );
   // return ( 2*sizeof(unsigned short) + count*pv_sizeof(datatype) );
}

PV_Stream * PV_fopen(const char * path, const char * mode, bool verifyWrites) {
   if (mode==NULL) {
      pvErrorNoExit().printf("PV_fopen: mode argument must be a string (path was \"%s\").\n", path);
      errno = EINVAL;
      return NULL;
   }
   char * realPath = expandLeadingTilde(path);
   long filepos = 0L;
   long filelength = 0L;
   if (mode[0]=='r' || mode[0]=='a') {
      struct stat statbuf;
      int statstatus = stat(realPath, &statbuf);
      if (statstatus == 0) {
         filelength = (long) statbuf.st_size;
         if (mode[0]=='a') {
            filepos = filelength;
         }
      }
      else if (errno != ENOENT) {
         pvError().printf("PV_fopen: unable to stat \"%s\" with mode \"%s\": %s\n", realPath, mode, strerror(errno));
      }
   }
   int fopencounts = 0;
   PV_Stream * streampointer = NULL;
   FILE * fp = NULL;
   while (fp == NULL) {
      errno = 0;
      fp = fopen(realPath, mode);
      if (fp != NULL) break;
      fopencounts++;
      pvWarn().printf("fopen failure for \"%s\" on attempt %d: %s\n", realPath, fopencounts, strerror(errno));
      if (fopencounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fp == NULL) {
      pvErrorNoExit().printf("PV_fopen: exceeded MAX_FILESYSTEMCALL_TRIES = %d attempting to open \"%s\"\n", MAX_FILESYSTEMCALL_TRIES, realPath);
   }
   else {
      if (fopencounts>0) {
         pvWarn().printf("fopen succeeded for \"%s\" on attempt %d\n", realPath, fopencounts+1);
      }
      streampointer = (PV_Stream *) calloc(1, sizeof(PV_Stream));
      if (streampointer != NULL) {
         streampointer->name = strdup(realPath);
         streampointer->mode = strdup(mode);
         streampointer->fp = fp;
         streampointer->filepos = filepos;
         streampointer->filelength = filelength;
         streampointer->isfile = 1;
         streampointer->verifyWrites = verifyWrites;
      }
      else {
         pvErrorNoExit().printf("PV_fopen failure for \"%s\": %s\n", realPath, strerror(errno));
         fclose(fp);
      }
   }
   free(realPath);
   return streampointer;
}

int PV_stat(const char * path, struct stat * buf) {
   // Call stat library function, trying up to MAX_FILESYSTEMCALL_TRIES times if an error is returned.
   // If an error results on all MAX_FILESYSTEMCALL_TRIES times, returns -1 (the error return value) for stat()
   // and errno is the error of the last attempt.
   char * realPath = expandLeadingTilde(path);
   int attempt = 0;
   int retval = -1;
   while (retval != 0) {
      errno = 0;
      retval = stat(realPath, buf);
      if (retval == 0) break;
      attempt++;
      pvWarn().printf("stat() failure for \"%s\" on attempt %d: %s\n", path, attempt, strerror(errno));
      if (attempt < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (retval != 0) {
      pvErrorNoExit().printf("PV_stat exceeded MAX_FILESYSTEMCALL_TRIES = %d for \"%s\"\n", MAX_FILESYSTEMCALL_TRIES, path);
   }
   free(realPath);
   return retval;
}

long int PV_ftell_primitive(PV_Stream * pvstream) {
   // Calls ftell() and returns value ftell returns, but doesn't compare or change stream's fpos
   int ftellcounts = 0;
   long filepos = -1;
   while (filepos < 0) {
      errno = 0;
      filepos = ftell(pvstream->fp);
      if (filepos >= 0) break;
      ftellcounts++;
      pvWarn().printf("ftell failure for \"%s\" on attempt %d: %s\n", pvstream->name, ftellcounts, strerror(errno));
      if (ftellcounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (filepos<0) {
      pvErrorNoExit().printf("PV_ftell failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES);
   }
   else if (ftellcounts>0) {
      pvWarn().printf("PV_ftell succeeded for \"%s\" on attempt %d", pvstream->name, ftellcounts+1);
   }
   return filepos;
}

long int getPV_StreamFilepos(PV_Stream * pvstream) {
   return pvstream->filepos;
}

long int updatePV_StreamFilepos(PV_Stream * pvstream) {
   long int filepos = PV_ftell_primitive(pvstream);
   pvstream->filepos = filepos;
   return filepos;
}

// Use getPV_StreamFilepos instead of PV_ftell whenever possible, since NMC cluster's ftell is currently unreliable
long int PV_ftell(PV_Stream * pvstream) {
   long int filepos = PV_ftell_primitive(pvstream);
   if (pvstream->filepos != filepos)
   {
      pvWarn().printf("ftell for \"%s\" returned %ld instead of the expected %ld\n", pvstream->name, filepos, pvstream->filepos);
   }
   return filepos;
}

int PV_fseek(PV_Stream * pvstream, long offset, int whence) {
   int fseekcounts = 0;
   int fseekstatus = -1;
   while (fseekstatus != 0) {
      errno = 0;
      fseekstatus = fseek(pvstream->fp, offset, whence);
      if (fseekstatus==0) break;
      fseekcounts++;
      pvWarn().printf("fseek failure for \"%s\" on attempt %d: %s\n", pvstream->name, fseekcounts, strerror(errno));
      if (fseekcounts<MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fseekstatus!=0) {
      pvErrorNoExit().printf("PV_fseek failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES);
   }
   else if (fseekcounts>0) {
      pvWarn().printf("PV_fseek succeeded for \"%s\" on attempt %d\n", pvstream->name, fseekcounts+1);
   }
   if (pvstream->mode[0] != 'a') {
      switch(whence) {
      case SEEK_SET:
         pvstream->filepos = offset;
         break;
      case SEEK_CUR:
         pvstream->filepos += offset;
         break;
      case SEEK_END:
         pvstream->filepos = pvstream->filelength + offset;
         break;
      default:
         assert(0);
         break;
      }
   }
   return fseekstatus;
}

/**
 * @brief A wrapper for fwrite() with feedback for errors and the possibility of error recovery.
 * @detail The syntax and purpose of PV_fwrite follows that of the standard C function fwrite(), with the following changes.
 * The FILE* argument is replaced with a PV_Stream* pointer, and the additional argument verify (which defaults to true)
 * provides some error checking.
 *
 * The function calls fwrite().  If it gets an error, it tries again, up to 5 times (the number is controlled by
 * the preprocessor directive MAX_FILESYSTEMCALL_TRIES).  If it fails all 5 times, it fseeks to the position
 * it was in at the start of the call, and returns zero.  If it succeeds in any of the 5 times, it returns nitems, and
 * the file position is at the end of the written data.
 *
 * If verify is true and pvstream is a file (isfile is true), then after writing, the file is opened for reading
 * and the size*nitems characters are compared to the write buffer.  If there is an error reading the data back
 * or the data read back does not match the data written, the function returns zero and the fseek() is called
 * to restore the file to the position it was in at the start of the call.
 *
 * NOTE: the purpose of this wrapper is to provide some attempts at recovery if a file system is imperfect (such as the one we've struggled with).
 * We hope that a successful return value indicates actual success and that the feedback provided by failures prove helpful.
 * However, the function cannot guarantee recovery from errors.
 */
size_t PV_fwrite(const void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream) {
   assert(ferror(pvstream->fp)==0);
   int fwritecounts = 0;
   size_t writesize = nitems*size;
   size_t charswritten = (size_t) 0;
   const char * RESTRICT curptr = (const char * RESTRICT) ptr;
   long int fpos = pvstream->filepos; // PV_ftell(pvstream);
   if (fpos<0) {
      pvError().printf("PV_fwrite error: unable to determine file position of \"%s\".  Fatal error\n", pvstream->name);
   }
   long int ftellresult = ftell(pvstream->fp);
   if(pvstream->isfile && fpos != ftellresult) {
      pvError().printf("PV_fwrite error for \"%s\": fpos = %ld but ftell() returned %ld\n", pvstream->name, fpos, ftellresult);
      exit(EXIT_FAILURE);
   }
   bool hasfailed = false;
   for (int fwritecounts=1; fwritecounts<=MAX_FILESYSTEMCALL_TRIES; fwritecounts++) {
      charswritten = fwrite(ptr, 1UL, writesize, pvstream->fp);
      if (charswritten == writesize) {
         if (hasfailed) {
            clearerr(pvstream->fp);
            pvWarn().printf("fwrite succeeded for \"%s\" on attempt %d.\n", pvstream->name, fwritecounts);
         }
         break;
      }
      else {
         hasfailed = true;
         pvWarn(fwriteFailure);
         fwriteFailure.printf("fwrite failure for \"%s\" on attempt %d.  Return value %zu instead of %zu.  ", pvstream->name, fwritecounts, charswritten, writesize);
         if (ferror(pvstream->fp)) {
            fwriteFailure.printf("   Error: %s\n", strerror(errno));
         }
         if (fwritecounts<MAX_FILESYSTEMCALL_TRIES) {
            fwriteFailure.printf("Retrying.\n");
            sleep(1);
            int fseekstatus = fseek(pvstream->fp, fpos, SEEK_SET);
            if (fseekstatus != 0) {
               pvError().printf("PV_fwrite error: Unable to reset file position of \"%s\".  Fatal error: %s\n", pvstream->name, strerror(errno));
            }
            long int ftellreturn = ftell(pvstream->fp);
            if (fpos != ftellreturn) {
               pvError().printf("PV_fwrite error: attempted to reset file position of \"%s\" to %ld, but ftell() returned %ld.  Fatal error.\n", pvstream->name, fpos, ftellreturn);
            }
         }
         else {
            pvErrorNoExit().printf("MAX_FILESYSTEMCALL_TRIES exceeded.\n");
            return (size_t) 0;
         }
      }
   }
   if (pvstream->verifyWrites && pvstream->isfile) {
      fflush(pvstream->fp);
      int status = PV_SUCCESS;
      PV_Stream * readStream = PV_fopen(pvstream->name, "r", false/*verifyWrites*/);
      if (readStream==NULL) {
         pvErrorNoExit().printf("PV_fwrite verification: unable to open \"%s\" for reading: %s\n", pvstream->name, strerror(errno));
         status = PV_FAILURE;
      }
      if (status == PV_SUCCESS) {
         if (fseek(readStream->fp, pvstream->filepos, SEEK_SET)!=0) {
            pvErrorNoExit().printf("PV_fwrite verification: unable to verify \"%s\" write of %zu chars from position %ld: %s\n", pvstream->name, writesize, pvstream->filepos, strerror(errno));
            status = PV_FAILURE;
         }
      }
      char * read_buffer = NULL;
      if (status == PV_SUCCESS) {
         read_buffer = (char *) malloc(writesize);
         if (read_buffer==NULL) {
            pvErrorNoExit().printf("PV_fwrite verification: unable to create readback buffer of size %zu to verify \"%s\"\n", writesize, pvstream->name);
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         for(size_t n=0; n<writesize; n++) { read_buffer[n] = ~((char *)ptr)[n]; } // Make sure read_buffer is different from ptr before reading
      }
      if (status == PV_SUCCESS) {
         size_t numread = fread(read_buffer, (size_t) 1, writesize, readStream->fp);
         if (numread != writesize) {
            pvErrorNoExit().printf("PV_fwrite verification: unable to read into readback buffer for \"%s\": fread returned %zu instead of %zu\n", pvstream->name, numread, writesize);
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         status = memcmp(ptr, read_buffer, writesize)==0 ? PV_SUCCESS : PV_FAILURE;
         if (status != PV_SUCCESS) {
            size_t badcount=0;
            for (size_t n=0; n<writesize; n++) { badcount += (((char *) ptr)[n]!=read_buffer[n]); }
            pvErrorNoExit().printf("PV_fwrite verification: readback of %zu bytes from \"%s\" starting at position %zu failed: %zu bytes disagree.\n", writesize, pvstream->name, pvstream->filepos, badcount);
         }
      }
      free(read_buffer);
      if (readStream) { PV_fclose(readStream); readStream = NULL; }
      if (status != PV_SUCCESS) {
         fseek(pvstream->fp, pvstream->filepos, SEEK_SET);
         return 0;
      }
   }
   pvstream->filepos += writesize;
   return nitems;
}

size_t PV_fread(void * RESTRICT ptr, size_t size, size_t nitems, PV_Stream * RESTRICT pvstream) {
   int freadcounts = 0;
   size_t readsize = nitems*size;
   size_t stilltoread = readsize;
   char * RESTRICT curptr = (char * RESTRICT) ptr;
   long int fpos = pvstream->filepos; // PV_ftell(pvstream);
   clearerr(pvstream->fp);
   if (fpos<0) {
      pvError().printf("PV_fread error: unable to determine file position of \"%s\".  Fatal error\n", pvstream->name);
   }
   while (stilltoread != 0UL) {
      size_t charsread_thispass = fread(curptr, 1UL, stilltoread, pvstream->fp);
      stilltoread -= charsread_thispass;
      pvstream->filepos += charsread_thispass;
      if (stilltoread == 0UL) {
         if (freadcounts>0) {
            pvWarn().printf("fread succeeded for \"%s\" on attempt %d.\n", pvstream->name, freadcounts+1);
         }
         break;
      }
      else {
         if (feof(pvstream->fp)) {
            pvWarn().printf("fread failure for \"%s\": end of file reached with %lu characters still unread.\n", pvstream->name, stilltoread);
            break;
         }
      }
      curptr += charsread_thispass;
      freadcounts++;
      if (freadcounts<MAX_FILESYSTEMCALL_TRIES) {
         pvWarn().printf("fread failure for \"%s\" on attempt %d.  %lu bytes read; %lu bytes still to read so far.\n", pvstream->name, freadcounts, charsread_thispass, stilltoread);
         sleep(1);
      }
      else {
         pvErrorNoExit().printf("PV_fread failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded, and %lu bytes of %lu read.\n", pvstream->name, MAX_FILESYSTEMCALL_TRIES, readsize-stilltoread, readsize);
         break;
      }
   }
   return (readsize - stilltoread)/size;
}

int PV_fclose(PV_Stream * pvstream) {
   int status = PV_SUCCESS;
   if (pvstream) {
      if (pvstream->fp && pvstream->isfile) {
         status = fclose(pvstream->fp); pvstream->fp = NULL;
         if (status!=0) {
            pvErrorNoExit().printf("fclose failure for \"%s\": %s", pvstream->name, strerror(errno));
         }
      }
      free(pvstream->name);
      free(pvstream->mode);
      free(pvstream); pvstream = NULL;
   }
   return status;
}

/**
 * Gets the number of patches for the given PVLayerLoc, in a non-shared weight context.
 * The return value is the number of patches for the global column (i.e. not a particular MPI process)
 * If asPostWeights is true, loc is interpreted as a postsynaptic layer's PVLayerLoc and patches are
 * not counted in the extended region.  If asPostWeights is false, loc is interpreted as a presynaptic
 * layer's PVLayerLoc and patches are counted in the extended region.
 */
int getNumGlobalPatches(PVLayerLoc const * loc, bool asPostWeights) {
   int nx = loc->nxGlobal;
   int ny = loc->nyGlobal;
   int nf = loc->nf;
   if (!asPostWeights) {
      PVHalo const * halo = &loc->halo;
      nx += halo->lt + halo->rt;
      ny += halo->dn + halo->up;
   }
   return nx*ny*nf;
}

/**
 * Copy patches into an unsigned char buffer
 */
int pvp_copy_patches(unsigned char * buf, PVPatch ** patches, pvwdata_t * dataStart, int numDataPatches,
                     int nxp, int nyp, int nfp, float minVal, float maxVal,
                     bool compressed=true) {
   // Copies data from patches and dataStart to buf.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype) characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken patches.
   // The values in patches[k] will be written to &buf[k].  (For PVP_KERNEL_FILE_TYPE, the values are always nx=nxp, ny=nyp, offset=0).
   // The numweights values from dataStart+k*numweights will be copied to buf starting at &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   unsigned char * cptr = buf;
   const int patchsize = nxp * nyp * nfp;
   int nx = nxp;
   int ny = nyp;
   int offset = 0;
   for (int k = 0; k < numDataPatches; k++) {
      if( patches != NULL ) {
         PVPatch * p = patches[k];
         nx = p->nx;
         ny = p->ny;
         offset = p->offset;
      }
      const pvwdata_t * data = dataStart + k*patchsize; // + offset; // Don't include offset as the entire patch will be copied

      unsigned short * nxny = (unsigned short *) cptr;
      nxny[0] = (unsigned short) nxp;
      nxny[1] = (unsigned short) nyp;
      cptr += 2 * sizeof(unsigned short);

      unsigned int * offsetptr = (unsigned int *) cptr;
      *offsetptr = 0;
      cptr += sizeof(unsigned int);

      if( compressed ) {
         for (int k = 0; k < patchsize; k++) {
            *cptr++ = compressWeight(data[k], minVal, maxVal);
         }
      }
      else {
         float * fptr = (float *) cptr;
         for (int k = 0; k < patchsize; k++) {
            *fptr++ = data[k];
         }
         cptr = (unsigned char *) fptr;
      }
   }

   return PV_SUCCESS;
}

/**
 * Set patches given an unsigned char input buffer
 */
int pvp_set_patches(const unsigned char * buf, const PVPatch * const * patches, pvwdata_t * dataStart, int numDataPatches,
                    int nxp, int nyp, int nfp, float minVal, float maxVal,
                    bool compress=true)
{
   // Copies weight values from buf to dataStart.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype) characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken patches.
   // The numweights values from dataStart+k*numweights will be copied from buf starting at &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   const unsigned char * cptr = buf;

   // const int sfp = 1;
   // const int sxp = nfp;
   // const int syp = nfp * nxp;
   const int patchsize = nxp * nyp * nfp; // syp * nyp;

   unsigned short nx = nxp;
   unsigned short ny = nyp;
   unsigned int offset = 0;
   for (int n = 0; n < numDataPatches; n++) {
      if( patches != NULL ) {
         const PVPatch * p = patches[n];
      }
      pvwdata_t * data = dataStart + n*patchsize; // Don't include offset as entire patch will be read from buf

      cptr += 2*sizeof(unsigned short)+sizeof(unsigned int);

      if( compress ) {
         for (int k = 0; k < patchsize; k++) {
            // values in buf are packed into chars
            data[k] += uncompressWeight(*cptr++, minVal, maxVal);
         }
      }
      else {
         const float * fptr = (const float *) cptr;
         for (int k = 0; k < patchsize; k++) {
            data[k] += *fptr++;
         }
         cptr = (unsigned char *) fptr;
      }
   }

   return PV_SUCCESS;
}


PV_Stream * pvp_open_read_file(const char * filename, Communicator * comm)
{
   PV_Stream * pvstream = NULL;
   if (comm->commRank() == 0) {
      pvstream = PV_fopen(filename, "rb", false/*verifyWrites*/);
      if (pvstream==NULL) {
        pvErrorNoExit().printf("pvp_open_read_file failed for \"%s\": %s\n", filename, strerror(errno));
      }
   }
   return pvstream;
}

PV_Stream * pvp_open_write_file(const char * filename, Communicator * comm, bool append)
{
   PV_Stream * pvstream = NULL;
   if (comm->commRank() == 0) {
      bool rwmode = false;
      if (append) {
         // If the file exists, need to use read/write mode (r+) since we'll navigate back to the header to update nbands
         // If the file does not exist, mode r+ gives an error
         struct stat filestat;
         char * realPath = expandLeadingTilde(filename);
         int status = stat(realPath, &filestat);
         free(realPath);
         if (status==0) {
            rwmode = true;
         }
         else {
            if (errno==ENOENT) {
               pvWarn().printf("activity file \"%s\" does not exist.  File will be created\n", filename);
               rwmode = false;
            }
            else {
               pvError().printf("Error opening activity file \"%s\": %s", filename, strerror(errno));
            }
         }
      }
      if (rwmode) {
         pvstream = PV_fopen(filename, "r+b", false/*verifyWrites*/);
         if (pvstream==NULL) {
            pvErrorNoExit().printf("pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
      else {
         pvstream = PV_fopen(filename, "wb", false/*verifyWrites*/);
         if (pvstream==NULL) {
            pvErrorNoExit().printf("pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
   }
   return pvstream;
}

int pvp_close_file(PV_Stream * pvstream, Communicator * comm)
{
   int status = PV_SUCCESS;
   if (comm->commRank()==0) {
      status = PV_fclose(pvstream);
   }
   return status;
}

int pvp_check_file_header(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams)
{
   int status = PV_SUCCESS;

   int nxProcs = comm->numCommColumns();
   int nyProcs = comm->numCommRows();
   int rank = comm->commRank();

   if (params[INDEX_NX_PROCS] != 1) {
      status = PV_FAILURE;
      if (rank==0) {
         pvErrorNoExit().printf("params[%d] = %d, should be 1\n", INDEX_NX_PROCS, params[INDEX_NX_PROCS]);
      }
   }
   if (params[INDEX_NY_PROCS] != 1) {
      status = PV_FAILURE;
      if (rank==0) {
         pvErrorNoExit().printf("params[%d] = %d, should be 1\n", INDEX_NY_PROCS, params[INDEX_NY_PROCS]);
      }
   }

   if (numParams < NUM_WGT_PARAMS) {
      status = PV_FAILURE;
      if (rank==0) {
         pvErrorNoExit().printf("pvp_check_file_header called with %d params (requires at least %zu)\n", numParams, NUM_WGT_PARAMS);
      }
   }
   
   if (numParams >= NUM_WGT_PARAMS) {
      int patchesInFile = params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES];
      int numGlobalRestricted = loc->nxGlobal*loc->nyGlobal*loc->nf;
      PVHalo const * halo = &loc->halo;
      int numGlobalExtended = (loc->nxGlobal+halo->lt+halo->rt)*(loc->nyGlobal+halo->dn+halo->up)*loc->nf;
      switch (params[INDEX_FILE_TYPE]) {
      case PVP_WGT_FILE_TYPE:
         if (patchesInFile != numGlobalRestricted && patchesInFile != numGlobalExtended) {
            status = PV_FAILURE;
            if (rank==0) {
               pvErrorNoExit(badNumParams);
               badNumParams.printf("params[%zu] = %d, should be ", NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES, params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES]);
               if (numGlobalExtended==numGlobalRestricted) {
                  badNumParams.printf("%d\n", numGlobalExtended);
               }
               else {
                  badNumParams.printf("either %d (as post weights) or %d (as pre weights)\n",
                        numGlobalRestricted, numGlobalExtended);
               }
            }
         }
         break;
      case PVP_KERNEL_FILE_TYPE:
         if (patchesInFile % loc->nf) { // Not enough information passed to function to get unit cell size
            status = PV_FAILURE;
            if (rank==0) {
               pvErrorNoExit().printf("params[%zu] = %d, should be a multiple of loc->nf=%d\n",
                     NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES, params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES], loc->nf);
            }
         }
         break;
      default:
         assert(0);
         break;
      }
   }

   if (status != 0) {
      for (int i = 0; i < numParams; i++) {
         pvErrorNoExit().printf("params[%2d]==%d\n", i, params[i]);
      }
   }

   return status;
} // pvp_check_file_header

#ifdef OBSOLETE // Marked obsolete June 27, 2016.
// Deprecated Nov 20, 2014.  Use pvp_check_file_header
int pvp_check_file_header_deprecated(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams)
{
   int status = PV_SUCCESS;
   int tmp_status = PV_SUCCESS;

   int nxProcs = comm->numCommColumns();
   int nyProcs = comm->numCommRows();

   if (loc->nx       != params[INDEX_NX])        {status = PV_FAILURE; tmp_status = INDEX_NX;}
   if (tmp_status == INDEX_NX) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("nx = %d != params[%d]==%d \n", loc->nx, INDEX_NX, params[INDEX_NX]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (loc->ny       != params[INDEX_NY])        {status = PV_FAILURE; tmp_status = INDEX_NY;}
   if (tmp_status == INDEX_NY) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("ny = %d != params[%d]==%d \n", loc->ny, INDEX_NY, params[INDEX_NY]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   // TODO: Fix the following check for the patch size.
   //if (loc->nf != params[INDEX_NF]) {status = PV_FAILURE; tmp_status = INDEX_NF;}
   //if (tmp_status == INDEX_NF) {
   //      pvErrorNoExit().printf("nBands = %d != params[%d]==%d \n", loc->nf, INDEX_NF, params[INDEX_NF]);
   //}
   if (loc->nxGlobal != params[INDEX_NX_GLOBAL]) {status = PV_FAILURE; tmp_status = INDEX_NX_GLOBAL;}
   if (tmp_status == INDEX_NX_GLOBAL) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("nxGlobal = %d != params[%d]==%d \n", loc->nxGlobal, INDEX_NX_GLOBAL, params[INDEX_NX_GLOBAL]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (loc->nyGlobal != params[INDEX_NY_GLOBAL]) {status = PV_FAILURE; tmp_status = INDEX_NY_GLOBAL;}
   if (tmp_status == INDEX_NY_GLOBAL) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("nyGlobal = %d != params[%d]==%d \n", loc->nyGlobal, INDEX_NY_GLOBAL, params[INDEX_NY_GLOBAL]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of layer size
      }
   }
   if (nxProcs != params[INDEX_NX_PROCS]) {status = PV_FAILURE; tmp_status = INDEX_NX_PROCS;}
   if (tmp_status == INDEX_NX_PROCS) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("nxProcs = %d != params[%d]==%d \n", nxProcs, INDEX_NX_PROCS, params[INDEX_NX_PROCS]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of num procs
      }
   }
   if (nyProcs != params[INDEX_NY_PROCS]) {status = PV_FAILURE; tmp_status = INDEX_NY_PROCS;}
   if (tmp_status == INDEX_NY_PROCS) {
      if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
         pvErrorNoExit().printf("nyProcs = %d != params[%d]==%d \n", nyProcs, INDEX_NY_PROCS, params[INDEX_NY_PROCS]);
      }
      else {
         status = PV_SUCCESS; // kernels can be used regardless of num procs
      }
   }
   // if (loc->nb != params[INDEX_NB]) {status = PV_FAILURE; tmp_status = INDEX_NB;}
   // if (tmp_status == INDEX_NB) {
   //    if (params[INDEX_FILE_TYPE] != PVP_KERNEL_FILE_TYPE){
   //       pvErrorNoExit().printf("nPad = %d != params[%d]==%d \n", loc->nb, INDEX_NB, params[INDEX_NB]);
   //    }
   //    else {
   //       status = PV_SUCCESS; // kernels can be used regardless of margin size
   //    }
   // }
   //TODO: remove? Duplicated check from above.
   //if (loc->nf != params[INDEX_NF]) {status = PV_FAILURE; tmp_status = INDEX_NF;}
   //if (tmp_status == INDEX_NF) {
   //      pvErrorNoExit().printf("nBands = %d != params[%d]==%d \n", loc->nf, INDEX_NF, params[INDEX_NF]);
   //}

   // (kx0,ky0) is for node 0 only (can be calculated otherwise)
   //
   //   if (loc->kx0      != params[INDEX_KX0])       status = -1;
   //   if (loc->ky0      != params[INDEX_KY0])       status = -1;

   if (status != 0) {
      pvErrorNoExit(paramsDump);
      for (int i = 0; i < numParams; i++) {
         paramsDump.printf("params[%d]==%d ", i, params[i]);
      }
      paramsDump.printf("\n");
   }

   return status;
} // pvp_check_file_header_deprecated
#endif // OBSOLETE // Marked obsolete June 27, 2016.


int pvp_read_header(PV_Stream * pvstream, Communicator * comm, int * params, int * numParams) {
   // Under MPI, called by all processes; nonroot processes should have pvstream==NULL
   // On entry, numParams is the size of the params buffer.
   // All process should have the same numParams on entry.
   // On exit, numParams is the number of params actually read, they're read into params[0] through params[(*numParams)-1]
   // All processes receive the same params, the same numParams, and the same return value (PV_SUCCESS or PV_FAILURE).
   // If the return value is PV_FAILURE, *numParams has information on the type of failure.
   int status = PV_SUCCESS;
   int numParamsRead = 0;
   int * mpi_buffer = (int *) calloc((size_t)(*numParams+2), sizeof(int));
   // int mpi_buffer[*numParams+2]; // space for params to be MPI_Bcast, along with space for status and number of params read
   if (comm->commRank()==0) {
      if (pvstream==NULL) {
         pvErrorNoExit().printf("pvp_read_header: pvstream==NULL for rank zero");
         status = PV_FAILURE;
      }
      if (*numParams < 2) {
         numParamsRead = 0;
         status = PV_FAILURE;
      }

      // find out how many parameters there are
      //
      if (status == PV_SUCCESS) {
         int numread = PV_fread(params, sizeof(int), 2, pvstream);
         if (numread != 2) {
            numParamsRead = -1;
            status = PV_FAILURE;
         }
      }
      int nParams = 0;
      if (status == PV_SUCCESS) {
         nParams = params[INDEX_NUM_PARAMS];
         if (params[INDEX_HEADER_SIZE] != nParams * (int)sizeof(int)) {
            numParamsRead = -2;
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         if (nParams > *numParams) {
            numParamsRead = nParams;
            status = PV_FAILURE;
         }
      }

      // read the rest
      //
      if (status == PV_SUCCESS && *numParams > 2) {
         size_t numRead = PV_fread(&params[2], sizeof(int), nParams - 2, pvstream);
         if (numRead != (size_t) nParams - 2) {
            status = PV_FAILURE;
            *numParams = numRead;
         }
      }
      if (status == PV_SUCCESS) {
         numParamsRead  = params[INDEX_NUM_PARAMS];
      }
      mpi_buffer[0] = status;
      mpi_buffer[1] = numParamsRead;
      memcpy(&mpi_buffer[2], params, sizeof(int)*(*numParams));
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0/*root*/, comm->communicator());
   } // comm->communicator()==0
   else {
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0/*root*/, comm->communicator());
      status = mpi_buffer[0];
      memcpy(params, &mpi_buffer[2], sizeof(int)*(*numParams));
   }
   *numParams = mpi_buffer[1];
   free(mpi_buffer);
   return status;
}

void read_header_err(const char * filename, Communicator * comm, int returned_num_params, int * params) {
   if (comm->commRank() != 0) {
      pvErrorNoExit(header_error);
      header_error.printf("readBufferFile error while reading \"%s\"\n", filename);
      switch(returned_num_params) {
      case 0:
         header_error.printf("   Called with fewer than 2 params (%d); at least two are required.\n", returned_num_params);
         break;
      case -1:
         header_error.printf("   Error reading first two params from file");
         break;
      case -2:
         header_error.printf("   Header size %d and number of params %d in file are not compatible.\n", params[INDEX_HEADER_SIZE], params[INDEX_NUM_PARAMS]);
         break;
      default:
         if (returned_num_params < (int) NUM_BIN_PARAMS) {
            header_error.printf("   Called with %d params but only %d params could be read from file.\n", (int) NUM_BIN_PARAMS, returned_num_params);
         }
         else {
            header_error.printf("   Called with %d params but file contains %d params.\n", (int) NUM_BIN_PARAMS, returned_num_params);
         }
         break;
      }
   }
   abort();
}

static
int pvp_read_header(PV_Stream * pvstream, double * time, int * filetype,
                    int * datatype, int params[], int * numParams)
{
   int status = PV_SUCCESS;

   if (*numParams < 2) {
      *numParams = 0;
      return -1;
   }

   // find out how many parameters there are
   //
   if ( PV_fread(params, sizeof(int), 2, pvstream) != 2 ) return -1;

   int nParams = params[INDEX_NUM_PARAMS];
   assert(params[INDEX_HEADER_SIZE] == (int) (nParams * sizeof(int)));
   if (nParams > *numParams) {
      *numParams = 2;
      return -1;
   }

   // read the rest
   //
   if (PV_fread(&params[2], sizeof(int), nParams - 2, pvstream) != (unsigned int) nParams - 2) return -1;

   *numParams  = params[INDEX_NUM_PARAMS];
   *filetype   = params[INDEX_FILE_TYPE];
   *datatype   = params[INDEX_DATA_TYPE];

   // make sure the parameters are what we are expecting
   //

   assert(params[INDEX_DATA_SIZE] == (int) pv_sizeof(*datatype));

   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_read_header(const char * filename, Communicator * comm, double * time,
                    int * filetype, int * datatype, int params[], int * numParams)
{
   int status = PV_SUCCESS;
   const int icRank = comm->commRank();

   if (icRank == 0) {
       PV_Stream * pvstream = pvp_open_read_file(filename, comm);
       if (pvstream == NULL) {
          pvError().printf("[%2d]: pvp_read_header: pvp_open_read_file failed to open file \"%s\"\n",
                  comm->commRank(), filename);
       }

       status = pvp_read_header(pvstream, time, filetype, datatype, params, numParams);
       pvp_close_file(pvstream, comm);
       if (status != 0) return status;
   }

   const int icRoot = 0;
#ifdef DEBUG_OUTPUT
   pvDebug().printf("[%2d]: pvp_read_header: will broadcast, numParams==%d\n",
           comm->commRank(), *numParams);
#endif // DEBUG_OUTPUT

   status = MPI_Bcast(params, *numParams, MPI_INT, icRoot, comm->communicator());

#ifdef DEBUG_OUTPUT
   pvDebug().printf("[%2d]: pvp_read_header: broadcast completed, numParams==%d\n",
           comm->commRank(), *numParams);
#endif // DEBUG_OUTPUT

   *filetype = params[INDEX_FILE_TYPE];
   *datatype = params[INDEX_DATA_TYPE];
   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_write_header(PV_Stream * pvstream, Communicator * comm, int * params, int numParams) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();
   if (rank == rootproc) {
      if ( (int) PV_fwrite(params, sizeof(int), numParams, pvstream) != numParams ) {
         status = -1;
      }
   }

   return status;
}



int pvp_write_header(PV_Stream * pvstream, Communicator * comm, double time, const PVLayerLoc * loc, int filetype,
                     int datatype, int numbands, bool extended, bool contiguous, unsigned int numParams, size_t recordSize)
{
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks;
//   int numItems;
   int params[NUM_BIN_PARAMS];

   if (comm->commRank() != 0) return status;

   const int headerSize = numParams * sizeof(int);

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//   const int nb = loc->nb;

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   // const size_t globalSize = (size_t) localSize * nxBlocks * nyBlocks;

   // make sure we don't blow out size of int for record size
   // assert(globalSize < 0xffffffff); // should have checked this before calling pvp_write_header().

   int numRecords;
   int paramNBands;

   switch(filetype) {
   case PVP_WGT_FILE_TYPE:
      numRecords = numbands * nxBlocks * nyBlocks; // Each process writes a record for each arbor
      paramNBands = numbands;
      break;
   case PVP_KERNEL_FILE_TYPE:
      numRecords = numbands; // Each arbor writes its own record; all processes have the same weights
      paramNBands = numbands;
      break;
   default:
      numRecords = nxBlocks * nyBlocks; // For activity files, each process writes its own record
      paramNBands = numbands * loc->nbatch;
      break;
   }

   params[INDEX_HEADER_SIZE] = headerSize;
   params[INDEX_NUM_PARAMS]  = numParams;
   params[INDEX_FILE_TYPE]   = filetype;
   params[INDEX_NX]          = contiguous ? loc->nxGlobal : loc->nx;
   params[INDEX_NY]          = contiguous ? loc->nyGlobal : loc->ny;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = numRecords;
   params[INDEX_RECORD_SIZE] = recordSize;
   params[INDEX_DATA_SIZE]   = pv_sizeof(datatype);
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = nxBlocks;
   params[INDEX_NY_PROCS]    = nyBlocks;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = loc->kx0;
   params[INDEX_KY0]         = loc->ky0;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   params[INDEX_NBANDS]      = paramNBands;

   timeToParams(time, &params[INDEX_TIME]);

   numParams = NUM_BIN_PARAMS;  // there may be more to come
   if ( PV_fwrite(params, sizeof(int), numParams, pvstream) != numParams ) {
      status = -1;
   }

   return status;
}

int * pvp_set_file_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf * datasize;
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   params[INDEX_NBANDS]      = numbands * loc->nbatch;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_activity_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params              = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_ACT_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf * datasize; // does not represent the size of the record in the file, but the size of the buffer
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBANDS]      = numbands * loc->nbatch;
   params[INDEX_NBATCH]      = loc->nbatch;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_weight_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   // numPatches in argument list is the number of patches per process, but what's saved in numPatches to the file is number of patches across all processes.
   int numParams = NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_WGT_FILE_TYPE;
   params[INDEX_NX]          = loc->nx; // not yet contiguous
   params[INDEX_NY]          = loc->ny;
   params[INDEX_NF]          = loc->nf;
   int nxProcs               = comm->numCommColumns();
   int nyProcs               = comm->numCommRows();
   int datasize              = pv_sizeof(datatype);
   params[INDEX_NUM_RECORDS] = numbands * nxProcs * nyProcs;
   params[INDEX_RECORD_SIZE] = numPatches * (8 + datasize*nxp*nyp*nfp);
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = nxProcs;
   params[INDEX_NY_PROCS]    = nyProcs;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   params[INDEX_NBANDS]      = numbands;
   timeToParams(timed, &params[INDEX_TIME]);
   set_weight_params(params, nxp, nyp, nfp, min, max, numPatches);
   return params;
}

int * pvp_set_nonspiking_act_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_NONSPIKING_ACT_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf;
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   params[INDEX_NBANDS]      = numbands * loc->nbatch;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * pvp_set_kernel_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   int numParams = NUM_BIN_PARAMS;
   int * params = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_KERNEL_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   // int nxProcs               = 1;
   // int nyProcs               = 1;
   int datasize              = pv_sizeof(datatype);
   params[INDEX_NUM_RECORDS] = numbands;
   params[INDEX_RECORD_SIZE] = numPatches * (8 + datasize*nxp*nyp*nfp);
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   timeToParams(timed, &params[INDEX_TIME]);
   set_weight_params(params, nxp, nyp, nfp, min, max, numPatches);
   return params;
}

int * pvp_set_nonspiking_sparse_act_params(Communicator * comm, double timed, const PVLayerLoc * loc, int datatype, int numbands) {
   int numParams = NUM_BIN_PARAMS;
   int * params              = alloc_params(numParams);
   assert(params!=NULL);
   params[INDEX_FILE_TYPE]   = PVP_ACT_SPARSEVALUES_FILE_TYPE;
   params[INDEX_NX]          = loc->nxGlobal;
   params[INDEX_NY]          = loc->nyGlobal;
   params[INDEX_NF]          = loc->nf;
   params[INDEX_NUM_RECORDS] = 1;
   int datasize = pv_sizeof(datatype);
   params[INDEX_RECORD_SIZE] = loc->nxGlobal * loc->nyGlobal * loc->nf * datasize; // does not represent the size of the record in the file, but the size of the buffer
   params[INDEX_DATA_SIZE]   = datasize;
   params[INDEX_DATA_TYPE]   = datatype;
   params[INDEX_NX_PROCS]    = 1;
   params[INDEX_NY_PROCS]    = 1;
   params[INDEX_NX_GLOBAL]   = loc->nxGlobal;
   params[INDEX_NY_GLOBAL]   = loc->nyGlobal;
   params[INDEX_KX0]         = 0;
   params[INDEX_KY0]         = 0;
   params[INDEX_NBANDS]      = numbands * loc->nbatch;
   params[INDEX_NBATCH]      = loc->nbatch; // loc->nb;
   timeToParams(timed, &params[INDEX_TIME]);
   return params;
}

int * alloc_params(int numParams) {
   int * params = NULL;
   if (numParams<2) {
      pvError().printf("alloc_params must be called with at least two params (called with %d).\n", numParams);
   }
   params = (int *) calloc((size_t) numParams, sizeof(int));
   if (params == NULL) {
      pvError().printf("alloc_params unable to allocate %d params: %s\n", numParams, strerror(errno));
   }
   params[INDEX_HEADER_SIZE] = sizeof(int)*numParams;
   params[INDEX_NUM_PARAMS] = numParams;
   return params;
}

int set_weight_params(int * params, int nxp, int nyp, int nfp, float min, float max, int numPatches) {
   int * wgtParams = &params[NUM_BIN_PARAMS];
   wgtParams[INDEX_WGT_NXP]  = nxp;
   wgtParams[INDEX_WGT_NYP]  = nyp;
   wgtParams[INDEX_WGT_NFP]  = nfp;
   assert(sizeof(int)==sizeof(float));
   union float_as_int {
      float f;
      int   i;
   };
   union float_as_int p;
   p.f = min;
   wgtParams[INDEX_WGT_MIN] = p.i;
   p.f = max;
   wgtParams[INDEX_WGT_MAX] = p.i;
   wgtParams[INDEX_WGT_NUMPATCHES]  = numPatches;
   return PV_SUCCESS;
}

int pvp_read_time(PV_Stream * pvstream, Communicator * comm, int root_process, double * timed) {
   // All processes call this routine simultaneously.
   // from the file at the current location, loaded into the variable timed, and
   // broadcast to all processes.  All processes have the same return value:
   // PV_SUCCESS if the read was successful, PV_FAILURE if not.
   int status = PV_SUCCESS;
   struct timeandstatus {
      int status;
      double time;
   };
   struct timeandstatus mpi_data;
   if (comm->commRank()==root_process) {
      if (pvstream==NULL) {
         pvErrorNoExit().printf("pvp_read_time: root process called with null stream argument.\n");
         abort();
      }
      int numread = PV_fread(timed, sizeof(*timed), 1, pvstream);
      mpi_data.status = (numread == 1) ? PV_SUCCESS : PV_FAILURE;
      mpi_data.time = *timed;
   }
   MPI_Bcast(&mpi_data, (int) sizeof(timeandstatus), MPI_CHAR, root_process, comm->communicator());
   status = mpi_data.status;
   *timed = mpi_data.time;
   return status;
}

int writeActivity(PV_Stream * pvstream, Communicator * comm, double timed, DataStore * store, const PVLayerLoc* loc)
{
   int status = PV_SUCCESS;

   // write header, but only at the beginning
   int rank = comm->commRank();

   for(int b = 0; b < loc->nbatch; b++){
      pvadata_t * data = (pvadata_t*) store->buffer(b);
      if( rank == 0 ) {
         long fpos = getPV_StreamFilepos(pvstream);
         if (fpos == 0L) {
            int * params = pvp_set_nonspiking_act_params(comm, timed, loc, PV_FLOAT_TYPE, 1/*numbands*/);
            assert(params && params[1]==NUM_BIN_PARAMS);
            int numParams = params[1];
            status = pvp_write_header(pvstream, comm, params, numParams);
            free(params);
         }
         // HyPerLayer::writeActivity calls HyPerLayer::incrementNBands, which maintains the value of numbands in the header.

         // write time
         //
         if ( PV_fwrite(&timed, sizeof(double), 1, pvstream) != 1 ) {
            pvError().printf("fwrite of timestamp in PV::writeActivity failed in file \"%s\" at time %f\n", pvstream->name, timed);
            return -1;
         }
      }

      if (gatherActivity(pvstream, comm, 0/*root process*/, data, loc, true/*extended*/)!=PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

int writeActivitySparse(PV_Stream * pvstream, Communicator * comm, double timed, DataStore * store, const PVLayerLoc* loc, bool includeValues)
{
   int status = PV_SUCCESS;

   //Grab active indices and local active from datastore comm

   const int icRoot = 0;
   const int icRank = comm->commRank();

   for(int b = 0; b < loc->nbatch; b++){

      int localActive = *(store->numActiveBuffer(b));
      unsigned int * indices = store->activeIndicesBuffer(b);
      pvadata_t * valueData = (pvadata_t*) store->buffer(b);

      indexvaluepair * indexvaluepairs = NULL;
      unsigned int * globalResIndices = NULL;
      int localResActive = 0;

#ifdef PV_USE_MPI
      const int tag = includeValues?PVP_ACT_SPARSEVALUES_FILE_TYPE:PVP_ACT_FILE_TYPE;
      const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI


      if (icRank != icRoot) {

#ifdef PV_USE_MPI
         const int dest = icRoot;
#ifdef DEBUG_OUTPUT
         pvDebug().printf("[%2d]: writeActivitySparseNonspiking: sent localActive value of %d to %d\n",
                 comm->commRank, localActive, dest);
#endif // DEBUG_OUTPUT
         void * data = NULL;
         size_t datasize = 0UL;
         MPI_Datatype mpi_type = NULL;
         if (includeValues) {
            indexvaluepairs = (indexvaluepair *) malloc(localActive*sizeof(indexvaluepair));
            if (indexvaluepairs==NULL) {
               pvErrorNoExit().printf("writeActivitySparseNonspiking: Rank %d process unable to allocate memory for indexvaluepairs: %s\n",
                       icRank, strerror(errno));
               exit(EXIT_FAILURE);
            }
            int pairsIdx = 0;
            for (int j=0; j<localActive; j++) {
               int localExtK = indices[j];
               int globalResK = localExtToGlobalRes(localExtK, loc);
               if(globalResK == -1){
                  continue;
               }
               indexvaluepairs[pairsIdx].index = globalResK;
               indexvaluepairs[pairsIdx].value = valueData[localExtK];
               pairsIdx++;
            }
            data = (void *) indexvaluepairs;
            datasize = sizeof(indexvaluepair);
            localResActive = pairsIdx;
         }
         else {
            //Change local ext indices to global res index
            globalResIndices = (unsigned int *) malloc(localActive*sizeof(unsigned int));
            int indiciesIdx = 0;
            for (int j=0; j<localActive; j++) {
               int localExtK = indices[j];
               int globalResK = localExtToGlobalRes(localExtK, loc);
               if(globalResK == -1){
                  continue;
               }
               globalResIndices[indiciesIdx] = globalResK;
               indiciesIdx++;
            }
            data = (void *) globalResIndices;
            datasize = sizeof(unsigned int);
            localResActive = indiciesIdx;
         }
         MPI_Ssend(&localResActive, 1, MPI_INT, dest, tag, mpi_comm);

         MPI_Ssend(data, localResActive*datasize, MPI_CHAR, dest, tag,mpi_comm);

#ifdef DEBUG_OUTPUT
         pvDebug(debugWriteActivitySparseSent);
         debugWriteActivitySparseSent.printf("[%2d]: writeActivitySparse: sent to %d, localActive==%d\n",
                 comm->commRank(), dest, localResActive);
         debugWriteActivitySparse.flush();
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

         // leaving not root-process section
         //
      }
      else {
         void * data = NULL;
         // we are io root process
         //
         if (localActive > 0) {
            if (includeValues) {
               indexvaluepairs = (indexvaluepair *) malloc(localActive*sizeof(indexvaluepair));
               assert(indexvaluepairs); /* lazy; fix with a proper error message */

               int pairsIdx = 0;
               for (int k=0; k<localActive; k++) {
                  int localExtK = indices[k];
                  int globalResK = localExtToGlobalRes(localExtK, loc);
                  if(globalResK == -1){
                     continue;
                  }
                  indexvaluepairs[pairsIdx].index = globalResK;
                  indexvaluepairs[pairsIdx].value = valueData[localExtK];
                  pairsIdx++;
               }
               localResActive = pairsIdx;
            }
            else{
               //Change local ext indices to global res index
               globalResIndices = (unsigned int *) malloc(localActive*sizeof(unsigned int));
               int indiciesIdx = 0;
               for (int j=0; j<localActive; j++) {
                  int localExtK = indices[j];
                  int globalResK = localExtToGlobalRes(localExtK, loc);
                  if(globalResK == -1){
                     continue;
                  }
                  globalResIndices[indiciesIdx] = globalResK;
                  indiciesIdx++;
               }
               localResActive = indiciesIdx;
            }
         }

         //Need to calculate totalResActive of root process
         unsigned int totalActive = localResActive;

#ifdef PV_USE_MPI
         // get the number active from each process
         //
         unsigned int * numActive = NULL;
         const int icSize = comm->commSize();

         if (icSize > 1) {
            // otherwise numActive is not used
            numActive = (unsigned int *) malloc(icSize*sizeof(unsigned int));
            if (numActive == NULL) {
               pvErrorNoExit().printf("writeActivitySparseNonspiking: Root process unable to allocate memory for numActive array: %s\n",
                       strerror(errno));
               exit(EXIT_FAILURE);
            }
         }

         for (int p = 1; p < icSize; p++) {
#ifdef DEBUG_OUTPUT
            pvDebug().printf("[%2d]: writeActivitySparseNonspiking: receiving numActive value from %d\n",
                    comm->commRank(), p);
#endif // DEBUG_OUTPUT
            MPI_Status mpi_status;
            MPI_Recv(&numActive[p], 1, MPI_INT, p, tag, mpi_comm, &mpi_status);
            totalActive += numActive[p];
         }
#endif // PV_USE_MPI

         bool extended   = false;
         bool contiguous = true;

         const int datatype = includeValues?PV_SPARSEVALUES_TYPE:PV_INT_TYPE;
         const int filetype = includeValues?PVP_ACT_SPARSEVALUES_FILE_TYPE:PVP_ACT_FILE_TYPE;

         // write activity header
         //
         long fpos = getPV_StreamFilepos(pvstream);
         if (fpos == 0L) {
            int numParams = NUM_BIN_PARAMS;
            status = pvp_write_header(pvstream, comm, timed, loc, filetype,
                                      datatype, 1, extended, contiguous,
                                      numParams, (size_t) totalActive);
            if (status != 0) {
               pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in pvp_write_header, numParams==%d, localActive==%d\n",
                       comm->commRank(), numParams, localActive);
               return status;
            }
         }

         // write time, total active count, and local activity
         //
         status = (PV_fwrite(&timed, sizeof(double), 1, pvstream) != 1 );
         if (status != 0) {
            pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in fwrite(&timed), time==%f\n",
                    comm->commRank(), timed);
            return status;
         }
         status = ( PV_fwrite(&totalActive, sizeof(unsigned int), 1, pvstream) != 1 );
         if (status != 0) {
            pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in fwrite(&totalActive), totalActive==%d\n",
                    comm->commRank(), totalActive);
            return status;
         }
         
         //Write to seperate file this current file position
         //long filepos = pvstream->filepos;
         //status = (PV_fwrite(&filepos, sizeof(long), 1, posstream) != 1);
         //fflush(posstream->fp);
         
         //if (status != 0) {
         //   pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in fwrite(&filepos), filepos==%ld\n",
         //           comm->commRank(), filepos);
         //   return status;
         //}

         if (localResActive > 0) {
            if (includeValues) {
               status = (PV_fwrite(indexvaluepairs, sizeof(indexvaluepair), localResActive, pvstream) != (size_t) localResActive);
            }
            else {
               status = (PV_fwrite(globalResIndices, sizeof(unsigned int), localResActive, pvstream) != (size_t) localResActive);
            }
            if (status != 0) {
               pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in PV_fwrite(indices), localActive==%d\n",
                     comm->commRank(), localResActive);
               return status;
            }
         }

         // recv and write non-local activity
         //
#ifdef PV_USE_MPI
         for (int p = 1; p < icSize; p++) {
            void * data = NULL;
            size_t datasize = 0UL;
            if (includeValues) {
               datasize = sizeof(indexvaluepair);
               free(indexvaluepairs);
               indexvaluepairs = (indexvaluepair *) malloc(numActive[p]*datasize);
               assert(indexvaluepairs); /* lazy; fix with proper error message */
               data = (void *) indexvaluepairs;
            }
            else {
               datasize = sizeof(unsigned int);
               free(globalResIndices);
               globalResIndices = (unsigned int *) malloc(numActive[p]*datasize);
               assert(globalResIndices);
               data = (void *) globalResIndices;
            }
#ifdef DEBUG_OUTPUT
            pvDebug(debugWriteActivitySparseReceiving);
            debugWriteActivitySparseReceiving.printf("[%2d]: writeActivitySparse: receiving from %d, numActive==%d\n",
                    comm->commRank(), p, numActive[p]);
            debugWriteActivitySparseReceiving.flush();
#endif // DEBUG_OUTPUT
            
            MPI_Status mpi_status;
            MPI_Recv(data, numActive[p]*datasize, MPI_CHAR, p, tag, mpi_comm, &mpi_status);

            status = (PV_fwrite(data, datasize, numActive[p], pvstream) != numActive[p] );
            if (status != 0) {
               pvErrorNoExit().printf("[%2d]: writeActivitySparse: failed in PV_fwrite(indices), numActive[p]==%d, p=%d\n",
                       comm->commRank(), numActive[p], p);
               return status;
            }
         }
         free(numActive);
#endif // PV_USE_MPI

         // leaving root-process section
         //
      }
      free(indexvaluepairs);
      free(globalResIndices);
   }
   return status;
}

int readWeights(PVPatch *** patches, pvwdata_t ** dataStart, int numArbors, int numPatches,
      int nxp, int nyp, int nfp, const char * filename, Communicator * comm, double * timed, const PVLayerLoc * loc) {
   int header_data_type;
   int header_file_type;

   int numParams = NUM_WGT_PARAMS;
   int params[NUM_WGT_PARAMS];
   int * wgtParams = &params[NUM_BIN_PARAMS];
   int status = pvp_read_header(filename, comm, timed, &header_file_type, &header_data_type, params, &numParams);

   // rank zero process broadcasts params to all processes, so it's enough for rank zero process to do the error checking
   if (comm->commRank()==0) {
      if (numParams != NUM_WGT_PARAMS) {
         pvError().printf("Reading weights file \"%s\": expected %zu parameters in header but received %d\n", filename, NUM_WGT_PARAMS, numParams);
      }
      if (params[NUM_BIN_PARAMS+INDEX_WGT_NXP] != nxp || params[NUM_BIN_PARAMS+INDEX_WGT_NYP] != nyp) {
         pvError().printf("readWeights error: called with nxp=%d, nyp=%d, but \"%s\" has nxp=%d, nyp=%d\n", nxp, nyp, filename, params[NUM_BIN_PARAMS+INDEX_WGT_NXP], params[NUM_BIN_PARAMS+INDEX_WGT_NYP]);
      }
   }

   const int nxFileBlocks = params[INDEX_NX_PROCS];
   const int nyFileBlocks = params[INDEX_NY_PROCS];
   
   status = pvp_check_file_header(comm, loc, params, numParams);
   
#ifdef OBSOLETE // readWeightsDeprecated was marked obsolete Jun 27, 2016.
   // If file header is not compatible, try the old MPI-dependent file format (deprecated on Nov 20, 2014).
   if (status != 0) {
      return readWeightsDeprecated(patches, dataStart, numArbors, numPatches, nxp, nyp, nfp, filename, comm, timed, loc);
   }
#endif // OBSOLETE // readWeightsDeprecated was marked obsolete Jun 27, 2016.

   if (status != 0) {
      pvErrorNoExit().printf("[%2d]: readWeights: failed in pvp_check_file_header, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }
   assert(params[INDEX_NX_PROCS]==1 && params[INDEX_NY_PROCS]==1);
   if( params[INDEX_NBANDS] > numArbors ) {
      pvErrorNoExit().printf("PV::readWeights: file \"%s\" has %d arbors, but readWeights was called with only %d arbors", filename, params[INDEX_NBANDS], numArbors);
      return -1;
   }
   
   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize = pv_sizeof_patch(numPatchItems, header_data_type);
   const size_t localSize = numPatches * patchSize;

   // Have to use memcpy instead of casting floats because of strict aliasing rules, since some are int and some are float
   float minVal = 0.0f;
   memcpy(&minVal, &wgtParams[INDEX_WGT_MIN], sizeof(float));
   float maxVal = 0.0f;
   memcpy(&maxVal, &wgtParams[INDEX_WGT_MAX], sizeof(float));
   // const float minVal = * ((float*) &wgtFloatParams[INDEX_WGT_MIN]);
   // const float maxVal = * ((float*) &wgtFloatParams[INDEX_WGT_MAX]);

   const int icRank = comm->commRank();

   bool compress = header_data_type == PV_BYTE_TYPE;
   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   if(cbuf == NULL) {
      pvError(errorMessage);
      errorMessage.printf("Rank %d: writeWeights unable to allocate memory to write to \"%s\": %s", icRank, filename, strerror(errno));
      errorMessage.printf("    (nxp=%d, nyp=%d, nfp=%d, numPatchItems=%d, writing weights as %s)\n", nxp, nyp, nfp, numPatchItems, compress ? "bytes" : "floats");
   }

   const int expected_file_type = patches == NULL ? PVP_KERNEL_FILE_TYPE : PVP_WGT_FILE_TYPE;
   const int tagbase = expected_file_type;
#ifdef PV_USE_MPI
   const MPI_Comm mpi_comm = comm->communicator();
#else
   const MPI_Comm mpi_comm = NULL;
#endif // PV_USE_MPI
   const int src = 0;
   if (expected_file_type != header_file_type) {
      if (icRank==0) {
         pvErrorNoExit().printf("readWeights: file \"%s\" has type %d but readWeights was called expecting type %d\n",
               filename, header_file_type, expected_file_type);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(mpi_comm);
#endif // PV_USE_MPI
      exit(EXIT_FAILURE);
   }
   if (icRank > 0) {
#ifdef PV_USE_MPI
      for (int arbor=0; arbor<params[INDEX_NBANDS]; arbor++) {
         if (header_file_type == PVP_KERNEL_FILE_TYPE) {
#ifdef DEBUG_OUTPUT
            pvDebug().printf("[%2d]: readWeights: bcast from %d, arbor %d, numPatchItems %d, numPatches==%d, localSize==%zu\n",
                    comm->commRank(), src, arbor, numPatchItems, numPatches, localSize);
#endif // DEBUG_OUTPUT
            MPI_Bcast(cbuf, localSize, MPI_BYTE, src, mpi_comm);
         }
         else {
            assert(header_file_type == PVP_WGT_FILE_TYPE);
#ifdef DEBUG_OUTPUT
            pvDebug().printf("[%2d]: readWeights: recv from %d, arbor %d, numPatchItems %d, numPatches==%d, localSize==%zu\n",
                    comm->commRank(), src, arbor, numPatchItems, numPatches, localSize);
#endif // DEBUG_OUTPUT
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tagbase+arbor, mpi_comm, MPI_STATUS_IGNORE);
         }
         pvp_set_patches(cbuf, patches ? patches[arbor] : NULL, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
      }
#endif // PV_USE_MPI
   } // icRank > 0
   else /*icRank == 0*/ {
      PV_Stream * pvstream = pvp_open_read_file(filename, comm);
      const int headerSize = numParams * sizeof(int);
      for (int arbor=0; arbor<params[INDEX_NBANDS]; arbor++) {
         long int arborStart = headerSize + localSize * arbor;
         if (header_file_type == PVP_KERNEL_FILE_TYPE) {
            PV_fseek(pvstream, arborStart, SEEK_SET);
            int numRead = PV_fread(cbuf, localSize, 1, pvstream);
            if (numRead != 1) {
               pvError().printf("readWeights error reading arbor %d of %zu bytes from position %ld of \"%s\".\n", arbor, localSize, arborStart, filename);
            };
#ifdef DEBUG_OUTPUT
            pvDebug().printf("[%2d]: readWeights: bcast from %d, arbor %d, numPatchItems %d, numPatches==%d, localSize==%zu\n",
                    comm->commRank(), src, arbor, numPatchItems, numPatches, localSize);
#endif // DEBUG_OUTPUT
            if(comm->commSize() > 1){
               MPI_Bcast(cbuf, localSize, MPI_BYTE, src, mpi_comm);
            }
         }
         else {
            assert(header_file_type == PVP_WGT_FILE_TYPE);
            int globalSize = patchSize * wgtParams[INDEX_WGT_NUMPATCHES];
            for (int proc = 0; proc <= comm->commSize(); proc++) {
               if (proc==src) { continue; } // Do local section last
               int procrow, proccolumn;
               if (proc==comm->commSize()) {
                  procrow = rowFromRank(src, comm->numCommRows(), comm->numCommColumns());
                  proccolumn = columnFromRank(src, comm->numCommRows(), comm->numCommColumns());
               }
               else {
                  procrow = rowFromRank(proc, comm->numCommRows(), comm->numCommColumns());
                  proccolumn = columnFromRank(proc, comm->numCommRows(), comm->numCommColumns());
               }
               for (int k=0; k<numPatches; k++) {
                  // TODO: don't have to read each patch singly; can read all patches for a single y at once.
                  unsigned char * cbufpatch = &cbuf[k*patchSize];
                  int x = kxPos(k, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf)+proccolumn*loc->nx;
                  int y = kyPos(k, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf)+procrow*loc->ny;
                  int f = featureIndex(k, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
                  int globalIndex = kIndex(x,y,f,loc->nxGlobal, loc->nyGlobal, loc->nf);
                  PV_fseek(pvstream, arborStart+globalIndex*patchSize, SEEK_SET);
                  int numread = PV_fread(cbufpatch, patchSize, 1, pvstream);
                  if (numread != 1) {
                     pvError().printf("readWeights error reading arbor %d, patch %d of %zu bytes from position %ld of \"%s\".\n", arbor, k, patchSize, arborStart+globalIndex*patchSize, filename);
                  }
               } // Loop over patches
               if (proc != comm->commSize()) {
                  MPI_Send(cbuf, localSize, MPI_BYTE, proc, tagbase+arbor, mpi_comm);
#ifdef DEBUG_OUTPUT
                  pvDebug().printf("[%2d]: readWeights: recv from %d, arbor %d, numPatchItems %d, numPatches==%d, localSize==%zu\n",
                        comm->commRank(), src, arbor, numPatchItems, numPatches, localSize);
#endif // DEBUG_OUTPUT
               }
            } // Loop over processes
            // Local section was done last, so cbuf should contain data from local section
         } // if-statement for header_file_type
         pvp_set_patches(cbuf, patches ? patches[arbor] : NULL, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
      } // loop over arbors
      pvp_close_file(pvstream, comm);
   } // if-statement for icRank
   
   free(cbuf); cbuf = NULL;
   
   return status;
}

#ifdef OBSOLETE // Marked obsolete Jun 27, 2016.
// readWeights was changed on Nov 20, 2014, to read weight files saved in an MPI-independent manner.
// readWeightsDeprecated is the old version of the file
int readWeightsDeprecated(PVPatch *** patches, pvwdata_t ** dataStart, int numArbors, int numPatches,
      int nxp, int nyp, int nfp, const char * filename, Communicator * comm, double * timed, const PVLayerLoc * loc){
   int status = PV_SUCCESS;
   int header_data_type;
   int header_file_type;

   int numParams = NUM_WGT_PARAMS;
   int params[NUM_WGT_PARAMS];

   int nxBlocks, nyBlocks;

   bool contiguous = false;   // for now

   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();

   const int icRank = comm->commRank();

   status = pvp_read_header(filename, comm, timed, &header_file_type, &header_data_type, params, &numParams);
   if( status != PV_SUCCESS ) {
      pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_read_header, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }

   status = pvp_check_file_header_deprecated(comm, loc, params, numParams);
   if (status != 0) {
      pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_check_file_header_deprecated, numParams==%d\n",
              comm->commRank(), numParams);
      return status;
   }

   const int nxFileBlocks = params[INDEX_NX_PROCS];
   const int nyFileBlocks = params[INDEX_NY_PROCS];

   int * wgtParams = &params[NUM_BIN_PARAMS];

   if (nxp != wgtParams[INDEX_WGT_NXP] || nyp != wgtParams[INDEX_WGT_NYP] || nfp != wgtParams[INDEX_WGT_NFP]) {
      if (icRank==0) {
         pvErrorNoExit().printf("readWeightsDeprecated: file \"%s\" patch dimensions (nxp=%d, nyp=%d, nfp=%d) do not agree with expected values (%d,%d,%d).\n",
               filename, wgtParams[INDEX_WGT_NXP], wgtParams[INDEX_WGT_NYP], wgtParams[INDEX_WGT_NFP], nxp, nyp, nfp);
      }
      MPI_Barrier(comm->communicator());
      exit(EXIT_FAILURE);
   }
   assert(nyp == wgtParams[INDEX_WGT_NYP]);
   assert(nfp == wgtParams[INDEX_WGT_NFP]);

   // Have to use memcpy instead of casting floats because of strict aliasing rules, since some are int and some are float
   float minVal = 0.0f;
   memcpy(&minVal, &wgtParams[INDEX_WGT_MIN], sizeof(float));
   float maxVal = 0.0f;
   memcpy(&maxVal, &wgtParams[INDEX_WGT_MAX], sizeof(float));
   // const float minVal = * ((float*) &wgtFloatParams[INDEX_WGT_MIN]);
   // const float maxVal = * ((float*) &wgtFloatParams[INDEX_WGT_MAX]);

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   // make sure file is consistent with expectations
   //
   status = ( header_data_type != PV_BYTE_TYPE && header_data_type != PV_FLOAT_TYPE );
   if (status != 0) {
      pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_check_file_header, datatype==%d\n",
              comm->commRank(), header_data_type);
      return status;
   }
   if (header_file_type != PVP_KERNEL_FILE_TYPE){
      status = (nxBlocks != nxFileBlocks || nyBlocks != nyFileBlocks);
      if (status != 0) {
         pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_check_file_header, "
               "nxFileBlocks==%d, nyFileBlocks==%d\n, nxBlocks==%d, nyBlocks==%d\n",
               comm->commRank(), nxFileBlocks, nyFileBlocks, nxBlocks, nyBlocks);
         return status;
      }
   }
   if (header_file_type != PVP_KERNEL_FILE_TYPE){
      status = (numPatches*nxProcs*nyProcs != wgtParams[INDEX_WGT_NUMPATCHES]);
   }
   else{
      status = ((numPatches != wgtParams[INDEX_WGT_NUMPATCHES]));
   }
   if (status != 0) {
      pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_check_file_header, "
            "numPatches==%d, nxProcs==%d\n, nyProcs==%d, wgtParams[INDEX_WGT_NUMPATCHES]==%d\n",
            comm->commRank(), numPatches, nxProcs, nyProcs, wgtParams[INDEX_WGT_NUMPATCHES]);
      return status;
   }


   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize = pv_sizeof_patch(numPatchItems, header_data_type);
   const size_t localSize = numPatches * patchSize;

   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   assert(cbuf != NULL);

#ifdef PV_USE_MPI
   const int tag = header_file_type;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI

   // read weights and send using MPI
   if( params[INDEX_NBANDS] > numArbors ) {
      pvErrorNoExit().printf("PV::readWeightsDeprecated: file \"%s\" has %d arbors, but readWeightsDeprecated was called with only %d arbors", filename, params[INDEX_NBANDS], numArbors);
      return -1;
   }
   PV_Stream * pvstream = pvp_open_read_file(filename, comm);
   for(int arborId=0; arborId<params[INDEX_NBANDS]; arborId++) {
      if (icRank > 0) {

#ifdef PV_USE_MPI
         const int src = 0;

#ifdef DEBUG_OUTPUT
         pvDebug().printf("[%2d]: readWeightsDeprecated: recv from %d, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
                 comm->commRank(), src, nxBlocks, nyBlocks, numPatches);
#endif // DEBUG_OUTPUT
         MPI_Recv(cbuf, localSize, MPI_BYTE, src, tag, mpi_comm, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
         pvDebug().printf("[%2d]: readWeightsDeprecated: recv from %d completed\n",
                 comm->commRank(), src);
#endif // DEBUG_OUTPUT
#endif // PV_USE_MPI

      }
      else {
         const int headerSize = numParams * sizeof(int);

         if (pvstream == NULL) {
            pvErrorNoExit().printf("PV::readWeightsDeprecated: unable to open file %s\n", filename);
            return -1;
         }

         long arborStart;
#ifdef PV_USE_MPI
         int dest = -1;
#endif // PV_USE_MPI
         if( header_file_type == PVP_KERNEL_FILE_TYPE ) {
            arborStart = headerSize + localSize*arborId;
            long offset = arborStart;
            PV_fseek(pvstream, offset, SEEK_SET);
            int numRead = PV_fread(cbuf, localSize, 1, pvstream);
            if( numRead != 1 ) return -1;
#ifdef PV_USE_MPI
            for( int py=0; py<nyProcs; py++ ) {
               for( int px=0; px<nxProcs; px++ ) {
                  if( ++dest == 0 ) continue;
                  MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
               }
            }
#endif // PV_USE_MPI
         }
         else {
            arborStart = headerSize + localSize*nxBlocks*nyBlocks*arborId;
#ifdef PV_USE_MPI
            for( int py=0; py<nyProcs; py++ ) {
               for( int px=0; px<nxProcs; px++ ) {
                  if( ++dest == 0 ) continue;
                  long offset = arborStart + dest*localSize;
                  PV_fseek(pvstream, offset, SEEK_SET);
                  int numRead = PV_fread(cbuf, localSize, 1, pvstream);
                  if( numRead != 1 ) return -1;
                  MPI_Send(cbuf, localSize, MPI_BYTE, dest, tag, mpi_comm);
               }
            }
#endif // PV_USE_MPI
         }

         // read local portion
         // numPatches - each neuron has a patch; pre-synaptic neurons live in extended layer
         //
#ifdef PV_USE_MPI
         bool readLocalPortion = header_file_type != PVP_KERNEL_FILE_TYPE;
#else
         bool readLocalPortion = true;
#endif // PV_USE_MPI
         if( readLocalPortion ) {
            long offset = arborStart + 0*localSize;
            PV_fseek(pvstream, offset, SEEK_SET);
            int numRead = PV_fread(cbuf, localSize, 1, pvstream);
            if  (numRead != 1) {
               pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in PV_fread, offset==%ld\n",
                     comm->commRank(), offset);
            }
         }
      }  // if rank == 0

      // set the contents of the weights patches from the unsigned character buffer, cbuf
      //
      bool compress = header_data_type == PV_BYTE_TYPE;

      assert(status == PV_SUCCESS);
      // check that nx,ny,offset info in patches (passed as an input argument, typically determined by params file)
      // is consistent with the nx,ny,offset info in cbuf (read from the filename)
      // (used to be in pvp_set_patches, but that doesn't have filename to use in the error messages)
      if (patches != NULL) {
         const unsigned char * patchinfofromfile = (const unsigned char *) cbuf;
         for (int patchindex=0; patchindex<numPatches; patchindex++) {
            unsigned short int const nx = *(unsigned short *) patchinfofromfile;
            patchinfofromfile += sizeof(unsigned short);
            unsigned short int const ny = *(unsigned short *) patchinfofromfile;
            patchinfofromfile += sizeof(unsigned short);
            unsigned short int const offset = *(unsigned int *) patchinfofromfile;
            patchinfofromfile += sizeof(unsigned int);
            const PVPatch * patch = patches[arborId][patchindex];
            if (offset != patch->offset ||
                nx != patch->nx ||
                ny != patch->ny) {
               pvErrorNoExit().printf("readWeightsDeprecated: Rank %d process, patch %d: geometry from filename \"%s\" is not consistent with geometry from patches input argument: ", comm->commRank(), patchindex, filename);
               pvErrorNoExit().printf("filename has nx=%hu, ny=%hu, offset=%u; patches[%d] has nx=%hu, ny=%hu, offset=%u.\n", nx, ny, offset, patchindex, patch->nx, patch->ny, patch->offset);
               status = PV_FAILURE;
            }
            patchinfofromfile += (size_t) (nxp*nyp*nfp)*pv_sizeof(header_data_type);
         }
      }
      if (status != PV_SUCCESS) {
          exit(EXIT_FAILURE);
      }
      status = pvp_set_patches(cbuf, patches ? patches[arborId] : NULL, dataStart[arborId], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
      if (status != PV_SUCCESS) {
         pvErrorNoExit().printf("[%2d]: readWeightsDeprecated: failed in pvp_set_patches, numPatches==%d\n",
                 comm->commRank(), numPatches);
      }
   } // loop over arborId
   free(cbuf);
   status = pvp_close_file(pvstream, comm)==PV_SUCCESS ? status : PV_FAILURE;
   return status;
}
#endif // OBSOLETE // Marked obsolete Jun 27, 2016.
/**
 * @fd
 * @patch
 */
int pv_text_write_patch(OutStream * outStream, PVPatch * patch, pvwdata_t * data, int nf, int sx, int sy, int sf)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   //const int nf = (int) patch->nf;

   //const int sx = (int) patch->sx;  assert(sx == nf);
   //const int sy = (int) patch->sy;  //assert(sy == nf*nx);
   //const int sf = (int) patch->sf;  assert(sf == 1);

   assert(outStream != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            outStream->printf("%7.5f ", data[i*sx + j*sy + f*sf]);
         }
         outStream->printf("\n");
      }
      outStream->printf("\n");
   }

   return 0;
}

int writeWeights(const char * filename, Communicator * comm, double timed, bool append,
      const PVLayerLoc * preLoc, const PVLayerLoc * postLoc, int nxp, int nyp, int nfp, float minVal, float maxVal,
      PVPatch *** patches, pvwdata_t ** dataStart, int numPatches, int numArbors, bool compress, int file_type) {
   int status = PV_SUCCESS;

   int datatype = compress ? PV_BYTE_TYPE : PV_FLOAT_TYPE;

   const int icRank = comm->commRank();

   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize = pv_sizeof_patch(numPatchItems, datatype);
   const size_t localSize = numPatches * patchSize;

   unsigned char * cbuf = (unsigned char *) malloc(localSize);
   if(cbuf == NULL) {
      pvError(errorMessage);
      errorMessage.printf("Rank %d: writeWeights unable to allocate memory to write to \"%s\": %s", icRank, filename, strerror(errno));
      errorMessage.printf("    (nxp=%d, nyp=%d, nfp=%d, numPatchItems=%d, writing weights as %s)\n", nxp, nyp, nfp, numPatchItems, compress ? "bytes" : "floats");
   }

#ifdef PV_USE_MPI
   const int tagbase = file_type; // PVP_WGT_FILE_TYPE;
   const MPI_Comm mpi_comm = comm->communicator();
#endif // PV_USE_MPI
   if( icRank > 0 ) {
#ifdef PV_USE_MPI
      if( file_type != PVP_KERNEL_FILE_TYPE ) { // No MPI needed for kernel weights (sharedWeights==true).  If keepKernelsSynchronized is false, synchronize kernels before entering PV::writeWeights()
         const int dest = 0;
         for( int arbor=0; arbor<numArbors; arbor++ ) {
            pvp_copy_patches(cbuf, patches[arbor], dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
            MPI_Send(cbuf, localSize, MPI_BYTE, dest, tagbase+arbor, mpi_comm);
#ifdef DEBUG_OUTPUT
            pvDebug().printf("[%2d]: writeWeights: sent to 0, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
                    comm->commRank(), nxBlocks, nyBlocks, numPatches);
#endif // DEBUG_OUTPUT
         }
      }
#endif // PV_USE_MPI
   } // icRank > 0
   else /* icRank==0 */ {
      void * wgtExtraParams = calloc(NUM_WGT_EXTRA_PARAMS, sizeof(int));
      if (wgtExtraParams == NULL) {
         pvError().printf("writeWeights unable to allocate memory to write weight params to \"%s\": %s\n", filename, strerror(errno));
      }
      int * wgtExtraIntParams = (int *) wgtExtraParams;
      float * wgtExtraFloatParams = (float *) wgtExtraParams;

      int numParams = NUM_WGT_PARAMS;

      PV_Stream * pvstream = pvp_open_write_file(filename, comm, append);

      if (pvstream == NULL) {
         pvErrorNoExit().printf("PV::writeWeights: unable to open file \"%s\"\n", filename);
         return -1;
      }
      if (append) PV_fseek(pvstream, 0L, SEEK_END); // If append is true we open in "r+" mode so we need to move to the end of the file.

      int numGlobalPatches;
      bool asPostWeights;
      switch(file_type) {
      case PVP_WGT_FILE_TYPE:
         if (numPatches == (preLoc->nx+preLoc->halo.lt+preLoc->halo.rt)*(preLoc->ny+preLoc->halo.dn+preLoc->halo.up)*preLoc->nf) {
            asPostWeights = false;
         }
         else if (numPatches == preLoc->nx*preLoc->ny*preLoc->nf) {
            asPostWeights = true;
         }
         else {
            pvErrorNoExit().printf("writeWeights: in file \"%s\", numPatches %d is not compatible with layer dimensions nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n",
                  filename, numPatches, preLoc->nx, preLoc->ny, preLoc->nf, preLoc->halo.lt, preLoc->halo.rt, preLoc->halo.dn, preLoc->halo.up);
         }
         numGlobalPatches = getNumGlobalPatches(preLoc, asPostWeights);
         break;
      case PVP_KERNEL_FILE_TYPE:
         asPostWeights = false;
         numGlobalPatches = numPatches;
         break;
      default:
         assert(0); // only possibilities for file_type are WGT and KERNEL
         break;
      }
      size_t globalSize = numGlobalPatches * patchSize;
      status = pvp_write_header(pvstream, comm, timed, preLoc, file_type,
                                datatype, numArbors, true/*extended*/, true/*contiguous*/, numParams, globalSize);
      if (status != PV_SUCCESS) {
         pvError().printf("Error writing writing weights header to \"%s\".\n", filename);
      }

      // write extra weight parameters
      //
      wgtExtraIntParams[INDEX_WGT_NXP] = nxp;
      wgtExtraIntParams[INDEX_WGT_NYP] = nyp;
      wgtExtraIntParams[INDEX_WGT_NFP] = nfp;

      wgtExtraFloatParams[INDEX_WGT_MIN] = minVal;
      wgtExtraFloatParams[INDEX_WGT_MAX] = maxVal;
      wgtExtraIntParams[INDEX_WGT_NUMPATCHES] = numGlobalPatches;

      numParams = NUM_WGT_EXTRA_PARAMS;
      unsigned int num_written = PV_fwrite(wgtExtraParams, sizeof(int), numParams, pvstream);
      free(wgtExtraParams); wgtExtraParams=NULL; wgtExtraIntParams=NULL; wgtExtraFloatParams=NULL;
      if ( num_written != (unsigned int) numParams ) {
         pvErrorNoExit().printf("PV::writeWeights: unable to write weight header to file %s\n", filename);
         return -1;
      }

      for (int arbor=0; arbor<numArbors; arbor++) {
         PVPatch ** arborPatches = file_type == PVP_KERNEL_FILE_TYPE ? NULL : patches[arbor];
         if (file_type == PVP_KERNEL_FILE_TYPE) {
            pvp_copy_patches(cbuf, arborPatches, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
            size_t numfwritten = PV_fwrite(cbuf, localSize, 1, pvstream);
            if (numfwritten != 1) {
               pvErrorNoExit().printf("PV::writeWeights: unable to write weight data to file %s\n", filename);
               return -1;
            }
         }
         else {
            assert(file_type == PVP_WGT_FILE_TYPE);
            long int arborstartfile = getPV_StreamFilepos(pvstream);
            long int arborendfile = arborstartfile + globalSize;
            PV_fseek(pvstream, arborendfile-1, SEEK_SET);
            char endarborchar = (char) 0;
            PV_fwrite(&endarborchar, 1, 1, pvstream); // Makes sure the file is the correct length even if the last patch is shrunken
            for (int proc=0; proc<comm->commSize(); proc++) {
#ifdef PV_USE_MPI
               if (proc==0) /*local portion*/ {
                  pvp_copy_patches(cbuf, arborPatches, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
               }
               else /*receive other portion via MPI*/ {
                  MPI_Recv(cbuf, localSize, MPI_BYTE, proc, tagbase+arbor, mpi_comm, MPI_STATUS_IGNORE);
               }
#else // PV_USE_MPI
               assert(proc==0);
               pvp_copy_patches(cbuf, arborPatches, dataStart[arbor], numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
#endif // PV_USE_MPI
               int procrow = rowFromRank(proc, comm->numCommRows(), comm->numCommColumns());
               int proccolumn = columnFromRank(proc, comm->numCommRows(), comm->numCommColumns());
               for (int k=0; k<numPatches; k++) {
                  unsigned char * cbufpatch = &cbuf[k*patchSize];
                  int globalIndex;
                  if (asPostWeights) {
                     int x = kxPos(k, preLoc->nx, preLoc->ny, preLoc->nf)+proccolumn*preLoc->nx;
                     int y = kyPos(k, preLoc->nx, preLoc->ny, preLoc->nf)+procrow*preLoc->ny;
                     int f = featureIndex(k, preLoc->nx, preLoc->ny, preLoc->nf);
                     globalIndex = kIndex(x,y,f,preLoc->nxGlobal,preLoc->nyGlobal,preLoc->nf);
                  }
                  else {
                     int x = kxPos(k, preLoc->nx+preLoc->halo.lt+preLoc->halo.rt, preLoc->ny+preLoc->halo.dn+preLoc->halo.up, preLoc->nf)+proccolumn*preLoc->nx;
                     int y = kyPos(k, preLoc->nx+preLoc->halo.lt+preLoc->halo.rt, preLoc->ny+preLoc->halo.dn+preLoc->halo.up, preLoc->nf)+procrow*preLoc->ny;
                     int f = featureIndex(k, preLoc->nx+preLoc->halo.lt+preLoc->halo.rt, preLoc->ny+preLoc->halo.dn+preLoc->halo.up, preLoc->nf);
                     globalIndex = kIndex(x, y, f, preLoc->nxGlobal+preLoc->halo.lt+preLoc->halo.rt, preLoc->nyGlobal+preLoc->halo.dn+preLoc->halo.up, preLoc->nf);
                  }
                  unsigned short int pnx = *(unsigned short int *) (cbuf);
                  unsigned short int pny = *(unsigned short int *) (cbuf+sizeof(unsigned short int));
                  unsigned int p_offset = *(unsigned int *) (cbuf+2*sizeof(unsigned short int));
                  if (pnx == nxp && pny == nyp) /*not shrunken*/ {
                     PV_fseek(pvstream, arborstartfile+globalIndex*patchSize, SEEK_SET); // TODO: error handling
                     PV_fwrite(cbufpatch, patchSize, (size_t) 1, pvstream); // TODO: error handling
                  }
                  else {
                     PV_fseek(pvstream, arborstartfile+k*patchSize, SEEK_SET); // TODO: error handling
                     size_t const patchheadersize = 2*sizeof(short int)+sizeof(int);
                     PV_fwrite(cbufpatch, patchheadersize, 1, pvstream); // TODO: error handling
                     int datasize = pv_sizeof(datatype);
                     const int syw = nfp*nxp;
                     for (int y=0; y<pny; y++) {
                        unsigned int memoffset = patchheadersize + (p_offset+y*syw)*datasize;
                        PV_fseek(pvstream, arborstartfile+globalIndex*patchSize+memoffset, SEEK_SET); // TODO: error handling
                        PV_fwrite(cbufpatch + memoffset, pnx*nfp*datasize, 1, pvstream); // TODO: error handling
                     } // Loop over line within patch
                  } // if (p->nx == nxp && p->ny == nyp)
               } // Loop over patches
            } // Loop over processes
            PV_fseek(pvstream, arborendfile, SEEK_SET);
         } // if-statement for file_type
      } // loop over arbors
      pvp_close_file(pvstream, comm);
   } // if-statement for process rank

   free(cbuf); cbuf = NULL;

   return status;
}

int writeRandState(const char * filename, Communicator * comm, taus_uint4 * randState, const PVLayerLoc * loc, bool isExtended, bool verifyWrites) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   PV_Stream * pvstream = NULL;
   if (rank == rootproc) {
      pvstream = PV_fopen(filename, "w", verifyWrites);
      if (pvstream==NULL) {
         pvErrorNoExit().printf("writeRandState: unable to open path %s for writing.\n", filename);
         abort();
      }
   }

   for(int b = 0; b < loc->nbatch; b++){
      taus_uint4 * randStateBatch;
      if(isExtended){
         randStateBatch = randState + b * (loc->nx + loc->halo.rt + loc->halo.lt) * (loc->ny + loc->halo.up + loc->halo.dn) * loc->nf;
      }
      else{
         randStateBatch = randState + b * loc->nx * loc->ny * loc->nf;
      }
      status = gatherActivity(pvstream, comm, rootproc, randStateBatch, loc, isExtended/*extended*/);
   }
   if (rank==rootproc) {
      PV_fclose(pvstream); pvstream = NULL;
   }
   return status;
}

int readRandState(const char * filename, Communicator * comm, taus_uint4 * randState, const PVLayerLoc * loc, bool isExtended) {
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();

   PV_Stream * pvstream = NULL;
   if (rank == rootproc) {
      pvstream = PV_fopen(filename, "r", false/*verifyWrites*/);
      if (pvstream==NULL) {
         pvErrorNoExit().printf("readRandState: unable to open path %s for reading.\n", filename);
         abort();
      }
   }
   for(int b = 0; b < loc->nbatch; b++){
      taus_uint4 * randStateBatch;
      if(isExtended){
         randStateBatch = randState + b * (loc->nx + loc->halo.rt + loc->halo.lt) * (loc->ny + loc->halo.up + loc->halo.dn) * loc->nf;
      }
      else{
         randStateBatch = randState + b * loc->nx * loc->ny * loc->nf;
      }
      status = scatterActivity(pvstream, comm, rootproc, randStateBatch, loc, isExtended/*extended*/);
   }
   if (rank==rootproc) {
      PV_fclose(pvstream); pvstream = NULL;
   }
   return status;
}

template <typename T> int gatherActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended) {
   // In MPI when this process is called, all processes must call it.
   // Only the root process uses the file pointer.
   int status = PV_SUCCESS;

   int numLocalNeurons = layerLoc->nx * layerLoc->ny * layerLoc->nf;

   int xLineStart = 0;
   int yLineStart = 0;
   int xBufSize = layerLoc->nx;
   int yBufSize = layerLoc->ny;
   PVHalo halo;
   if (extended) {
      memcpy(&halo,&layerLoc->halo,sizeof(halo));
      xLineStart = halo.lt;
      yLineStart = halo.up;
      xBufSize += halo.lt+halo.rt;
      yBufSize += halo.dn+halo.up;
   }
   else {
      halo.lt = halo.rt = halo.dn = halo.up = 0;
   }

   int linesize = layerLoc->nx*layerLoc->nf; // All values across x and f for a specific y are contiguous; do a single write for each y.
   size_t datasize = sizeof(T);
   // read into a temporary buffer since buffer may be extended but the file only contains the restricted part.
   T * temp_buffer = (T *) calloc(numLocalNeurons, datasize);
   if (temp_buffer==NULL) {
      status = PV_FAILURE;
      pvError().printf("gatherActivity unable to allocate memory for temp_buffer.\n");
   }

   int rank = comm->commRank();
   if (rank==rootproc) {
      if (pvstream == NULL) {
         status = PV_FAILURE;
         pvError().printf("gatherActivity: file pointer on root process is null.\n");
      }
      long startpos = getPV_StreamFilepos(pvstream);
      if (startpos == -1) {
         status = PV_FAILURE;
         pvErrorNoExit().printf("gatherActivity: failure getting file position: %s\n", strerror(errno));
      }
      // Write zeroes to make sure the file is big enough since we'll write nonsequentially under MPI.  This may not be necessary.
      int comm_size = comm->commSize();
      for (int r=0; r<comm_size; r++) {
         int numwritten = PV_fwrite(temp_buffer, datasize, numLocalNeurons, pvstream);
         if (numwritten != numLocalNeurons) {
            status = PV_FAILURE;
            pvError().printf("gatherActivity error when writing: number of bytes attempted %d, number written %d\n", numwritten, numLocalNeurons);
         }
      }
      int fseekstatus = PV_fseek(pvstream, startpos, SEEK_SET);
      if (fseekstatus != 0) {
         status = PV_FAILURE;
         pvError().printf("gatherActivity error when setting file position: %s\n", strerror(errno));
      }

      for (int r=0; r<comm_size; r++) {
         if (r==rootproc) {
            if (extended) {
               for (int y=0; y<layerLoc->ny; y++) {
                  int k_extended = kIndex(halo.lt, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
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
                  pvErrorNoExit().printf("gatherActivity failure writing to \"%s\": number of bytes attempted %zu, number written %d\n", pvstream->name, datasize*linesize, numwritten);
                  status = PV_FAILURE;
               }
            }
            else {
               status = PV_FAILURE;
               pvErrorNoExit().printf("gatherActivity failure setting file position: %s\n", strerror(errno));
            }
         }
      }
      PV_fseek(pvstream, startpos+numLocalNeurons*datasize*comm_size, SEEK_SET);
   }
   else {
      if (halo.lt || halo.rt || halo.dn || halo.up) {
         // temp_buffer is a restricted buffer, but if extended is true, buffer is an extended buffer.
         for (int y=0; y<layerLoc->ny; y++) {
            int k_extended = kIndex(halo.lt, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
            int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            memcpy(&temp_buffer[k_restricted], &buffer[k_extended], datasize*linesize);
         }
         MPI_Send(temp_buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
      else {
         MPI_Send(buffer, numLocalNeurons*datasize, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator());
      }
   }

   free(temp_buffer); temp_buffer = NULL;
   return status;
}

// Declare the instantiations of gatherActivity that occur in other .cpp files; otherwise you may get linker errors.
template int gatherActivity<unsigned char>(PV_Stream * pvstream, Communicator * comm, int rootproc, unsigned char * buffer, const PVLayerLoc * layerLoc, bool extended);
template int gatherActivity<pvdata_t>(PV_Stream * pvstream, Communicator * comm, int rootproc, pvdata_t * buffer, const PVLayerLoc * layerLoc, bool extended);
template int gatherActivity<taus_uint4>(PV_Stream * pvstream, Communicator * comm, int rootproc, taus_uint4 * buffer, const PVLayerLoc * layerLoc, bool extended);

template <typename T> int scatterActivity(PV_Stream * pvstream, Communicator * comm, int rootproc, T * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc, int offsetX, int offsetY, int filetype, int numActive) {
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
   int foo;
   int numLocalNeurons = layerLoc->nx * layerLoc->ny * layerLoc->nf;
   size_t datasize;
   int xLineStart = 0;
   int yLineStart = 0;
   int xBufSize = layerLoc->nx;
   int yBufSize = layerLoc->ny;
   int kx0;
   int ky0;
   int * activeNeurons;
   T * TBuff;
   pvdata_t * TBuff1;
   PVHalo halo;
   if (extended) {
      memcpy(&halo, &layerLoc->halo, sizeof(halo));
      xLineStart = halo.lt;
      yLineStart = halo.up;
      xBufSize += halo.lt+halo.rt;
      yBufSize += halo.dn+halo.up;
   }
   int linesize = layerLoc->nx * layerLoc->nf;
   switch (filetype) {
   case PVP_NONSPIKING_ACT_FILE_TYPE:
      datasize = sizeof(T);
      TBuff =  (T *) calloc(numLocalNeurons, datasize);
      if (TBuff==NULL) {
         status = PV_FAILURE;
         pvErrorNoExit().printf("scatterActivity unable to allocate memory for temp_buffer.\n");
      }
      break;
   case PVP_ACT_FILE_TYPE:
   case PVP_ACT_SPARSEVALUES_FILE_TYPE:
      datasize = sizeof(pvdata_t);
      TBuff1 =  (pvdata_t *) calloc(numLocalNeurons, datasize);
      if (TBuff1==NULL) {
         status = PV_FAILURE;
         pvError().printf("scatterActivity unable to allocate memory for temp_buffer.\n");
      }
      break;
   }

   int rank = comm->commRank();
   if (rank==rootproc) {
      if (pvstream == NULL) {
         status = PV_FAILURE;
         pvError().printf("scatterActivity: file pointer on root process is null.\n");
      }
      long startpos = getPV_StreamFilepos(pvstream);
      if (startpos == -1) {
         status = PV_FAILURE;
         pvError().printf("scatterActivity unable to get file position: %s\n", strerror(errno));
      }
      if (fileLoc==NULL) fileLoc = layerLoc;
      if (fileLoc->nf != layerLoc->nf) {
         pvError().printf("scatterActivity: layerLoc->nf and fileLoc->nf must be equal (they are %d and %d)\n", layerLoc->nf, fileLoc->nf);
      }

      int comm_size = comm->commSize();
      switch (filetype) {
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         //TODO for non-spiking
         if (offsetX < 0 || offsetX + layerLoc->nxGlobal > fileLoc->nxGlobal ||
               offsetY < 0 || offsetY + layerLoc->nyGlobal > fileLoc->nyGlobal) {
            pvError().printf("scatterActivity error: offset window does not completely fit inside frame defined by image file \"%s\". This case has not been implemented yet for nonspiking activity files.\n", pvstream->name);
         }

         for (int r=0; r<comm_size; r++) {
            if (r==rootproc) continue; // Need to load root process last, or subsequent processes will clobber temp_buffer.
            for (int y=0; y<layerLoc->ny; y++) {
               int ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
               int kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
               int k_inmemory = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
               int k_infile = kIndex(offsetX+kx0, offsetY+ky0+y, 0, fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
               PV_fseek(pvstream, startpos + k_infile*(long) datasize, SEEK_SET);
               int numread = PV_fread(&TBuff[k_inmemory], datasize, linesize, pvstream);
               if (numread != linesize) {
                  pvError().printf("scatterActivity error when reading: number of bytes attempted %d, number read %d\n", numread, numLocalNeurons);
               }
            }
            MPI_Send(TBuff, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
         }
         for (int y=0; y<layerLoc->ny; y++) {
            int ky0 = layerLoc->ny*rowFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
            int kx0 = layerLoc->nx*columnFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
            int k_inmemory = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
            int k_infile = kIndex(offsetX+kx0, offsetY+ky0+y, 0, fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
            PV_fseek(pvstream, startpos + k_infile*(long) datasize, SEEK_SET);
            int numread = PV_fread(&TBuff[k_inmemory], datasize, linesize, pvstream);
            if (numread != linesize) {
               pvError().printf("scatterActivity error when reading: number of bytes attempted %d, number written %d\n", linesize, numread);
            }
         }
         PV_fseek(pvstream, startpos+numLocalNeurons*datasize*comm_size, SEEK_SET);
         break;
      case PVP_ACT_FILE_TYPE:

         //Read list of active neurons
         activeNeurons = (int *) calloc(numActive,datasize);
         foo = PV_fread(activeNeurons, datasize, numActive, pvstream);
         // Root process constructs buffers of other processes
         for (int r=0; r<comm_size; r++) {
            if (r==rootproc) continue; // Need to load root process last, or subsequent processes will clobber temp_buffer.
            // Global X and Y coordinates of "top left" of process
            ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
            kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
            // Loop through active neurons, calculate their global coordinates, if process contains a position of an active neuron,
            // set its value = 1
            for (int i = 0; i < numActive; i++) {
               int xpos = kxPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
               int ypos = kyPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
                  if((xpos >= kx0+offsetX) && (xpos < kx0+layerLoc->nx+offsetX) && (ypos >= ky0+offsetY) && (ypos < ky0 + layerLoc->ny+offsetY)) {
                     int fpos = featureIndex(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
                     TBuff1[(ypos-ky0-offsetY)*linesize + (xpos-kx0-offsetX)*layerLoc->nf + fpos] = 1;
                  }
            }
            //Send buffer to appropriate mpi process
            MPI_Send(TBuff1, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
            //Clear the buffer so rootproc can calculate the next process's buffer.
            for (int i = 0; i < numLocalNeurons; i++) {
               TBuff1[i] = 0;
            }
         }
         // Same thing for root process
         ky0 = layerLoc->ny*rowFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         kx0 = layerLoc->nx*columnFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         for (int i = 0; i < numActive; i++) {
             int xpos = kxPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
             int ypos = kyPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
                if((xpos >= kx0+offsetX) && (xpos < kx0+layerLoc->nx+offsetX) && (ypos >= ky0+offsetY) && (ypos < ky0 + layerLoc->ny+offsetY)) {
                   int fpos = featureIndex(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
                   TBuff1[(ypos-ky0-offsetY)*linesize + (xpos-kx0-offsetX)*layerLoc->nf + fpos] = 1;
                }
          }
         free(activeNeurons);
         break;

     case PVP_ACT_SPARSEVALUES_FILE_TYPE:
         //Read list of active neurons and their values
         activeNeurons = (int *) calloc(numActive,datasize);
         if (activeNeurons==NULL) {
            pvError().printf("scatterActivity error for \"%s\": unable to allocate buffer for active neuron locations: %s\n", pvstream->name, strerror(errno));
         }
         pvdata_t * vals =  (pvdata_t *) calloc(numActive, datasize);
         if (activeNeurons==NULL) {
            pvError().printf("scatterActivity error for \"%s\": unable to allocate buffer for active neuron values: %s\n", pvstream->name, strerror(errno));
         }
         for (int i = 0; i < numActive; i++) {
            foo = PV_fread(&activeNeurons[i], datasize, 1, pvstream);
            foo = PV_fread(&vals[i], datasize, 1, pvstream);
         }
         // Root process constructs buffers of other processes
         for (int r=0; r<comm_size; r++) {
            if (r==rootproc) continue; // Need to load root process last, or subsequent processes will clobber temp_buffer.
            // Global X and Y coordinates of "top left" of process
            ky0 = layerLoc->ny*rowFromRank(r, comm->numCommRows(), comm->numCommColumns());
            kx0 = layerLoc->nx*columnFromRank(r, comm->numCommRows(), comm->numCommColumns());
            // Loop through active neurons, calculate their global coordinates, if process contains a position of an active neuron,
            // set its value
            for (int i = 0; i < numActive; i++) {
               int xpos = kxPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
               int ypos = kyPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
               if((xpos >= kx0+offsetX) && (xpos < kx0+layerLoc->nx+offsetX) && (ypos >= ky0+offsetY) && (ypos < ky0 + layerLoc->ny+offsetY)) {
                  int fpos = featureIndex(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
                  TBuff1[(ypos-ky0-offsetY)*linesize + (xpos-kx0-offsetX)*layerLoc->nf + fpos] = vals[i];
               }
            }
            //Send buffer to appropriate mpi process
            MPI_Send(TBuff1, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
            //Clear the buffer so rootproc can calculate the next process's buffer.
            for (int i = 0; i < numLocalNeurons; i++) {
               TBuff1[i] = 0;
            }
         }
         // Same thing for root process
         ky0 = layerLoc->ny*rowFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         kx0 = layerLoc->nx*columnFromRank(rootproc, comm->numCommRows(), comm->numCommColumns());
         for (int i = 0; i < numActive; i++) {
            int xpos = kxPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
            int ypos = kyPos(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
            if((xpos >= kx0+offsetX) && (xpos < kx0+layerLoc->nx+offsetX) && (ypos >= ky0+offsetY) && (ypos < ky0 + layerLoc->ny+offsetY)) {
               int fpos = featureIndex(activeNeurons[i], fileLoc->nxGlobal, fileLoc->nyGlobal, layerLoc->nf);
               //Testing this
               TBuff1[(ypos-ky0-offsetY)*linesize + (xpos-kx0-offsetX)*layerLoc->nf + fpos] = vals[i];
            }
         }
         free(activeNeurons);
         free(vals);
         break;
      }  //switch filetype
   }  //rank==rootproc
   else {
      switch (filetype) {
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         MPI_Recv(TBuff, datasize*numLocalNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
         break;
      case PVP_ACT_FILE_TYPE:
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:
         //Receive buffers from rootproc
         MPI_Recv(TBuff1, datasize*numLocalNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
         break;
      }

   }  //rank==rootproc

   // At this point, each process has the data, as a restricted layer, in temp_buffer.
   // Each process now copies the data to buffer, which may be extended.
   if (extended) {
      for (int y=0; y<layerLoc->ny; y++) {
         int k_extended = kIndex(xLineStart, y+yLineStart, 0, xBufSize, yBufSize, layerLoc->nf);
         int k_restricted = kIndex(0, y, 0, layerLoc->nx, layerLoc->ny, layerLoc->nf);
         switch (filetype) {
         case PVP_NONSPIKING_ACT_FILE_TYPE:
            memcpy(&buffer[k_extended], &TBuff[k_restricted], (size_t)linesize*datasize);
            break;
         case PVP_ACT_FILE_TYPE:
         case PVP_ACT_SPARSEVALUES_FILE_TYPE:
            memcpy(&buffer[k_extended], &TBuff1[k_restricted], (size_t)linesize*datasize);
            break;
         }

      }
   }
   else {
      switch (filetype) {
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         memcpy(buffer, TBuff, (size_t) numLocalNeurons*datasize);
         break;
      case PVP_ACT_FILE_TYPE:
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:
         memcpy(buffer, TBuff1, (size_t) numLocalNeurons*datasize);
         break;
      }
   }
   switch (filetype) {
   case PVP_NONSPIKING_ACT_FILE_TYPE:
      free(TBuff); TBuff = NULL;
      break;
   case PVP_ACT_FILE_TYPE:
   case PVP_ACT_SPARSEVALUES_FILE_TYPE:
      free(TBuff1); TBuff1 = NULL;
      break;
   }

   return status;
}
// Declare the instantiations of scatterActivity that occur in other .cpp files; otherwise you may get linker errors.
template int scatterActivity<float>(PV_Stream * pvstream, Communicator * icComm, int rootproc, float * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc, int offsetX, int offsetY, int filetype, int numActive);
// template int scatterActivity<pvdata_t>(PV_Stream * pvstream, Communicator * icComm, int rootproc, pvdata_t * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc, int offsetX, int offsetY); // duplicates float since pvdata_t is currently float, but this may in principle change
template int scatterActivity<taus_uint4>(PV_Stream * pvstream, Communicator * icComm, int rootproc, taus_uint4 * buffer, const PVLayerLoc * layerLoc, bool extended, const PVLayerLoc * fileLoc, int offsetX, int offsetY, int filetype, int numActive);

} // namespace PV
