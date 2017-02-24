/*
 * fileio.cpp
 *
 *  Created on: Oct 21, 2009
 *      Author: Craig Rasmussen
 */

#include "fileio.hpp"
#include "connections/weight_conversions.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.h"

#include <assert.h>
#include <iostream>

#undef DEBUG_OUTPUT

namespace PV {

// timeToParams and timeFromParams use memcpy instead of casting pointers
// because casting pointers violates strict aliasing.

void timeToParams(double time, void *params) { memcpy(params, &time, sizeof(double)); }

double timeFromParams(void *params) {
   double x;
   memcpy(&x, params, sizeof(double));
   return x;
}

size_t pv_sizeof(int datatype) {
   if (datatype == BufferUtils::INT) {
      return sizeof(int);
   }
   if (datatype == BufferUtils::FLOAT) {
      return sizeof(float);
   }
   if (datatype == BufferUtils::BYTE) {
      return sizeof(unsigned char);
   }
   if (datatype == BufferUtils::TAUS_UINT4) {
      return sizeof(taus_uint4);
   }

   // shouldn't arrive here
   assert(false);
   return 0;
}

/**
 * Returns the size of a patch when read or written. The size includes the size
 * of the (nxp,nyp) patch header.
 */
size_t pv_sizeof_patch(int count, int datatype) {
   return (2 * sizeof(unsigned short) + sizeof(unsigned int) + count * pv_sizeof(datatype));
}

PV_Stream *PV_fopen(const char *path, const char *mode, bool verifyWrites) {
   if (mode == NULL) {
      ErrorLog().printf("PV_fopen: mode argument must be a string (path was \"%s\").\n", path);
      errno = EINVAL;
      return NULL;
   }
   char *realPath  = strdup(expandLeadingTilde(path).c_str());
   long filepos    = 0L;
   long filelength = 0L;
   if (mode[0] == 'r' || mode[0] == 'a') {
      struct stat statbuf;
      int statstatus = stat(realPath, &statbuf);
      if (statstatus == 0) {
         filelength = (long)statbuf.st_size;
         if (mode[0] == 'a') {
            filepos = filelength;
         }
      }
      else if (errno != ENOENT) {
         Fatal().printf(
               "PV_fopen: unable to stat \"%s\" with mode \"%s\": %s\n",
               realPath,
               mode,
               strerror(errno));
      }
   }
   int fopencounts          = 0;
   PV_Stream *streampointer = NULL;
   FILE *fp                 = NULL;
   while (fp == NULL) {
      errno = 0;
      fp    = fopen(realPath, mode);
      if (fp != NULL)
         break;
      fopencounts++;
      WarnLog().printf(
            "fopen failure for \"%s\" on attempt %d: %s\n", realPath, fopencounts, strerror(errno));
      if (fopencounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fp == NULL) {
      ErrorLog().printf(
            "PV_fopen: exceeded MAX_FILESYSTEMCALL_TRIES = %d attempting to open \"%s\"\n",
            MAX_FILESYSTEMCALL_TRIES,
            realPath);
   }
   else {
      if (fopencounts > 0) {
         WarnLog().printf("fopen succeeded for \"%s\" on attempt %d\n", realPath, fopencounts + 1);
      }
      streampointer = (PV_Stream *)calloc(1, sizeof(PV_Stream));
      if (streampointer != NULL) {
         streampointer->name         = strdup(realPath);
         streampointer->mode         = strdup(mode);
         streampointer->fp           = fp;
         streampointer->filepos      = filepos;
         streampointer->filelength   = filelength;
         streampointer->isfile       = 1;
         streampointer->verifyWrites = verifyWrites;
      }
      else {
         ErrorLog().printf("PV_fopen failure for \"%s\": %s\n", realPath, strerror(errno));
         fclose(fp);
      }
   }
   free(realPath);
   return streampointer;
}

int PV_stat(const char *path, struct stat *buf) {
   // Call stat library function, trying up to MAX_FILESYSTEMCALL_TRIES times if an error is
   // returned.
   // If an error results on all MAX_FILESYSTEMCALL_TRIES times, returns -1 (the error return value)
   // for stat()
   // and errno is the error of the last attempt.
   char *realPath = strdup(expandLeadingTilde(path).c_str());
   int attempt    = 0;
   int retval     = -1;
   while (retval != 0) {
      errno  = 0;
      retval = stat(realPath, buf);
      if (retval == 0)
         break;
      attempt++;
      WarnLog().printf(
            "stat() failure for \"%s\" on attempt %d: %s\n", path, attempt, strerror(errno));
      if (attempt < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (retval != 0) {
      ErrorLog().printf(
            "PV_stat exceeded MAX_FILESYSTEMCALL_TRIES = %d for \"%s\"\n",
            MAX_FILESYSTEMCALL_TRIES,
            path);
   }
   free(realPath);
   return retval;
}

long int PV_ftell_primitive(PV_Stream *pvstream) {
   // Calls ftell() and returns value ftell returns, but doesn't compare or change stream's fpos
   int ftellcounts = 0;
   long filepos    = -1;
   while (filepos < 0) {
      errno   = 0;
      filepos = ftell(pvstream->fp);
      if (filepos >= 0)
         break;
      ftellcounts++;
      WarnLog().printf(
            "ftell failure for \"%s\" on attempt %d: %s\n",
            pvstream->name,
            ftellcounts,
            strerror(errno));
      if (ftellcounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (filepos < 0) {
      ErrorLog().printf(
            "PV_ftell failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n",
            pvstream->name,
            MAX_FILESYSTEMCALL_TRIES);
   }
   else if (ftellcounts > 0) {
      WarnLog().printf(
            "PV_ftell succeeded for \"%s\" on attempt %d", pvstream->name, ftellcounts + 1);
   }
   return filepos;
}

long int getPV_StreamFilepos(PV_Stream *pvstream) { return pvstream->filepos; }

// Use getPV_StreamFilepos instead of PV_ftell whenever possible, since NMC cluster's ftell is
// currently unreliable
long int PV_ftell(PV_Stream *pvstream) {
   long int filepos = PV_ftell_primitive(pvstream);
   if (pvstream->filepos != filepos) {
      WarnLog().printf(
            "ftell for \"%s\" returned %ld instead of the expected %ld\n",
            pvstream->name,
            filepos,
            pvstream->filepos);
   }
   return filepos;
}

int PV_fseek(PV_Stream *pvstream, long offset, int whence) {
   int fseekcounts = 0;
   int fseekstatus = -1;
   while (fseekstatus != 0) {
      errno       = 0;
      fseekstatus = fseek(pvstream->fp, offset, whence);
      if (fseekstatus == 0)
         break;
      fseekcounts++;
      WarnLog().printf(
            "fseek failure for \"%s\" on attempt %d: %s\n",
            pvstream->name,
            fseekcounts,
            strerror(errno));
      if (fseekcounts < MAX_FILESYSTEMCALL_TRIES) {
         sleep(1);
      }
      else {
         break;
      }
   }
   if (fseekstatus != 0) {
      ErrorLog().printf(
            "PV_fseek failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded\n",
            pvstream->name,
            MAX_FILESYSTEMCALL_TRIES);
   }
   else if (fseekcounts > 0) {
      WarnLog().printf(
            "PV_fseek succeeded for \"%s\" on attempt %d\n", pvstream->name, fseekcounts + 1);
   }
   if (pvstream->mode[0] != 'a') {
      switch (whence) {
         case SEEK_SET: pvstream->filepos = offset; break;
         case SEEK_CUR: pvstream->filepos += offset; break;
         case SEEK_END: pvstream->filepos = pvstream->filelength + offset; break;
         default: assert(0); break;
      }
   }
   return fseekstatus;
}

/**
 * @brief A wrapper for fwrite() with feedback for errors and the possibility of error recovery.
 * @detail The syntax and purpose of PV_fwrite follows that of the standard C function fwrite(),
 * with the following changes.
 * The FILE* argument is replaced with a PV_Stream* pointer, and the additional argument verify
 * (which defaults to true)
 * provides some error checking.
 *
 * The function calls fwrite().  If it gets an error, it tries again, up to 5 times (the number is
 * controlled by
 * the preprocessor directive MAX_FILESYSTEMCALL_TRIES).  If it fails all 5 times, it fseeks to the
 * position
 * it was in at the start of the call, and returns zero.  If it succeeds in any of the 5 times, it
 * returns nitems, and
 * the file position is at the end of the written data.
 *
 * If verify is true and pvstream is a file (isfile is true), then after writing, the file is opened
 * for reading
 * and the size*nitems characters are compared to the write buffer.  If there is an error reading
 * the data back
 * or the data read back does not match the data written, the function returns zero and the fseek()
 * is called
 * to restore the file to the position it was in at the start of the call.
 *
 * NOTE: the purpose of this wrapper is to provide some attempts at recovery if a file system is
 * imperfect (such as the one we've struggled with).
 * We hope that a successful return value indicates actual success and that the feedback provided by
 * failures prove helpful.
 * However, the function cannot guarantee recovery from errors.
 */
size_t
PV_fwrite(const void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream) {
   assert(ferror(pvstream->fp) == 0);
   int fwritecounts            = 0;
   size_t writesize            = nitems * size;
   size_t charswritten         = (size_t)0;
   const char *RESTRICT curptr = (const char *RESTRICT)ptr;
   long int fpos               = pvstream->filepos;
   if (fpos < 0) {
      Fatal().printf(
            "PV_fwrite error: unable to determine file position of \"%s\".  Fatal error\n",
            pvstream->name);
   }
   long int ftellresult = ftell(pvstream->fp);
   if (pvstream->isfile && fpos != ftellresult) {
      Fatal().printf(
            "PV_fwrite error for \"%s\": fpos = %ld but ftell() returned %ld\n",
            pvstream->name,
            fpos,
            ftellresult);
      exit(EXIT_FAILURE);
   }
   bool hasfailed = false;
   for (int fwritecounts = 1; fwritecounts <= MAX_FILESYSTEMCALL_TRIES; fwritecounts++) {
      charswritten = fwrite(ptr, 1UL, writesize, pvstream->fp);
      if (charswritten == writesize) {
         if (hasfailed) {
            clearerr(pvstream->fp);
            WarnLog().printf(
                  "fwrite succeeded for \"%s\" on attempt %d.\n", pvstream->name, fwritecounts);
         }
         break;
      }
      else {
         hasfailed = true;
         WarnLog(fwriteFailure);
         fwriteFailure.printf(
               "fwrite failure for \"%s\" on attempt %d.  Return value %zu instead of %zu.  ",
               pvstream->name,
               fwritecounts,
               charswritten,
               writesize);
         if (ferror(pvstream->fp)) {
            fwriteFailure.printf("   Error: %s\n", strerror(errno));
         }
         if (fwritecounts < MAX_FILESYSTEMCALL_TRIES) {
            fwriteFailure.printf("Retrying.\n");
            sleep(1);
            int fseekstatus = fseek(pvstream->fp, fpos, SEEK_SET);
            if (fseekstatus != 0) {
               Fatal().printf(
                     "PV_fwrite error: Unable to reset file position of \"%s\".  Fatal error: %s\n",
                     pvstream->name,
                     strerror(errno));
            }
            long int ftellreturn = ftell(pvstream->fp);
            if (fpos != ftellreturn) {
               Fatal().printf(
                     "PV_fwrite error: attempted to reset file position of \"%s\" to %ld, but "
                     "ftell() returned %ld.  Fatal error.\n",
                     pvstream->name,
                     fpos,
                     ftellreturn);
            }
         }
         else {
            ErrorLog().printf("MAX_FILESYSTEMCALL_TRIES exceeded.\n");
            return (size_t)0;
         }
      }
   }
   if (pvstream->verifyWrites && pvstream->isfile) {
      fflush(pvstream->fp);
      int status            = PV_SUCCESS;
      PV_Stream *readStream = PV_fopen(pvstream->name, "r", false /*verifyWrites*/);
      if (readStream == NULL) {
         ErrorLog().printf(
               "PV_fwrite verification: unable to open \"%s\" for reading: %s\n",
               pvstream->name,
               strerror(errno));
         status = PV_FAILURE;
      }
      if (status == PV_SUCCESS) {
         if (fseek(readStream->fp, pvstream->filepos, SEEK_SET) != 0) {
            ErrorLog().printf(
                  "PV_fwrite verification: unable to verify \"%s\" write of %zu chars from "
                  "position %ld: %s\n",
                  pvstream->name,
                  writesize,
                  pvstream->filepos,
                  strerror(errno));
            status = PV_FAILURE;
         }
      }
      char *read_buffer = NULL;
      if (status == PV_SUCCESS) {
         read_buffer = (char *)malloc(writesize);
         if (read_buffer == NULL) {
            ErrorLog().printf(
                  "PV_fwrite verification: unable to create readback buffer of size %zu to verify "
                  "\"%s\"\n",
                  writesize,
                  pvstream->name);
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         for (size_t n = 0; n < writesize; n++) {
            read_buffer[n] = ~((char *)ptr)[n];
         } // Make sure read_buffer is different from ptr before reading
      }
      if (status == PV_SUCCESS) {
         size_t numread = fread(read_buffer, (size_t)1, writesize, readStream->fp);
         if (numread != writesize) {
            ErrorLog().printf(
                  "PV_fwrite verification: unable to read into readback buffer for \"%s\": fread "
                  "returned %zu instead of %zu\n",
                  pvstream->name,
                  numread,
                  writesize);
            status = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         status = memcmp(ptr, read_buffer, writesize) == 0 ? PV_SUCCESS : PV_FAILURE;
         if (status != PV_SUCCESS) {
            size_t badcount = 0;
            for (size_t n = 0; n < writesize; n++) {
               badcount += (((char *)ptr)[n] != read_buffer[n]);
            }
            ErrorLog().printf(
                  "PV_fwrite verification: readback of %zu bytes from \"%s\" starting at position "
                  "%zu failed: %zu bytes disagree.\n",
                  writesize,
                  pvstream->name,
                  pvstream->filepos,
                  badcount);
         }
      }
      free(read_buffer);
      if (readStream) {
         PV_fclose(readStream);
         readStream = NULL;
      }
      if (status != PV_SUCCESS) {
         fseek(pvstream->fp, pvstream->filepos, SEEK_SET);
         return 0;
      }
   }
   pvstream->filepos += writesize;
   return nitems;
}

size_t PV_fread(void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream) {
   int freadcounts       = 0;
   size_t readsize       = nitems * size;
   size_t stilltoread    = readsize;
   char *RESTRICT curptr = (char *RESTRICT)ptr;
   long int fpos         = pvstream->filepos;
   clearerr(pvstream->fp);
   if (fpos < 0) {
      Fatal().printf(
            "PV_fread error: unable to determine file position of \"%s\".  Fatal error\n",
            pvstream->name);
   }
   while (stilltoread != 0UL) {
      size_t charsread_thispass = fread(curptr, 1UL, stilltoread, pvstream->fp);
      stilltoread -= charsread_thispass;
      pvstream->filepos += charsread_thispass;
      if (stilltoread == 0UL) {
         if (freadcounts > 0) {
            WarnLog().printf(
                  "fread succeeded for \"%s\" on attempt %d.\n", pvstream->name, freadcounts + 1);
         }
         break;
      }
      else {
         if (feof(pvstream->fp)) {
            WarnLog().printf(
                  "fread failure for \"%s\": end of file reached with %lu characters still "
                  "unread.\n",
                  pvstream->name,
                  stilltoread);
            break;
         }
      }
      curptr += charsread_thispass;
      freadcounts++;
      if (freadcounts < MAX_FILESYSTEMCALL_TRIES) {
         WarnLog().printf(
               "fread failure for \"%s\" on attempt %d.  %lu bytes read; %lu bytes still to read "
               "so far.\n",
               pvstream->name,
               freadcounts,
               charsread_thispass,
               stilltoread);
         sleep(1);
      }
      else {
         ErrorLog().printf(
               "PV_fread failure for \"%s\": MAX_FILESYSTEMCALL_TRIES = %d exceeded, and %lu bytes "
               "of %lu read.\n",
               pvstream->name,
               MAX_FILESYSTEMCALL_TRIES,
               readsize - stilltoread,
               readsize);
         break;
      }
   }
   return (readsize - stilltoread) / size;
}

int PV_fclose(PV_Stream *pvstream) {
   int status = PV_SUCCESS;
   if (pvstream) {
      if (pvstream->fp && pvstream->isfile) {
         status       = fclose(pvstream->fp);
         pvstream->fp = NULL;
         if (status != 0) {
            ErrorLog().printf("fclose failure for \"%s\": %s", pvstream->name, strerror(errno));
         }
      }
      free(pvstream->name);
      free(pvstream->mode);
      free(pvstream);
      pvstream = NULL;
   }
   return status;
}

int checkDirExists(MPIBlock const *mpiBlock, const char *dirname, struct stat *pathstat) {
   // check if the given directory name exists for the rank zero process
   // the return value is zero if a successful stat(2) call and the error
   // if unsuccessful.  pathstat contains the result of the buffer from the stat call.
   // The rank zero process is the only one that calls stat();
   // nonzero rank processes return PV_SUCCESS immediately.
   pvAssert(pathstat);

   int rank = mpiBlock->getRank();
   if (rank != 0) {
      return 0;
   }
   int status;
   int errorcode;
   char *expandedDirName = strdup(expandLeadingTilde(dirname).c_str());
   status                = stat(dirname, pathstat);
   free(expandedDirName);
   return status ? errno : 0;
}

static inline int makeDirectory(char const *dir) {
   mode_t dirmode = S_IRWXU | S_IRWXG | S_IRWXO;
   int status     = 0;

   char *workingDir = strdup(dir);
   FatalIf(workingDir == nullptr, "makeDirectory: unable to duplicate path \"%s\".", dir);

   int len = strlen(workingDir);
   if (workingDir[len - 1] == '/')
      workingDir[len - 1] = '\0';

   for (char *p = workingDir + 1; *p; p++)
      if (*p == '/') {
         *p = '\0';
         status |= mkdir(workingDir, dirmode);
         if (status != 0 && errno != EEXIST) {
            return status;
         }
         *p = '/';
      }
   status |= mkdir(workingDir, dirmode);
   if (errno == EEXIST) {
      status = 0;
   }
   return status;
}

int ensureDirExists(MPIBlock const *mpiBlock, char const *dirname) {
   // If rank zero, see if path exists, and try to create it if it doesn't.
   // If not rank zero, the routine does nothing.
   int rank = mpiBlock->getRank();
   struct stat pathstat;
   int resultcode = checkDirExists(mpiBlock, dirname, &pathstat);

   if (resultcode == 0) { // mOutputPath exists; now check if it's a directory.
      FatalIf(
            rank == 0 && !(pathstat.st_mode & S_IFDIR),
            "Path \"%s\" exists but is not a directory\n",
            dirname);
   }
   else if (resultcode == ENOENT /* No such file or directory */) {
      if (rank == 0) {
         InfoLog().printf("Directory \"%s\" does not exist; attempting to create\n", dirname);

         // Try up to 5 times until it works
         int const numAttempts = 5;
         for (int attemptNum = 0; attemptNum < numAttempts; attemptNum++) {
            int mkdirstatus = makeDirectory(dirname);
            if (mkdirstatus != 0) {
               if (attemptNum == numAttempts - 1) {
                  Fatal().printf(
                        "Directory \"%s\" could not be created: %s; Exiting\n",
                        dirname,
                        strerror(errno));
               }
               else {
                  getOutputStream().flush();
                  WarnLog().printf(
                        "Directory \"%s\" could not be created: %s; Retrying %d out of %d\n",
                        dirname,
                        strerror(errno),
                        attemptNum + 1,
                        numAttempts);
                  sleep(1);
               }
            }
            else {
               break;
            }
         }
      }
   }
   else {
      if (rank == 0) {
         ErrorLog().printf(
               "Error checking status of directory \"%s\": %s\n", dirname, strerror(resultcode));
      }
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

/**
 * Gets the number of patches for the given PVLayerLoc, in a non-shared weight context.
 * The return value is the number of patches for the global column (i.e. not a particular MPI
 * process)
 * If asPostWeights is true, loc is interpreted as a postsynaptic layer's PVLayerLoc and patches are
 * not counted in the extended region.  If asPostWeights is false, loc is interpreted as a
 * presynaptic
 * layer's PVLayerLoc and patches are counted in the extended region.
 */
int getNumGlobalPatches(PVLayerLoc const *loc, bool asPostWeights) {
   int nx = loc->nxGlobal;
   int ny = loc->nyGlobal;
   int nf = loc->nf;
   if (!asPostWeights) {
      PVHalo const *halo = &loc->halo;
      nx += halo->lt + halo->rt;
      ny += halo->dn + halo->up;
   }
   return nx * ny * nf;
}

/**
 * Copy patches into an unsigned char buffer
 */
int pvp_copy_patches(
      unsigned char *buf,
      PVPatch **patches,
      float const *dataStart,
      int numDataPatches,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      bool compressed = true) {
   // Copies data from patches and dataStart to buf.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype)
   // characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and
   // PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point
   // to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken
   // patches.
   // The values in patches[k] will be written to &buf[k].  (For PVP_KERNEL_FILE_TYPE, the values
   // are always nx=nxp, ny=nyp, offset=0).
   // The numweights values from dataStart+k*numweights will be copied to buf starting at
   // &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   unsigned char *cptr = buf;
   const int patchsize = nxp * nyp * nfp;
   int nx              = nxp;
   int ny              = nyp;
   int offset          = 0;
   for (int k = 0; k < numDataPatches; k++) {
      if (patches != NULL) {
         PVPatch *p = patches[k];
         nx         = p->nx;
         ny         = p->ny;
         offset     = p->offset;
      }
      const float *data    = dataStart + k * patchsize;
      unsigned short *nxny = (unsigned short *)cptr;
      nxny[0]              = (unsigned short)nxp;
      nxny[1]              = (unsigned short)nyp;
      cptr += 2 * sizeof(unsigned short);

      unsigned int *offsetptr = (unsigned int *)cptr;
      *offsetptr              = 0;
      cptr += sizeof(unsigned int);

      if (compressed) {
         for (int k = 0; k < patchsize; k++) {
            *cptr++ = compressWeight(data[k], minVal, maxVal);
         }
      }
      else {
         float *fptr = (float *)cptr;
         for (int k = 0; k < patchsize; k++) {
            *fptr++ = data[k];
         }
         cptr = (unsigned char *)fptr;
      }
   }

   return PV_SUCCESS;
}

/**
 * Set patches given an unsigned char input buffer
 */
int pvp_set_patches(
      const unsigned char *buf,
      const PVPatch *const *patches,
      float *dataStart,
      int numDataPatches,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      bool compress = true) {
   // Copies weight values from buf to dataStart.
   // buf should point to a buffer of size numDataPatches*pv_sizeof_patch(numweights,datatype)
   // characters,
   // where numweights is nxp*nyp*nfp; and datatype is PV_FLOAT_TYPE for uncompressed weights and
   // PV_BYTE_TYPE for compressed.
   // The calling routine is responsible for allocating and for freeing buf.
   // For PVP_KERNEL_FILE_TYPE, patches should be null.  For PVP_WGT_FILE_TYPE, patches should point
   // to the weight patches for one arbor.
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken
   // patches.
   // The numweights values from dataStart+k*numweights will be copied from buf starting at
   // &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   const unsigned char *cptr = buf;

   const int patchsize = nxp * nyp * nfp;

   unsigned short nx   = nxp;
   unsigned short ny   = nyp;
   unsigned int offset = 0;
   for (int n = 0; n < numDataPatches; n++) {
      if (patches != NULL) {
         const PVPatch *p = patches[n];
      }
      float *data =
            dataStart + n * patchsize; // Don't include offset as entire patch will be read from buf

      cptr += 2 * sizeof(unsigned short) + sizeof(unsigned int);

      if (compress) {
         for (int k = 0; k < patchsize; k++) {
            // values in buf are packed into chars
            data[k] += uncompressWeight(*cptr++, minVal, maxVal);
         }
      }
      else {
         const float *fptr = (const float *)cptr;
         for (int k = 0; k < patchsize; k++) {
            data[k] += *fptr++;
         }
         cptr = (unsigned char *)fptr;
      }
   }

   return PV_SUCCESS;
}

PV_Stream *pvp_open_read_file(const char *filename, MPIBlock const *mpiBlock) {
   PV_Stream *pvstream = NULL;
   if (mpiBlock->getRank() == 0) {
      pvstream = PV_fopen(filename, "rb", false /*verifyWrites*/);
      if (pvstream == NULL) {
         ErrorLog().printf("pvp_open_read_file failed for \"%s\": %s\n", filename, strerror(errno));
      }
   }
   return pvstream;
}

PV_Stream *pvp_open_write_file(const char *filename, MPIBlock const *mpiBlock, bool append) {
   PV_Stream *pvstream = NULL;
   if (mpiBlock->getRank() == 0) {
      bool rwmode = false;
      if (append) {
         // If the file exists, need to use read/write mode (r+) since we'll navigate back to the
         // header to update nbands
         // If the file does not exist, mode r+ gives an error
         struct stat filestat;
         char *realPath = strdup(expandLeadingTilde(filename).c_str());
         int status     = stat(realPath, &filestat);
         free(realPath);
         if (status == 0) {
            rwmode = true;
         }
         else {
            if (errno == ENOENT) {
               WarnLog().printf(
                     "activity file \"%s\" does not exist.  File will be created\n", filename);
               rwmode = false;
            }
            else {
               Fatal().printf("Error opening activity file \"%s\": %s", filename, strerror(errno));
            }
         }
      }
      if (rwmode) {
         pvstream = PV_fopen(filename, "r+b", false /*verifyWrites*/);
         if (pvstream == NULL) {
            ErrorLog().printf(
                  "pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
      else {
         pvstream = PV_fopen(filename, "wb", false /*verifyWrites*/);
         if (pvstream == NULL) {
            ErrorLog().printf(
                  "pvp_open_write_file failed for \"%s\": %s\n", filename, strerror(errno));
         }
      }
   }
   return pvstream;
}

int pvp_close_file(PV_Stream *pvstream, MPIBlock const *mpiBlock) {
   int status = PV_SUCCESS;
   if (mpiBlock->getRank() == 0) {
      status = PV_fclose(pvstream);
   }
   return status;
}

int pvp_check_file_header(
      MPIBlock const *mpiBlock,
      const PVLayerLoc *loc,
      int params[],
      int numParams) {
   int status = PV_SUCCESS;

   int nxProcs = mpiBlock->getNumColumns();
   int nyProcs = mpiBlock->getNumRows();
   int rank    = mpiBlock->getRank();

   if (params[INDEX_NX_PROCS] != 1) {
      status = PV_FAILURE;
      if (rank == 0) {
         ErrorLog().printf(
               "params[%d] = %d, should be 1\n", INDEX_NX_PROCS, params[INDEX_NX_PROCS]);
      }
   }
   if (params[INDEX_NY_PROCS] != 1) {
      status = PV_FAILURE;
      if (rank == 0) {
         ErrorLog().printf(
               "params[%d] = %d, should be 1\n", INDEX_NY_PROCS, params[INDEX_NY_PROCS]);
      }
   }

   if (numParams < NUM_WGT_PARAMS) {
      status = PV_FAILURE;
      if (rank == 0) {
         ErrorLog().printf(
               "pvp_check_file_header called with %d params (requires at least %zu)\n",
               numParams,
               NUM_WGT_PARAMS);
      }
   }

   if (numParams >= NUM_WGT_PARAMS) {
      int patchesInFile       = params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES];
      int numGlobalRestricted = loc->nxGlobal * loc->nyGlobal * loc->nf;
      PVHalo const *halo      = &loc->halo;
      int numGlobalExtended =
            (loc->nxGlobal + halo->lt + halo->rt) * (loc->nyGlobal + halo->dn + halo->up) * loc->nf;
      switch (params[INDEX_FILE_TYPE]) {
         case PVP_WGT_FILE_TYPE:
            if (patchesInFile != numGlobalRestricted && patchesInFile != numGlobalExtended) {
               status = PV_FAILURE;
               if (rank == 0) {
                  ErrorLog(badNumParams);
                  badNumParams.printf(
                        "params[%zu] = %d, should be ",
                        NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES,
                        params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES]);
                  if (numGlobalExtended == numGlobalRestricted) {
                     badNumParams.printf("%d\n", numGlobalExtended);
                  }
                  else {
                     badNumParams.printf(
                           "either %d (as post weights) or %d (as pre weights)\n",
                           numGlobalRestricted,
                           numGlobalExtended);
                  }
               }
            }
            break;
         case PVP_KERNEL_FILE_TYPE:
            if (patchesInFile
                % loc->nf) { // Not enough information passed to function to get unit cell size
               status = PV_FAILURE;
               if (rank == 0) {
                  ErrorLog().printf(
                        "params[%zu] = %d, should be a multiple of loc->nf=%d\n",
                        NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES,
                        params[NUM_BIN_PARAMS + INDEX_WGT_NUMPATCHES],
                        loc->nf);
               }
            }
            break;
         default: assert(0); break;
      }
   }

   if (status != 0) {
      for (int i = 0; i < numParams; i++) {
         ErrorLog().printf("params[%2d]==%d\n", i, params[i]);
      }
   }

   return status;
} // pvp_check_file_header

int pvp_read_header(PV_Stream *pvstream, MPIBlock const *mpiBlock, int *params, int *numParams) {
   // Under MPI, called by all processes; nonroot processes should have pvstream==NULL
   // On entry, numParams is the size of the params buffer.
   // All process should have the same numParams on entry.
   // On exit, numParams is the number of params actually read, they're read into params[0] through
   // params[(*numParams)-1]
   // All processes receive the same params, the same numParams, and the same return value
   // (PV_SUCCESS or PV_FAILURE).
   // If the return value is PV_FAILURE, *numParams has information on the type of failure.
   int status        = PV_SUCCESS;
   int numParamsRead = 0;
   int *mpi_buffer   = (int *)calloc((size_t)(*numParams + 2), sizeof(int));
   if (mpiBlock->getRank() == 0) {
      if (pvstream == NULL) {
         ErrorLog().printf("pvp_read_header: pvstream==NULL for rank zero");
         status = PV_FAILURE;
      }
      if (*numParams < 2) {
         numParamsRead = 0;
         status        = PV_FAILURE;
      }

      // find out how many parameters there are
      //
      if (status == PV_SUCCESS) {
         int numread = PV_fread(params, sizeof(int), 2, pvstream);
         if (numread != 2) {
            numParamsRead = -1;
            status        = PV_FAILURE;
         }
      }
      int nParams = 0;
      if (status == PV_SUCCESS) {
         nParams = params[INDEX_NUM_PARAMS];
         if (params[INDEX_HEADER_SIZE] != nParams * (int)sizeof(int)) {
            numParamsRead = -2;
            status        = PV_FAILURE;
         }
      }
      if (status == PV_SUCCESS) {
         if (nParams > *numParams) {
            numParamsRead = nParams;
            status        = PV_FAILURE;
         }
      }

      // read the rest
      //
      if (status == PV_SUCCESS && *numParams > 2) {
         size_t numRead = PV_fread(&params[2], sizeof(int), nParams - 2, pvstream);
         if (numRead != (size_t)nParams - 2) {
            status     = PV_FAILURE;
            *numParams = numRead;
         }
      }
      if (status == PV_SUCCESS) {
         numParamsRead = params[INDEX_NUM_PARAMS];
      }
      mpi_buffer[0] = status;
      mpi_buffer[1] = numParamsRead;
      memcpy(&mpi_buffer[2], params, sizeof(int) * (*numParams));
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0 /*root*/, mpiBlock->getComm());
   } // mpiBlock->getRank()==0
   else {
      MPI_Bcast(mpi_buffer, 22, MPI_INT, 0 /*root*/, mpiBlock->getComm());
      status = mpi_buffer[0];
      memcpy(params, &mpi_buffer[2], sizeof(int) * (*numParams));
   }
   *numParams = mpi_buffer[1];
   free(mpi_buffer);
   return status;
}

int pvp_read_header(
      PV_Stream *pvstream,
      double *time,
      int *filetype,
      int *datatype,
      int params[],
      int *numParams) {
   int status = PV_SUCCESS;

   if (*numParams < 2) {
      *numParams = 0;
      return -1;
   }

   // find out how many parameters there are
   //
   if (PV_fread(params, sizeof(int), 2, pvstream) != 2)
      return -1;

   int nParams = params[INDEX_NUM_PARAMS];
   assert(params[INDEX_HEADER_SIZE] == (int)(nParams * sizeof(int)));
   if (nParams > *numParams) {
      *numParams = 2;
      return -1;
   }

   // read the rest
   //
   if (PV_fread(&params[2], sizeof(int), nParams - 2, pvstream) != (unsigned int)nParams - 2)
      return -1;

   *numParams = params[INDEX_NUM_PARAMS];
   *filetype  = params[INDEX_FILE_TYPE];
   *datatype  = params[INDEX_DATA_TYPE];

   // make sure the parameters are what we are expecting
   //

   assert(params[INDEX_DATA_SIZE] == (int)pv_sizeof(*datatype));

   *time = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_read_header(
      const char *filename,
      MPIBlock const *mpiBlock,
      double *time,
      int *filetype,
      int *datatype,
      int params[],
      int *numParams) {
   int status       = PV_SUCCESS;
   const int icRank = mpiBlock->getRank();

   if (icRank == 0) {
      PV_Stream *pvstream = pvp_open_read_file(filename, mpiBlock);
      if (pvstream == NULL) {
         Fatal().printf(
               "[%2d]: pvp_read_header: pvp_open_read_file failed to open file \"%s\"\n",
               mpiBlock->getRank(),
               filename);
      }

      status = pvp_read_header(pvstream, time, filetype, datatype, params, numParams);
      pvp_close_file(pvstream, mpiBlock);
      if (status != 0)
         return status;
   }

   const int icRoot = 0;
#ifdef DEBUG_OUTPUT
   DebugLog().printf(
         "[%2d]: pvp_read_header: will broadcast, numParams==%d\n",
         mpiBlock->getRank(),
         *numParams);
#endif // DEBUG_OUTPUT

   status = MPI_Bcast(params, *numParams, MPI_INT, icRoot, mpiBlock->getComm());

#ifdef DEBUG_OUTPUT
   DebugLog().printf(
         "[%2d]: pvp_read_header: broadcast completed, numParams==%d\n",
         mpiBlock->getRank(),
         *numParams);
#endif // DEBUG_OUTPUT

   *filetype = params[INDEX_FILE_TYPE];
   *datatype = params[INDEX_DATA_TYPE];
   *time     = timeFromParams(&params[INDEX_TIME]);

   return status;
}

int pvp_write_header(
      PV_Stream *pvstream,
      MPIBlock const *mpiBlock,
      double time,
      const PVLayerLoc *loc,
      int filetype,
      int datatype,
      int numbands,
      bool extended,
      bool contiguous,
      unsigned int numParams,
      size_t recordSize) {
   int status = PV_SUCCESS;
   int nxBlocks, nyBlocks;
   int params[NUM_BIN_PARAMS];

   if (mpiBlock->getRank() != 0)
      return status;

   const int headerSize = numParams * sizeof(int);

   const int nxProcs = mpiBlock->getNumColumns();
   const int nyProcs = mpiBlock->getNumRows();

   if (contiguous) {
      nxBlocks = 1;
      nyBlocks = 1;
   }
   else {
      nxBlocks = nxProcs;
      nyBlocks = nyProcs;
   }

   // make sure we don't blow out size of int for record size

   int numRecords;
   int paramNBands;

   switch (filetype) {
      case PVP_WGT_FILE_TYPE:
         numRecords = numbands * nxBlocks * nyBlocks; // Each process writes a record for each arbor
         paramNBands = numbands;
         break;
      case PVP_KERNEL_FILE_TYPE:
         numRecords =
               numbands; // Each arbor writes its own record; all processes have the same weights
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
   params[INDEX_NBATCH]      = loc->nbatch;
   params[INDEX_NBANDS]      = paramNBands;

   timeToParams(time, &params[INDEX_TIME]);

   numParams = NUM_BIN_PARAMS; // there may be more to come
   if (PV_fwrite(params, sizeof(int), numParams, pvstream) != numParams) {
      status = -1;
   }

   return status;
}

// Unused function pvp_set_activity_params was removed Jan 26, 2017.
// Unused function pvp_set_weight_params was removed Jan 26, 2017.
// Unused function pvp_set_nonspiking_act_params was removed Feb 21, 2017.
// Unused function pvp_set_nonspiking_sparse_act_params was removed Feb 21, 2017.
// Unused function alloc_params was removed Feb 21, 2017.

// writeActivity and writeActivitySparse removed Feb 17, 2017.
// Corresponding HyPerLayer methods now use BufferUtils routines
// gatherActivity and scatterActivity were also removed.
// Use BufferUtils::gather and BufferUtils::scatter instead.

int readWeights(
      PVPatch ***patches,
      float **dataStart,
      int numArbors,
      int numPatches,
      int nxp,
      int nyp,
      int nfp,
      const char *filename,
      MPIBlock const *mpiBlock,
      double *timed,
      const PVLayerLoc *loc) {
   int header_data_type;
   int header_file_type;

   int numParams = NUM_WGT_PARAMS;
   int params[NUM_WGT_PARAMS];
   int *wgtParams = &params[NUM_BIN_PARAMS];
   int status     = pvp_read_header(
         filename, mpiBlock, timed, &header_file_type, &header_data_type, params, &numParams);

   // rank zero process broadcasts params to all processes, so it's enough for rank zero process to
   // do the error checking
   if (mpiBlock->getRank() == 0) {
      if (numParams != NUM_WGT_PARAMS) {
         Fatal().printf(
               "Reading weights file \"%s\": expected %zu parameters in header but received %d\n",
               filename,
               NUM_WGT_PARAMS,
               numParams);
      }
      if (params[NUM_BIN_PARAMS + INDEX_WGT_NXP] != nxp
          || params[NUM_BIN_PARAMS + INDEX_WGT_NYP] != nyp) {
         Fatal().printf(
               "readWeights error: called with nxp=%d, nyp=%d, but \"%s\" has nxp=%d, nyp=%d\n",
               nxp,
               nyp,
               filename,
               params[NUM_BIN_PARAMS + INDEX_WGT_NXP],
               params[NUM_BIN_PARAMS + INDEX_WGT_NYP]);
      }
   }

   const int nxFileBlocks = params[INDEX_NX_PROCS];
   const int nyFileBlocks = params[INDEX_NY_PROCS];

   status = pvp_check_file_header(mpiBlock, loc, params, numParams);

   if (status != 0) {
      ErrorLog().printf(
            "[%2d]: readWeights: failed in pvp_check_file_header, numParams==%d\n",
            mpiBlock->getRank(),
            numParams);
      return status;
   }
   assert(params[INDEX_NX_PROCS] == 1 && params[INDEX_NY_PROCS] == 1);
   if (params[INDEX_NBANDS] > numArbors) {
      ErrorLog().printf(
            "PV::readWeights: file \"%s\" has %d arbors, but readWeights was called with only %d "
            "arbors",
            filename,
            params[INDEX_NBANDS],
            numArbors);
      return -1;
   }

   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize  = pv_sizeof_patch(numPatchItems, header_data_type);
   const size_t localSize  = numPatches * patchSize;

   // Have to use memcpy instead of casting floats because of strict aliasing rules, since some are
   // int and some are float
   float minVal = 0.0f;
   memcpy(&minVal, &wgtParams[INDEX_WGT_MIN], sizeof(float));
   float maxVal = 0.0f;
   memcpy(&maxVal, &wgtParams[INDEX_WGT_MAX], sizeof(float));

   const int icRank = mpiBlock->getRank();

   bool compress       = header_data_type == BufferUtils::BYTE;
   unsigned char *cbuf = (unsigned char *)malloc(localSize);
   if (cbuf == NULL) {
      Fatal(errorMessage);
      errorMessage.printf(
            "Rank %d: readWeights unable to allocate memory to write to \"%s\": %s",
            icRank,
            filename,
            strerror(errno));
      errorMessage.printf(
            "    (nxp=%d, nyp=%d, nfp=%d, numPatchItems=%d, writing weights as %s)\n",
            nxp,
            nyp,
            nfp,
            numPatchItems,
            compress ? "bytes" : "floats");
   }

   const int expected_file_type = patches == NULL ? PVP_KERNEL_FILE_TYPE : PVP_WGT_FILE_TYPE;
   const int tagbase            = expected_file_type;
#ifdef PV_USE_MPI
   const MPI_Comm mpi_comm = mpiBlock->getComm();
#else
   const MPI_Comm mpi_comm = NULL;
#endif // PV_USE_MPI
   const int src = 0;
   if (expected_file_type != header_file_type) {
      if (icRank == 0) {
         ErrorLog().printf(
               "readWeights: file \"%s\" has type %d but readWeights was called expecting type "
               "%d\n",
               filename,
               header_file_type,
               expected_file_type);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(mpi_comm);
#endif // PV_USE_MPI
      exit(EXIT_FAILURE);
   }
   if (icRank > 0) {
#ifdef PV_USE_MPI
      for (int arbor = 0; arbor < params[INDEX_NBANDS]; arbor++) {
         if (header_file_type == PVP_KERNEL_FILE_TYPE) {
#ifdef DEBUG_OUTPUT
            DebugLog().printf(
                  "[%2d]: readWeights: bcast from %d, arbor %d, numPatchItems %d, numPatches==%d, "
                  "localSize==%zu\n",
                  mpiBlock->getRank(),
                  src,
                  arbor,
                  numPatchItems,
                  numPatches,
                  localSize);
#endif // DEBUG_OUTPUT
            MPI_Bcast(cbuf, localSize, MPI_BYTE, src, mpi_comm);
         }
         else {
            assert(header_file_type == PVP_WGT_FILE_TYPE);
#ifdef DEBUG_OUTPUT
            DebugLog().printf(
                  "[%2d]: readWeights: recv from %d, arbor %d, numPatchItems %d, numPatches==%d, "
                  "localSize==%zu\n",
                  mpiBlock->getRank(),
                  src,
                  arbor,
                  numPatchItems,
                  numPatches,
                  localSize);
#endif // DEBUG_OUTPUT
            MPI_Recv(cbuf, localSize, MPI_BYTE, src, tagbase + arbor, mpi_comm, MPI_STATUS_IGNORE);
         }
         pvp_set_patches(
               cbuf,
               patches ? patches[arbor] : NULL,
               dataStart[arbor],
               numPatches,
               nxp,
               nyp,
               nfp,
               minVal,
               maxVal,
               compress);
      }
#endif // PV_USE_MPI
   } // icRank > 0
   else /*icRank == 0*/ {
      PV_Stream *pvstream  = pvp_open_read_file(filename, mpiBlock);
      const int headerSize = numParams * sizeof(int);
      for (int arbor = 0; arbor < params[INDEX_NBANDS]; arbor++) {
         long int arborStart = headerSize + localSize * arbor;
         if (header_file_type == PVP_KERNEL_FILE_TYPE) {
            PV_fseek(pvstream, arborStart, SEEK_SET);
            int numRead = PV_fread(cbuf, localSize, 1, pvstream);
            if (numRead != 1) {
               Fatal().printf(
                     "readWeights error reading arbor %d of %zu bytes from position %ld of "
                     "\"%s\".\n",
                     arbor,
                     localSize,
                     arborStart,
                     filename);
            };
#ifdef DEBUG_OUTPUT
            DebugLog().printf(
                  "[%2d]: readWeights: bcast from %d, arbor %d, numPatchItems %d, numPatches==%d, "
                  "localSize==%zu\n",
                  mpiBlock->getRank(),
                  src,
                  arbor,
                  numPatchItems,
                  numPatches,
                  localSize);
#endif // DEBUG_OUTPUT
            if (mpiBlock->getSize() > 1) {
               MPI_Bcast(cbuf, localSize, MPI_BYTE, src, mpi_comm);
            }
         }
         else {
            assert(header_file_type == PVP_WGT_FILE_TYPE);
            int globalSize = patchSize * wgtParams[INDEX_WGT_NUMPATCHES];
            for (int proc = 0; proc <= mpiBlock->getSize(); proc++) {
               if (proc == src) {
                  continue;
               } // Do local section last
               int procrow, proccolumn;
               if (proc == mpiBlock->getSize()) {
                  procrow = rowFromRank(src, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
                  proccolumn =
                        columnFromRank(src, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
               }
               else {
                  procrow = rowFromRank(proc, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
                  proccolumn =
                        columnFromRank(proc, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
               }
               for (int k = 0; k < numPatches; k++) {
                  // TODO: don't have to read each patch singly; can read all patches for a single y
                  // at once.
                  unsigned char *cbufpatch = &cbuf[k * patchSize];
                  int x                    = kxPos(k,
                                loc->nx + loc->halo.lt + loc->halo.rt,
                                loc->ny + loc->halo.dn + loc->halo.up,
                                loc->nf)
                          + proccolumn * loc->nx;
                  int y = kyPos(k,
                                loc->nx + loc->halo.lt + loc->halo.rt,
                                loc->ny + loc->halo.dn + loc->halo.up,
                                loc->nf)
                          + procrow * loc->ny;
                  int f = featureIndex(
                        k,
                        loc->nx + loc->halo.lt + loc->halo.rt,
                        loc->ny + loc->halo.dn + loc->halo.up,
                        loc->nf);
                  int globalIndex = kIndex(x, y, f, loc->nxGlobal, loc->nyGlobal, loc->nf);
                  PV_fseek(pvstream, arborStart + globalIndex * patchSize, SEEK_SET);
                  int numread = PV_fread(cbufpatch, patchSize, 1, pvstream);
                  if (numread != 1) {
                     Fatal().printf(
                           "readWeights error reading arbor %d, patch %d of %zu bytes from "
                           "position %ld of \"%s\".\n",
                           arbor,
                           k,
                           patchSize,
                           arborStart + globalIndex * patchSize,
                           filename);
                  }
               } // Loop over patches
               if (proc != mpiBlock->getSize()) {
                  MPI_Send(cbuf, localSize, MPI_BYTE, proc, tagbase + arbor, mpi_comm);
#ifdef DEBUG_OUTPUT
                  DebugLog().printf(
                        "[%2d]: readWeights: recv from %d, arbor %d, numPatchItems %d, "
                        "numPatches==%d, localSize==%zu\n",
                        mpiBlock->getRank(),
                        src,
                        arbor,
                        numPatchItems,
                        numPatches,
                        localSize);
#endif // DEBUG_OUTPUT
               }
            } // Loop over processes
            // Local section was done last, so cbuf should contain data from local section
         } // if-statement for header_file_type
         pvp_set_patches(
               cbuf,
               patches ? patches[arbor] : NULL,
               dataStart[arbor],
               numPatches,
               nxp,
               nyp,
               nfp,
               minVal,
               maxVal,
               compress);
      } // loop over arbors
      pvp_close_file(pvstream, mpiBlock);
   } // if-statement for icRank

   free(cbuf);
   cbuf = NULL;

   return status;
}

/**
 * @fd
 * @patch
 */
int pv_text_write_patch(
      PrintStream *outStream,
      PVPatch *patch,
      float *data,
      int nf,
      int sx,
      int sy,
      int sf) {
   int f, i, j;

   const int nx = (int)patch->nx;
   const int ny = (int)patch->ny;

   assert(outStream != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            outStream->printf("%7.5f ", (double)data[i * sx + j * sy + f * sf]);
         }
         outStream->printf("\n");
      }
      outStream->printf("\n");
   }

   return 0;
}

void writeSharedWeights(
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      double timed,
      PVLayerLoc const *preLoc,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      float **dataStart,
      int numPatches,
      int numArbors,
      bool compress) {
   if (fileStream == nullptr) {
      return;
   }

   PVHalo const &halo = preLoc->halo;
   int const nx       = preLoc->nx * mpiBlock->getNumColumns() * halo.lt + halo.rt;
   int const ny       = preLoc->nx * mpiBlock->getNumRows() * halo.dn + halo.up;
   int const nf       = preLoc->nf;
   int const nbatch   = preLoc->nbatch * mpiBlock->getBatchDimension();
   BufferUtils::WeightHeader header;
   std::size_t patchSize;
   if (compress) {
      header = BufferUtils::buildWeightHeader<unsigned char>(
            nxp, nyp, nfp, numPatches, numArbors, true, nx, ny, nf, nbatch, minVal, maxVal);
      patchSize = BufferUtils::weightPatchSize<unsigned char>(nxp * nyp * nfp);
   }
   else {
      header = BufferUtils::buildWeightHeader<float>(
            nxp, nyp, nfp, numPatches, numArbors, true, nx, ny, nf, nbatch, minVal, maxVal);
      patchSize = BufferUtils::weightPatchSize<float>(nxp * nyp * nfp);
   }
   fileStream->write(&header, sizeof(header));

   std::size_t const localSize = (std::size_t)numPatches * patchSize;
   std::vector<unsigned char> cbuf(localSize);
   for (int arbor = 0; arbor < numArbors; arbor++) {
      float const *arborStart = dataStart[arbor];
      pvp_copy_patches(
            cbuf.data(), nullptr, arborStart, numPatches, nxp, nyp, nfp, minVal, maxVal, compress);
      fileStream->write(cbuf.data(), cbuf.size());
   }
}

int writeWeights(
      const char *filename,
      MPIBlock const *mpiBlock,
      double timed,
      bool append,
      const PVLayerLoc *preLoc,
      const PVLayerLoc *postLoc,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      PVPatch ***patches,
      float **dataStart,
      int numPatches,
      int numArbors,
      bool compress,
      int file_type) {
   int status = PV_SUCCESS;

   int datatype = compress ? BufferUtils::BYTE : BufferUtils::FLOAT;

   const int icRank = mpiBlock->getRank();

   const int numPatchItems = nxp * nyp * nfp;
   const size_t patchSize  = pv_sizeof_patch(numPatchItems, datatype);
   const size_t localSize  = numPatches * patchSize;

   unsigned char *cbuf = (unsigned char *)malloc(localSize);
   if (cbuf == NULL) {
      Fatal(errorMessage);
      errorMessage.printf(
            "Rank %d: writeWeights unable to allocate memory to write to \"%s\": %s",
            icRank,
            filename,
            strerror(errno));
      errorMessage.printf(
            "    (nxp=%d, nyp=%d, nfp=%d, numPatchItems=%d, writing weights as %s)\n",
            nxp,
            nyp,
            nfp,
            numPatchItems,
            compress ? "bytes" : "floats");
   }

#ifdef PV_USE_MPI
   const int tagbase       = file_type; // PVP_WGT_FILE_TYPE;
   const MPI_Comm mpi_comm = mpiBlock->getComm();
#endif // PV_USE_MPI
   if (icRank > 0) {
#ifdef PV_USE_MPI
      if (file_type != PVP_KERNEL_FILE_TYPE) { // No MPI needed for kernel weights
         // (sharedWeights==true).  If keepKernelsSynchronized
         // is false, synchronize kernels before entering
         // PV::writeWeights()
         const int dest = 0;
         for (int arbor = 0; arbor < numArbors; arbor++) {
            pvp_copy_patches(
                  cbuf,
                  patches[arbor],
                  dataStart[arbor],
                  numPatches,
                  nxp,
                  nyp,
                  nfp,
                  minVal,
                  maxVal,
                  compress);
            MPI_Send(cbuf, localSize, MPI_BYTE, dest, tagbase + arbor, mpi_comm);
#ifdef DEBUG_OUTPUT
            DebugLog().printf(
                  "[%2d]: writeWeights: sent to 0, nxBlocks==%d nyBlocks==%d numPatches==%d\n",
                  mpiBlock->getRank(),
                  nxBlocks,
                  nyBlocks,
                  numPatches);
#endif // DEBUG_OUTPUT
         }
      }
#endif // PV_USE_MPI
   } // icRank > 0
   else /* icRank==0 */ {
      void *wgtExtraParams = calloc(NUM_WGT_EXTRA_PARAMS, sizeof(int));
      if (wgtExtraParams == NULL) {
         Fatal().printf(
               "writeWeights unable to allocate memory to write weight params to \"%s\": %s\n",
               filename,
               strerror(errno));
      }
      int *wgtExtraIntParams     = (int *)wgtExtraParams;
      float *wgtExtraFloatParams = (float *)wgtExtraParams;

      int numParams = NUM_WGT_PARAMS;

      PV_Stream *pvstream = pvp_open_write_file(filename, mpiBlock, append);

      if (pvstream == NULL) {
         ErrorLog().printf("PV::writeWeights: unable to open file \"%s\"\n", filename);
         return -1;
      }
      if (append)
         PV_fseek(pvstream, 0L, SEEK_END); // If append is true we open in "r+" mode so we need to
      // move to the end of the file.

      int numGlobalPatches;
      bool asPostWeights;
      switch (file_type) {
         case PVP_WGT_FILE_TYPE:
            if (numPatches
                == (preLoc->nx + preLoc->halo.lt + preLoc->halo.rt)
                         * (preLoc->ny + preLoc->halo.dn + preLoc->halo.up)
                         * preLoc->nf) {
               asPostWeights = false;
            }
            else if (numPatches == preLoc->nx * preLoc->ny * preLoc->nf) {
               asPostWeights = true;
            }
            else {
               ErrorLog().printf(
                     "writeWeights: in file \"%s\", numPatches %d is not compatible with layer "
                     "dimensions nx=%d, ny=%d, nf=%d, halo=(%d,%d,%d,%d)\n",
                     filename,
                     numPatches,
                     preLoc->nx,
                     preLoc->ny,
                     preLoc->nf,
                     preLoc->halo.lt,
                     preLoc->halo.rt,
                     preLoc->halo.dn,
                     preLoc->halo.up);
            }
            numGlobalPatches = getNumGlobalPatches(preLoc, asPostWeights);
            break;
         case PVP_KERNEL_FILE_TYPE:
            asPostWeights    = false;
            numGlobalPatches = numPatches;
            break;
         default:
            assert(0); // only possibilities for file_type are WGT and KERNEL
            break;
      }
      size_t globalSize = numGlobalPatches * patchSize;
      status            = pvp_write_header(
            pvstream,
            mpiBlock,
            timed,
            preLoc,
            file_type,
            datatype,
            numArbors,
            true /*extended*/,
            true /*contiguous*/,
            numParams,
            globalSize);
      if (status != PV_SUCCESS) {
         Fatal().printf("Error writing writing weights header to \"%s\".\n", filename);
      }

      // write extra weight parameters
      //
      wgtExtraIntParams[INDEX_WGT_NXP] = nxp;
      wgtExtraIntParams[INDEX_WGT_NYP] = nyp;
      wgtExtraIntParams[INDEX_WGT_NFP] = nfp;

      wgtExtraFloatParams[INDEX_WGT_MIN]      = minVal;
      wgtExtraFloatParams[INDEX_WGT_MAX]      = maxVal;
      wgtExtraIntParams[INDEX_WGT_NUMPATCHES] = numGlobalPatches;

      numParams                = NUM_WGT_EXTRA_PARAMS;
      unsigned int num_written = PV_fwrite(wgtExtraParams, sizeof(int), numParams, pvstream);
      free(wgtExtraParams);
      wgtExtraParams      = NULL;
      wgtExtraIntParams   = NULL;
      wgtExtraFloatParams = NULL;
      if (num_written != (unsigned int)numParams) {
         ErrorLog().printf(
               "PV::writeWeights: unable to write weight header to file %s\n", filename);
         return -1;
      }

      for (int arbor = 0; arbor < numArbors; arbor++) {
         PVPatch **arborPatches = file_type == PVP_KERNEL_FILE_TYPE ? NULL : patches[arbor];
         if (file_type == PVP_KERNEL_FILE_TYPE) {
            pvp_copy_patches(
                  cbuf,
                  arborPatches,
                  dataStart[arbor],
                  numPatches,
                  nxp,
                  nyp,
                  nfp,
                  minVal,
                  maxVal,
                  compress);
            size_t numfwritten = PV_fwrite(cbuf, localSize, 1, pvstream);
            if (numfwritten != 1) {
               ErrorLog().printf(
                     "PV::writeWeights: unable to write weight data to file %s\n", filename);
               return -1;
            }
         }
         else {
            assert(file_type == PVP_WGT_FILE_TYPE);
            long int arborstartfile = getPV_StreamFilepos(pvstream);
            long int arborendfile   = arborstartfile + globalSize;
            PV_fseek(pvstream, arborendfile - 1, SEEK_SET);
            char endarborchar = (char)0;
            PV_fwrite(&endarborchar, 1, 1, pvstream); // Makes sure the file is the correct length
            // even if the last patch is shrunken
            for (int proc = 0; proc < mpiBlock->getSize(); proc++) {
#ifdef PV_USE_MPI
               if (proc == 0) /*local portion*/ {
                  pvp_copy_patches(
                        cbuf,
                        arborPatches,
                        dataStart[arbor],
                        numPatches,
                        nxp,
                        nyp,
                        nfp,
                        minVal,
                        maxVal,
                        compress);
               }
               else /*receive other portion via MPI*/ {
                  MPI_Recv(
                        cbuf,
                        localSize,
                        MPI_BYTE,
                        proc,
                        tagbase + arbor,
                        mpi_comm,
                        MPI_STATUS_IGNORE);
               }
#else // PV_USE_MPI
               assert(proc == 0);
               pvp_copy_patches(
                     cbuf,
                     arborPatches,
                     dataStart[arbor],
                     numPatches,
                     nxp,
                     nyp,
                     nfp,
                     minVal,
                     maxVal,
                     compress);
#endif // PV_USE_MPI
               int procrow = rowFromRank(proc, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
               int proccolumn =
                     columnFromRank(proc, mpiBlock->getNumRows(), mpiBlock->getNumColumns());
               for (int k = 0; k < numPatches; k++) {
                  unsigned char *cbufpatch = &cbuf[k * patchSize];
                  int globalIndex;
                  if (asPostWeights) {
                     int x = kxPos(k, preLoc->nx, preLoc->ny, preLoc->nf) + proccolumn * preLoc->nx;
                     int y = kyPos(k, preLoc->nx, preLoc->ny, preLoc->nf) + procrow * preLoc->ny;
                     int f = featureIndex(k, preLoc->nx, preLoc->ny, preLoc->nf);
                     globalIndex = kIndex(x, y, f, preLoc->nxGlobal, preLoc->nyGlobal, preLoc->nf);
                  }
                  else {
                     int x = kxPos(k,
                                   preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                                   preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                                   preLoc->nf)
                             + proccolumn * preLoc->nx;
                     int y = kyPos(k,
                                   preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                                   preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                                   preLoc->nf)
                             + procrow * preLoc->ny;
                     int f = featureIndex(
                           k,
                           preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                           preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                           preLoc->nf);
                     globalIndex = kIndex(
                           x,
                           y,
                           f,
                           preLoc->nxGlobal + preLoc->halo.lt + preLoc->halo.rt,
                           preLoc->nyGlobal + preLoc->halo.dn + preLoc->halo.up,
                           preLoc->nf);
                  }
                  unsigned short int pnx = *(unsigned short int *)(cbuf);
                  unsigned short int pny =
                        *(unsigned short int *)(cbuf + sizeof(unsigned short int));
                  unsigned int p_offset = *(unsigned int *)(cbuf + 2 * sizeof(unsigned short int));
                  if (pnx == nxp && pny == nyp) /*not shrunken*/ {
                     PV_fseek(
                           pvstream,
                           arborstartfile + globalIndex * patchSize,
                           SEEK_SET); // TODO: error handling
                     PV_fwrite(cbufpatch, patchSize, (size_t)1, pvstream); // TODO: error handling
                  }
                  else {
                     PV_fseek(
                           pvstream,
                           arborstartfile + k * patchSize,
                           SEEK_SET); // TODO: error handling
                     size_t const patchheadersize = 2 * sizeof(short int) + sizeof(int);
                     PV_fwrite(cbufpatch, patchheadersize, 1, pvstream); // TODO: error handling
                     int datasize  = pv_sizeof(datatype);
                     const int syw = nfp * nxp;
                     for (int y = 0; y < pny; y++) {
                        unsigned int memoffset = patchheadersize + (p_offset + y * syw) * datasize;
                        PV_fseek(
                              pvstream,
                              arborstartfile + globalIndex * patchSize + memoffset,
                              SEEK_SET); // TODO: error handling
                        PV_fwrite(
                              cbufpatch + memoffset,
                              pnx * nfp * datasize,
                              1,
                              pvstream); // TODO: error handling
                     } // Loop over line within patch
                  } // if (p->nx == nxp && p->ny == nyp)
               } // Loop over patches
            } // Loop over processes
            PV_fseek(pvstream, arborendfile, SEEK_SET);
         } // if-statement for file_type
      } // loop over arbors
      pvp_close_file(pvstream, mpiBlock);
   } // if-statement for process rank

   free(cbuf);
   cbuf = NULL;

   return status;
}

} // namespace PV
