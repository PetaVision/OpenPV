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
#include "utils/PVLog.hpp"
#include "utils/conversions.h"

#include <assert.h>
#include <iostream>

#undef DEBUG_OUTPUT

namespace PV {

// Unused function timeToParams was removed Mar 10, 2017.
// Unused function timeFromParams was removed Mar 15, 2017.
// Unused function pv_sizeof was removed Mar 15, 2017.
// Unused function pv_sizeof_patch was removed Mar 15, 2017.

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

// Unused function getNumGlobalPatches was removed Mar 15, 2017.
// Instead, use calcNumberOfPatches in utils/BufferUtilsPvp.*

/**
 * Copy patches into an unsigned char buffer
 */
int pvp_copy_patches(
      unsigned char *buf,
      PVPatch const *const *patchGeometry,
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
   // Each patch takes up pv_sizeof_patch(numweights,datatype) chars in buf --- even for shrunken
   // patches.
   // The values in patches[k] will be written to &buf[k].  (For PVP_KERNEL_FILE_TYPE, the values
   // are always nx=nxp, ny=nyp, offset=0).
   // The numweights values from dataStart+k*numweights will be copied to buf starting at
   // &buf[k*(numweights*datasize+2*sizeof(short)+sizeof(int))].
   unsigned char *cptr = buf;
   const int patchsize = nxp * nyp * nfp;
   for (int k = 0; k < numDataPatches; k++) {
      const float *data    = dataStart + k * patchsize;
      unsigned short *nxny = (unsigned short *)cptr;
      if (patchGeometry) {
         nxny[0] = patchGeometry[k]->nx;
         nxny[1] = patchGeometry[k]->ny;
      }
      else {
         nxny[0] = (unsigned short)nxp;
         nxny[1] = (unsigned short)nyp;
      }
      cptr += 2 * sizeof(unsigned short);
      unsigned int *offsetptr = (unsigned int *)cptr;
      if (patchGeometry) {
         *offsetptr = patchGeometry[k]->offset;
      }
      else {
         *offsetptr = 0;
      }
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

int pvp_copy_arbor(
      std::vector<unsigned char> &dataInPvpFormat,
      PVPatch const *const *patchGeometry,
      float const *dataStart,
      PatchListDescription const &patchListDesc,
      int nxp,
      int nyp,
      int nfp,
      std::size_t patchSize,
      float minVal,
      float maxVal,
      bool compressed = true) {
   int const numPatchesInLine = patchListDesc.mNumPatchesX * patchListDesc.mNumPatchesF;
   for (int y = 0; y < patchListDesc.mNumPatchesY; y++) {
      int startingPatch   = patchListDesc.mStartIndex + y * patchListDesc.mStrideY;
      float const *source = &dataStart[startingPatch * nxp * nyp * nfp];
      unsigned char *dest = &dataInPvpFormat.at(y * numPatchesInLine * patchSize);
      pvp_copy_patches(
            dest,
            &patchGeometry[startingPatch],
            source,
            numPatchesInLine,
            nxp,
            nyp,
            nfp,
            minVal,
            maxVal,
            compressed);
   }
   return PV_SUCCESS;
}

/**
 * Set patches given an unsigned char input buffer
 */
int pvp_set_patches(
      const unsigned char *buf,
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

int pvp_set_arbor(
      std::vector<unsigned char> const &dataInPvpFormat,
      float *dataStart,
      PatchListDescription const &patchListDesc,
      int nxp,
      int nyp,
      int nfp,
      std::size_t patchSize,
      float minVal,
      float maxVal,
      bool compressed = true) {
   int const numPatchesInLine = patchListDesc.mNumPatchesX * patchListDesc.mNumPatchesF;
   for (int y = 0; y < patchListDesc.mNumPatchesY; y++) {
      int startingPatch           = patchListDesc.mStartIndex + y * patchListDesc.mStrideY;
      unsigned char const *source = &dataInPvpFormat.at(y * numPatchesInLine * patchSize);
      float *dest                 = &dataStart[startingPatch * nxp * nyp * nfp];
      pvp_set_patches(source, dest, numPatchesInLine, nxp, nyp, nfp, minVal, maxVal, compressed);
   }
   return PV_SUCCESS;
}

// Unused function pvp_open_read_file was removed Mar 23, 2017. Instead, construct a FileStream.
// Unused function pvp_open_write_file was removed Mar 10, 2017. Instead, construct a FileStream.
// Unused function pvp_close_file was removed Mar 23, 2017.
// Unused function pvp_check_file_header was removed Mar 15, 2017.
// Unused functions pvp_read_header and pvp_write_header were removed Mar 15, 2017.
// Unused function pvp_set_activity_params was removed Jan 26, 2017.
// Unused function pvp_set_weight_params was removed Jan 26, 2017.
// Unused function pvp_set_nonspiking_act_params was removed Feb 21, 2017.
// Unused function pvp_set_nonspiking_sparse_act_params was removed Feb 21, 2017.
// Unused function alloc_params was removed Feb 21, 2017.

// writeActivity and writeActivitySparse removed Feb 17, 2017.
// Corresponding HyPerLayer methods now use BufferUtils routines
// gatherActivity and scatterActivity were also removed.
// Use BufferUtils::gather and BufferUtils::scatter instead.

// readWeights was removed Mar 15, 2017. Use readSharedWeights or readNonsharedWeights instead.

PatchListDescription createPatchListDescription(
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int nxp,
      int nyp,
      bool shared) {
   PatchListDescription patchListDesc;
   if (shared) {
      patchListDesc.mStartIndex  = 0;
      patchListDesc.mNumPatchesX = preLoc->nx > postLoc->nx ? preLoc->nx / postLoc->nx : 1;
      patchListDesc.mNumPatchesY = preLoc->ny > postLoc->ny ? preLoc->ny / postLoc->ny : 1;
      patchListDesc.mNumPatchesF = preLoc->nf;
      patchListDesc.mStrideY     = patchListDesc.mNumPatchesX * patchListDesc.mNumPatchesF;
   }
   else {
      int marginX        = requiredConvolveMargin(preLoc->nx, postLoc->nx, nxp);
      int numPatchesX    = preLoc->nx + marginX + marginX;
      int excessX        = preLoc->halo.lt - marginX;
      int numPatchesXExt = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
      int marginY        = requiredConvolveMargin(preLoc->ny, postLoc->ny, nyp);
      int numPatchesY    = preLoc->ny + marginY + marginY;
      int excessY        = preLoc->halo.up - marginY;
      int numPatchesYExt = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
      int numPatchesF    = preLoc->nf;
      patchListDesc.mStartIndex =
            kIndex(excessX, excessY, 0, numPatchesXExt, numPatchesYExt, numPatchesF);
      patchListDesc.mStrideY     = numPatchesXExt * numPatchesF;
      patchListDesc.mNumPatchesX = numPatchesX;
      patchListDesc.mNumPatchesY = numPatchesY;
      patchListDesc.mNumPatchesF = numPatchesF;
   }
   return patchListDesc;
}

void calcMinMaxPatch(
      float &minWeight,
      float &maxWeight,
      float const *patchData,
      unsigned int nf,
      unsigned int nx,
      unsigned int ny,
      unsigned int offset,
      unsigned int syp) {
   float const *patchDataOffset = &patchData[offset];
   int const nk                 = nx * nf;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         float v   = patchDataOffset[syp * y + k];
         minWeight = v < minWeight ? v : minWeight;
         maxWeight = v > maxWeight ? v : maxWeight;
      }
   }
}

void calcMinMaxNonsharedWeights(
      float &minWeight,
      float &maxWeight,
      float const *const *patchData,
      int numArbors,
      int nxp,
      int nyp,
      int nfp,
      PatchListDescription const &patchIndices,
      PVPatch const *const *const *patchGeometry) {
   int const patchStrideY           = nxp * nfp;
   int const numItemsInPatch        = patchStrideY * nyp;
   int const patchIndicesLineLength = patchIndices.mNumPatchesX * patchIndices.mNumPatchesF;
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int y = 0; y < patchIndices.mNumPatchesY; y++) {
         for (int k = 0; k < patchIndicesLineLength; k++) {
            int patchIndex = patchIndices.mStartIndex + patchIndices.mStrideY * y + k;
            PVPatch const *currentPatchGeometry = patchGeometry[arbor][patchIndex];
            unsigned int const nx               = (unsigned int)currentPatchGeometry->nx;
            unsigned int const ny               = (unsigned int)currentPatchGeometry->ny;
            unsigned int const offset           = currentPatchGeometry->offset;
            float const *currentPatchData       = &patchData[arbor][patchIndex * numItemsInPatch];
            calcMinMaxPatch(
                  minWeight, maxWeight, currentPatchData, nfp, nx, ny, offset, patchStrideY);
         }
      }
   }
}

void calcMinMaxSharedWeights(
      float &minWeight,
      float &maxWeight,
      float const *const *patchData,
      int numArbors,
      int nxp,
      int nyp,
      int nfp,
      PatchListDescription const &patchIndices) {
   int const patchStrideY    = nxp * nfp;
   int const numItemsInPatch = patchStrideY * nyp;
   int numPatches =
         patchIndices.mNumPatchesX * patchIndices.mNumPatchesY * patchIndices.mNumPatchesF;
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
         float const *currentPatchData = &patchData[arbor][patchIndex * numItemsInPatch];
         calcMinMaxPatch(minWeight, maxWeight, currentPatchData, nfp, nxp, nyp, 0, patchStrideY);
      }
   }
}

double readSharedWeights(
      FileStream *fileStream,
      int frameNumber,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF) {
   int const rootProc = 0;
   BufferUtils::WeightHeader header;
   if (mpiBlock->getRank() == rootProc) {
      setInPosByFrame(header, fileStream, frameNumber);
      // fileStream->read(&header, sizeof(header));
      FatalIf(
            header.baseHeader.fileType != PVP_KERNEL_FILE_TYPE,
            "readSharedWeights called with \"%s\", which is not a shared-weights file.\n",
            fileStream->getFileName().c_str());
   }
   MPI_Bcast(&header, (int)sizeof(header), MPI_BYTE, rootProc, mpiBlock->getComm());
   bool compressed = isCompressedHeader(header, fileStream->getFileName());

   FatalIf(
         header.baseHeader.nBands < numArbors,
         "File \"%s\" has only %d arbors, but readSharedWeights was called with %d arbors.\n",
         fileStream->getFileName().c_str(),
         header.baseHeader.nBands,
         numArbors);

   int numPatchItems           = nxp * nyp * nfp;
   std::size_t const patchSize = compressed
                                       ? BufferUtils::weightPatchSize<unsigned char>(numPatchItems)
                                       : BufferUtils::weightPatchSize<float>(numPatchItems);

   int const numPatches = (std::size_t)(numPatchesX * numPatchesY * numPatchesF);
   int const arborSize  = numPatches * (int)patchSize;
   std::vector<unsigned char> readBuffer(arborSize);
   for (int a = 0; a < numArbors; a++) {
      if (mpiBlock->getRank() == rootProc) {
         fileStream->read(readBuffer.data(), arborSize);
      }
      MPI_Bcast(readBuffer.data(), arborSize, MPI_BYTE, rootProc, mpiBlock->getComm());
      pvp_set_patches(
            readBuffer.data(),
            dataStart[a],
            numPatches,
            nxp,
            nyp,
            nfp,
            header.minVal,
            header.maxVal,
            compressed);
   }

   return header.baseHeader.timestamp;
}

double readNonsharedWeights(
      FileStream *fileStream,
      int frameNumber,
      MPIBlock const *mpiBlock,
      const PVLayerLoc *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool extended,
      const PVLayerLoc *postLoc,
      int offsetX,
      int offsetY) {
   int const rootProcess = 0;
   int const rank        = mpiBlock->getRank();
   BufferUtils::WeightHeader header;
   if (rank == rootProcess) {
      setInPosByFrame(header, fileStream, frameNumber);
      // fileStream->read(&header, sizeof(header));
      FatalIf(
            header.baseHeader.fileType != PVP_WGT_FILE_TYPE,
            "readSharedWeights called with \"%s\", which is not a shared-weights file.\n",
            fileStream->getFileName().c_str());
      FatalIf(
            header.baseHeader.nBands < numArbors,
            "File \"%s\" has only %d arbors, but readSharedWeights was called with %d arbors.\n",
            header.baseHeader.nBands,
            numArbors);
   }
   MPI_Bcast(&header, (int)sizeof(header), MPI_BYTE, 0 /*root*/, mpiBlock->getComm());
   bool compressed = isCompressedHeader(header, fileStream->getFileName());

   int numItemsPerPatch = nxp * nyp * nfp;
   std::size_t patchSize;
   if (compressed) {
      patchSize = BufferUtils::weightPatchSize<unsigned char>(numItemsPerPatch);
   }
   else {
      patchSize = BufferUtils::weightPatchSize<float>(numItemsPerPatch);
   }

   PatchListDescription patchListDesc =
         createPatchListDescription(preLoc, postLoc, nxp, nyp, false /*non-shared*/);

   int const numPatchesInLine = patchListDesc.mNumPatchesX * patchListDesc.mNumPatchesF;
   int const numPatchesLocal  = numPatchesInLine * patchListDesc.mNumPatchesY;

   std::size_t const localSize = (std::size_t)numPatchesLocal * patchSize;
   std::vector<unsigned char> arborDataInPvpFormat(localSize);

   int const numPatchesXGlobal =
         preLoc->nx * mpiBlock->getNumColumns() + (patchListDesc.mNumPatchesX - preLoc->nx);
   int const numPatchesYGlobal =
         preLoc->nx * mpiBlock->getNumRows() + (patchListDesc.mNumPatchesY - preLoc->ny);
   int const numPatchesGlobal = numPatchesXGlobal * numPatchesYGlobal * patchListDesc.mNumPatchesF;
   std::size_t const globalSize = (std::size_t)numPatchesGlobal * patchSize;

   int const tagBase = 600; // TODO: make static and increment each time readNonshared is called.

   if (rank == rootProcess) {
      long const frameStartInFile = fileStream->getOutPos();
      for (int arbor = 0; arbor < numArbors; arbor++) {
         long const arborStartInFile = frameStartInFile + (long)(arbor * globalSize);
         fileStream->setOutPos(arborStartInFile, true /*from beginning of file*/);
         int const rowsInBlock       = mpiBlock->getNumRows();
         int const columnsInBlock    = mpiBlock->getNumColumns();
         int const batchElemsInBlock = mpiBlock->getBatchDimension();
         for (int procRow = 0; procRow < rowsInBlock; procRow++) {
            for (int procColumn = 0; procColumn < columnsInBlock; procColumn++) {
               int const numPatchesX = patchListDesc.mNumPatchesX;
               int const numPatchesY = patchListDesc.mNumPatchesY;
               int const numPatchesF = patchListDesc.mNumPatchesF;
               int const numPatches  = numPatchesX * numPatchesY * numPatchesF;
               for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
                  unsigned char *patchStart =
                        &arborDataInPvpFormat[(std::size_t)patchIndex * patchSize];
                  int const xIndex = kxPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const yIndex = kyPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const fIndex =
                        featureIndex(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const xIndexGlobal     = xIndex + preLoc->nx * procColumn;
                  int const yIndexGlobal     = yIndex + preLoc->ny * procRow;
                  int const patchIndexGlobal = kIndex(
                        xIndexGlobal,
                        yIndexGlobal,
                        fIndex,
                        numPatchesXGlobal,
                        numPatchesYGlobal,
                        numPatchesF);
                  long const offsetIntoArbor  = (long)patchSize * (long)patchIndexGlobal;
                  long const patchStartInFile = arborStartInFile + offsetIntoArbor;
                  // When reading; just copy the whole patch, ignoring shrunken patch information.
                  fileStream->setOutPos(patchStartInFile, true);
                  fileStream->read(patchStart, patchSize);
               }
               // arborDataInPvpFormat now has the raw data from the file.
               for (int procBatchElem = 0; procBatchElem < batchElemsInBlock; procBatchElem++) {
                  int destRank =
                        mpiBlock->calcRankFromRowColBatch(procRow, procColumn, procBatchElem);
                  if (destRank == rank) {
                     pvp_set_arbor(
                           arborDataInPvpFormat,
                           dataStart[arbor],
                           patchListDesc,
                           nxp,
                           nyp,
                           nfp,
                           patchSize,
                           header.minVal,
                           header.maxVal,
                           compressed);
                  }
                  else {
                     MPI_Send(
                           arborDataInPvpFormat.data(),
                           localSize,
                           MPI_BYTE,
                           destRank,
                           tagBase + arbor,
                           mpiBlock->getComm());
                  }
               }
            }
         }
      }
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         MPI_Recv(
               arborDataInPvpFormat.data(),
               localSize,
               MPI_BYTE,
               rootProcess,
               tagBase + arbor,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
         pvp_set_arbor(
               arborDataInPvpFormat,
               dataStart[arbor],
               patchListDesc,
               nxp,
               nyp,
               nfp,
               patchSize,
               header.minVal,
               header.maxVal,
               compressed);
      }
   }

   return header.baseHeader.timestamp;
}

void setInPosByFrame(BufferUtils::WeightHeader &header, FileStream *fileStream, int frameNumber) {
   fileStream->setInPos(0L, true /*from beginning*/);
   for (int f = 0; f < frameNumber; f++) {
      fileStream->read(&header, sizeof(header));
      long recordSize = (long)(header.baseHeader.recordSize * header.baseHeader.numRecords);
      fileStream->setInPos(recordSize, false /*relative to current point*/);
   }
   fileStream->read(&header, sizeof(header));
}

bool isCompressedHeader(BufferUtils::WeightHeader const &header, std::string const &filename) {
   bool isCompressed;
   BufferUtils::ActivityHeader const &baseHeader = header.baseHeader;
   switch (baseHeader.dataType) {
      case BufferUtils::BYTE:
         FatalIf(
               baseHeader.dataSize != (int)sizeof(unsigned char),
               "File \"%s\" has dataSize=%d, inconsistent with dataType BYTE (%d)\n",
               filename.c_str(),
               baseHeader.dataSize,
               baseHeader.dataType);
         isCompressed = true;
         break;
      case BufferUtils::FLOAT:
         FatalIf(
               baseHeader.dataSize != (int)sizeof(float),
               "File \"%s\" has dataSize=%d, inconsistent with dataType FLOAT (%d)\n",
               filename.c_str(),
               baseHeader.dataSize,
               baseHeader.dataType);
         isCompressed = false;
         break;
      case BufferUtils::INT:
         Fatal().printf(
               "File \"%s\" has dataType INT. Only FLOAT and BYTE are supported.\n",
               filename.c_str());
         break;
      default: Fatal().printf("File \"%s\" has unrecognized datatype.\n", filename.c_str()); break;
   }
   return isCompressed;
}

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
      double timed,
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool compress,
      float minVal,
      float maxVal,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF) {
   if (fileStream == nullptr) {
      return;
   }

   BufferUtils::WeightHeader header = BufferUtils::buildSharedWeightHeader(
         nxp,
         nyp,
         nfp,
         numArbors,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         timed,
         compress,
         minVal,
         maxVal);
   fileStream->write(&header, sizeof(header));

   std::size_t const localSize = header.baseHeader.recordSize;
   std::vector<unsigned char> arborDataInPvpFormat(localSize);
   int const numPatches = numPatchesX * numPatchesY * numPatchesF;
   for (int arbor = 0; arbor < numArbors; arbor++) {
      float const *arborStart = dataStart[arbor];
      pvp_copy_patches(
            arborDataInPvpFormat.data(),
            nullptr,
            arborStart,
            numPatches,
            nxp,
            nyp,
            nfp,
            minVal,
            maxVal,
            compress);
      fileStream->write(arborDataInPvpFormat.data(), arborDataInPvpFormat.size());
   }
}

void writeNonsharedWeights(
      double timed,
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      const PVLayerLoc *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool compress,
      bool extended,
      const PVLayerLoc *postLoc,
      PVPatch const *const *const *patchGeometry) {
   // Assume weights are the same for each batch element; only write for first element.
   if (mpiBlock->getBatchIndex() != 0) {
      return;
   }

   PatchListDescription patchListDesc =
         createPatchListDescription(preLoc, postLoc, nxp, nyp, false /*non-shared*/);

   std::vector<float> extrema = {std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};

   calcMinMaxNonsharedWeights(
         extrema[0], extrema[1], dataStart, numArbors, nxp, nyp, nfp, patchListDesc, patchGeometry);

   // Reduce across MPI: min for element 0 and max for element 1.
   // To do in a single MPI_Allreduce call, switch sign of one, reduce, and switch back.
   extrema[0] = -extrema[0];
   MPI_Allreduce(MPI_IN_PLACE, extrema.data(), 2, MPI_FLOAT, MPI_MAX, mpiBlock->getComm());
   extrema[0] = -extrema[0];

   std::size_t patchSize;
   if (compress) {
      patchSize = BufferUtils::weightPatchSize<unsigned char>(nxp * nyp * nfp);
   }
   else {
      patchSize = BufferUtils::weightPatchSize<float>(nxp * nyp * nfp);
   }

   int const numPatchesInLine = patchListDesc.mNumPatchesX * patchListDesc.mNumPatchesF;
   int const numPatchesLocal  = numPatchesInLine * patchListDesc.mNumPatchesY;

   std::size_t const localSize = (std::size_t)numPatchesLocal * patchSize;
   std::vector<unsigned char> arborDataInPvpFormat(localSize);

   int const numPatchesXGlobal =
         preLoc->nx * mpiBlock->getNumColumns() + (patchListDesc.mNumPatchesX - preLoc->nx);
   int const numPatchesYGlobal =
         preLoc->ny * mpiBlock->getNumRows() + (patchListDesc.mNumPatchesY - preLoc->ny);
   int const numPatchesGlobal = numPatchesXGlobal * numPatchesYGlobal * patchListDesc.mNumPatchesF;
   std::size_t const globalSize = (std::size_t)numPatchesGlobal * patchSize;

   int const tagbase     = 500;
   int const rootProcess = 0;
   int const rank        = mpiBlock->getRank();
   if (rank == rootProcess) {
      pvAssert(fileStream != nullptr) BufferUtils::WeightHeader header =
            BufferUtils::buildNonsharedWeightHeader(
                  nxp,
                  nyp,
                  nfp,
                  numArbors,
                  extended,
                  timed,
                  preLoc,
                  postLoc,
                  mpiBlock->getNumColumns(),
                  mpiBlock->getNumRows(),
                  extrema[0],
                  extrema[1],
                  compress);
      fileStream->write(&header, sizeof(header));

      long const frameStartInFile = fileStream->getOutPos();
      for (int arbor = 0; arbor < numArbors; arbor++) {
         long const arborStartInFile = frameStartInFile + (long)(arbor * globalSize);
         fileStream->setOutPos(arborStartInFile, true /*from beginning of file*/);
         int const rowsInBlock    = mpiBlock->getNumRows();
         int const columnsInBlock = mpiBlock->getNumColumns();
         for (int procRow = 0; procRow < rowsInBlock; procRow++) {
            for (int procColumn = 0; procColumn < columnsInBlock; procColumn++) {
               int sourceRank = mpiBlock->calcRankFromRowColBatch(procRow, procColumn, 0);
               if (sourceRank == rank) {
                  pvp_copy_arbor(
                        arborDataInPvpFormat,
                        patchGeometry[arbor],
                        dataStart[arbor],
                        patchListDesc,
                        nxp,
                        nyp,
                        nfp,
                        patchSize,
                        extrema[0],
                        extrema[1],
                        compress);
               }
               else {
                  MPI_Recv(
                        arborDataInPvpFormat.data(),
                        localSize,
                        MPI_BYTE,
                        sourceRank,
                        tagbase + arbor,
                        mpiBlock->getComm(),
                        MPI_STATUS_IGNORE);
               }
               // arborData now has the patch information from the sourceRank process.
               // Write each patch to the correct part of the file.
               // For shrunken patches, write only the shrunken part.
               int const numPatchesX = patchListDesc.mNumPatchesX;
               int const numPatchesY = patchListDesc.mNumPatchesY;
               int const numPatchesF = patchListDesc.mNumPatchesF;
               int const numPatches  = numPatchesX * numPatchesY * numPatchesF;
               for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
                  unsigned char const *patchStart =
                        &arborDataInPvpFormat[(std::size_t)patchIndex * patchSize];
                  int const xIndex = kxPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const yIndex = kyPos(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const fIndex =
                        featureIndex(patchIndex, numPatchesX, numPatchesY, numPatchesF);
                  int const xIndexGlobal     = xIndex + preLoc->nx * procColumn;
                  int const yIndexGlobal     = yIndex + preLoc->ny * procRow;
                  int const patchIndexGlobal = kIndex(
                        xIndexGlobal,
                        yIndexGlobal,
                        fIndex,
                        numPatchesXGlobal,
                        numPatchesYGlobal,
                        numPatchesF);
                  long const offsetIntoArbor  = (long)patchSize * (long)patchIndexGlobal;
                  long const patchStartInFile = arborStartInFile + offsetIntoArbor;
                  PatchHeader patchHeader;
                  pvAssert(
                        sizeof(patchHeader) == 2 * sizeof(unsigned short) + sizeof(unsigned int));
                  memcpy(&patchHeader, patchStart, sizeof(patchHeader));
                  fileStream->setOutPos(patchStartInFile, true /*from beginning of file*/);
                  if (patchHeader.nx == nxp and patchHeader.ny == nyp) {
                     fileStream->write(patchStart, patchSize);
                  }
                  else { // handle shrunken patch
                     // For convenience, all patches are marked unshrunken in the file.
                     // If we didn't do this, we'd have to track the splitting of patches
                     // near
                     // an MPI boundary.
                     PatchHeader unshrunken;
                     unshrunken.nx     = (short int)nxp;
                     unshrunken.ny     = (short int)nxp;
                     unshrunken.offset = 0U;

                     // Only write the shrunken part of the patch.
                     fileStream->write(&unshrunken, sizeof(unshrunken));
                     std::size_t dataSize = (compress ? sizeof(unsigned char) : sizeof(float));
                     std::size_t lineSize = (std::size_t)patchHeader.nx * (std::size_t)nfp;
                     lineSize *= dataSize;
                     for (short int y = 0; y < patchHeader.ny; y++) {
                        unsigned lineOffset = patchHeader.offset + kIndex(0, y, 0, nxp, nyp, nfp);
                        lineOffset          = lineOffset * dataSize + sizeof(unshrunken);
                        fileStream->setOutPos(patchStartInFile + lineOffset, true);
                        fileStream->write(&patchStart[lineOffset], lineSize);
                     } // Loop over line within patch
                  } // end handle shrunken patch
               } // Loop over patchIndex
            } // Loop over procColumn
         } // Loop over procRow
      } // Loop over arbor
      long currentPosition = fileStream->getOutPos();
      long frameEndInFile  = frameStartInFile + (long)numArbors * (long)globalSize;
      long diff            = frameEndInFile - currentPosition;
      pvAssert(diff >= 0);
      if (diff > 0) {
         std::vector<unsigned char> endOfFrame(diff);
         fileStream->write(endOfFrame.data(), endOfFrame.size());
      }
      fileStream->setOutPos(frameEndInFile, true /*from beginning of file*/);
   } // end if(rank==rootProcess)
   else {
      pvAssert(fileStream == nullptr);

      for (int arbor = 0; arbor < numArbors; arbor++) {
         pvp_copy_arbor(
               arborDataInPvpFormat,
               patchGeometry[arbor],
               dataStart[arbor],
               patchListDesc,
               nxp,
               nyp,
               nfp,
               patchSize,
               extrema[0],
               extrema[1],
               compress);
         MPI_Send(
               arborDataInPvpFormat.data(),
               localSize,
               MPI_BYTE,
               rootProcess,
               tagbase + arbor,
               mpiBlock->getComm());
      }
   }
}

} // namespace PV
