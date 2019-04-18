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

void ensureDirExists(MPIBlock const *mpiBlock, char const *dirname) {
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
}

// Unused function getNumGlobalPatches was removed Mar 15, 2017.
// Instead, use calcNumberOfPatches in utils/BufferUtilsPvp.*

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

// readWeights was removed Mar 15, 2017. Use the WeightsFileIO class instead.

int pv_text_write_patch(
      PrintStream *outStream,
      Patch const *patch,
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

} // namespace PV
