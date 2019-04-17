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
#include "utils/ExpandLeadingTilde.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"

#include <assert.h>
#include <iostream>

#undef DEBUG_OUTPUT

namespace PV {

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
   if (mpiBlock->getRank() != 0) {
      return;
   }

   std::string expandedDirName = expandLeadingTilde(dirname);
   struct stat pathstat;
   int statresult = stat(expandedDirName.c_str(), &pathstat);

   // If path exists but is not a directory, fatal error.
   FatalIf(
         statresult == 0 and !(pathstat.st_mode & S_IFDIR),
         "Path \"%s\" exists but is not a directory\n",
         dirname);

   // Fatal error if checking the path gave an error other than No such file or directory
   FatalIf(
         statresult != 0 and errno != ENOENT,
         "Checking status of directory \"%s\" gave error \"%s\".\n",
         dirname,
         strerror(errno));

   InfoLog().printf("Directory \"%s\" does not exist; attempting to create\n", dirname);

   // Try up to MAX_FILESYSTEMCALL_TRIES times until it works
   for (int attemptNum = 0; attemptNum < MAX_FILESYSTEMCALL_TRIES; attemptNum++) {
      int mkdirstatus = makeDirectory(expandedDirName.c_str());
      if (mkdirstatus != 0) {
         if (attemptNum == MAX_FILESYSTEMCALL_TRIES - 1) {
            Fatal().printf(
                  "Directory \"%s\" could not be created: %s; Exiting\n", dirname, strerror(errno));
         }
         else {
            getOutputStream().flush();
            WarnLog().printf(
                  "Directory \"%s\" could not be created: %s; Retrying %d out of %d\n",
                  dirname,
                  strerror(errno),
                  attemptNum + 1,
                  MAX_FILESYSTEMCALL_TRIES);
            sleep(1);
         }
      }
      else {
         InfoLog().printf("Successfully created directory \"%s/\".\n", dirname);
         errno = 0;
         break;
      }
   }
}

} // namespace PV
