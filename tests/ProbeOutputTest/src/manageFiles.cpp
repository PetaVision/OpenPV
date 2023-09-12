#include "manageFiles.hpp"

int deleteLink(std::string const &path) {
   int status = ::unlink(path.c_str());
   if (status) {
      WarnLog().printf(
            "Failure deleting symbolic link\"%s\": %s\n", 
            path.c_str(), 
            std::strerror(errno));
   }
   return status ? PV_FAILURE : PV_SUCCESS;
}

int deleteRegularFile(std::string const &path) {
   int status = ::unlink(path.c_str());
   if (status) {
      WarnLog().printf(
            "Failure deleting regular file \"%s\": %s\n", 
            path.c_str(), 
            std::strerror(errno));
   }
   return status ? PV_FAILURE : PV_SUCCESS;
}

int recursiveDelete(std::string const &path) {
   struct stat statbuf;
   int statstatus = ::lstat(path.c_str(), &statbuf);
   if (statstatus != 0 and errno == ENOENT) { return PV_SUCCESS; }
   if (statstatus) {
      WarnLog() << "Failure accessing \"" << path << "\": " << std::strerror(errno) << "\n";
      return PV_FAILURE;
   }
   if (S_ISREG(statbuf.st_mode)) {
      return deleteRegularFile(path);
   }
   else if (S_ISDIR(statbuf.st_mode)) {
      return recursiveDeleteDirectory(path);
   }
   else if (S_ISLNK(statbuf.st_mode)) {
      return deleteLink(path);
   }
   else {
      WarnLog().printf(
            "Failure deleting \"%s\": Unrecognized inode file type %07o\n",
            path.c_str(),
            statbuf.st_mode & S_IFMT);
      return PV_FAILURE;
   }
}

int recursiveDelete(std::string const &path, Communicator *comm, bool warnIfAbsentFlag) {
   // delete a directory recursively, allowing for the possibility that more than one process might
   // see the same file system, even if they are in different M-to-N blocks. We don't use the
   // FileManager routines because we sometimes delete a directory not in a tree controlled by
   // a FileManager object.
   int status = PV_SUCCESS;
   int commSize;
   MPI_Comm_size(comm->globalCommunicator(), &commSize);
   int commRank;
   MPI_Comm_rank(comm->globalCommunicator(), &commRank);

   struct stat statbuf;
   int pathExists = 0;
   int statstatus = ::lstat(path.c_str(), &statbuf);
   if (!statstatus) { pathExists = 1; }

   auto globalComm = comm->globalCommunicator();
   MPI_Allreduce(MPI_IN_PLACE, &pathExists, 1 /*count*/, MPI_INT, MPI_LOR, globalComm);
   if (!pathExists) {
      if (warnIfAbsentFlag) {
         WarnLog().printf(
               "recursiveDelete() called for path \"%s\", but this path does not exist.\n",
               path.c_str());
         return PV_FAILURE;
      }
      else {
         return PV_SUCCESS;
      }
   }

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank) {
         if (comm->getIOMPIBlock()->getRank() == 0) {
            int status1 = recursiveDelete(path);
            if (status1 != PV_SUCCESS) { status = PV_FAILURE; }
         }
      }

      // Prevent collisions if multiple processes try to delete the same file.
      MPI_Barrier(comm->globalCommunicator());
   }
   return status;
}

int recursiveDeleteDirectory(std::string const &path) {
   int status = PV_SUCCESS;
   DIR *dir = opendir(path.c_str());
   if (dir == nullptr) {
      WarnLog().printf(
            "Failure listing directory \"%s\": %s\n",
            path.c_str(),
            std::strerror(errno));
      status = PV_FAILURE;
   }
   struct dirent *dirEntry;
   for (dirEntry = readdir(dir); dirEntry; dirEntry = readdir(dir)) {
      std::string dirEntryString(dirEntry->d_name);
      if (dirEntryString == "." or dirEntryString == "..") { continue; }
      int status1 = recursiveDelete(path + "/" + dirEntryString);
      if (status1 != PV_SUCCESS) { status = PV_FAILURE; }
   }
   if (status != PV_SUCCESS) {
      return status;
   }
   int rmdirstatus = ::rmdir(path.c_str());
   if (rmdirstatus) {
      WarnLog().printf(
            "Failure deleting directory \"%s\": %s\n",
            path.c_str(),
            std::strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}

int renamePath(std::string const &oldpath, std::string const &newpath, Communicator *comm) {
   int status = recursiveDelete(newpath, comm, false /*warnIfAbsentFlag*/);
   if (status != PV_SUCCESS) {
      ErrorLog().printf(
            "Failure deleting \"%s\" in order to rename \"%s\" to \"%s\": %s\n",
            newpath.c_str(), oldpath.c_str(), newpath.c_str());
      return PV_FAILURE;
   }

   int processRenamedFile = 0;

   int commSize;
   MPI_Comm_size(comm->globalCommunicator(), &commSize);
   int commRank;
   MPI_Comm_rank(comm->globalCommunicator(), &commRank);

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank) {
         bool processShouldRename = comm->getIOMPIBlock()->getRank() == 0;
         if (processShouldRename) {
            struct stat statbuf;
            int oldstatstatus = ::lstat(oldpath.c_str(), &statbuf);
            if (oldstatstatus != 0) {
               int lstatError = errno;
               if (errno != ENOENT) {
                  WarnLog().printf(
                        "Failure accessing \"%s\": %s\n",
                        oldpath.c_str(), std::strerror(lstatError));
                  status = PV_FAILURE;
               }
               processShouldRename = false;
            }
         }
         if (processShouldRename) {
            // If we're here, oldpath exists on the system.
            // Check that newpath does not exist (we called recursiveDelete above),
            // then call rename().
            struct stat statbuf;
            int newstatstatus = ::lstat(newpath.c_str(), &statbuf);
            if (newstatstatus == 0) {
               ErrorLog().printf(
                     "Failure moving \"%s\" to \"%s\": %s exists when it shouldn't.\n",
                     oldpath.c_str(), newpath.c_str(), newpath.c_str());
               status = PV_FAILURE;
               processShouldRename = false;
            }
            if (errno != ENOENT) {
               ErrorLog().printf(
                     "Failure getting status of \"%s\" for renaming \"%s\" to \"%s\": %s\n",
                     newpath.c_str(), oldpath.c_str(), newpath.c_str(), std::strerror(errno));
               status = PV_FAILURE;
               processShouldRename = false;
            }
         }
         if (processShouldRename) {
            int status1 = ::rename(oldpath.c_str(), newpath.c_str());
            if (status1 == 0) {
               processRenamedFile = 1;
            }
            else {
               int renameErrorNumber = errno;
               ErrorLog().printf(
                     "Failure renaming \"%s\" to \"%s\": %s\n",
                     oldpath.c_str(), newpath.c_str(), std::strerror(renameErrorNumber));
               status = PV_FAILURE;
            }
         }
      }

      // Prevent collisions if multiple processes try to move the same file.
      MPI_Barrier(comm->globalCommunicator());
   }
   auto globalComm = comm->globalCommunicator();
   MPI_Allreduce(
         MPI_IN_PLACE, &processRenamedFile, 1 /*count*/, MPI_INT, MPI_LOR, globalComm);
   if (!processRenamedFile) {
      ErrorLog().printf(
            "Failure moving \"%s\" to \"%s\": no MPI process reported moving the file.\n",
            oldpath.c_str(), newpath.c_str());
      status = PV_FAILURE;
   }
   return status;
}
