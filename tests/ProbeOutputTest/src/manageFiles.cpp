#include "manageFiles.hpp"

#include <filesystem>

int appendExtraneousData(std::filesystem::path const &path, Communicator *comm) {
   int status = PV_SUCCESS;
   auto globalComm = comm->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   auto const badSize = static_cast<std::uintmax_t>(-1);
   auto fileSize = badSize;
   if (comm->getIOMPIBlock()->getRank() == 0) {
      auto filestatus = std::filesystem::status(path);
      auto file_type = filestatus.type();
      switch (file_type) {
         case std::filesystem::file_type::regular:
            fileSize = std::filesystem::file_size(path);
            break;
         case std::filesystem::file_type::not_found:
            // NOP; presumably another process on a different filesystem will find the file
            break;
         default:
            ErrorLog().printf(
                  "appendExtraneousData() failed: \"%s\" is not a regular file.\n",
                  path.c_str());
            status = PV_FAILURE;
            break;
      }
   }
   MPI_Barrier(comm->globalCommunicator());

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank) {
         if (comm->getIOMPIBlock()->getRank() == 0) {
            if (status == PV_SUCCESS and fileSize != badSize) {
               auto newFileSize = std::filesystem::file_size(path);
               if (newFileSize == fileSize) {
                  std::fstream outputFile(path.string(), std::ios_base::app);
                  if (outputFile) {
                     outputFile << "PetaVision is an open source, object-oriented neural ";
                     outputFile << "simulation toolbox optimized for high-performance ";
                     outputFile << "multi-core, multi-node computer architectures.\n";
                  }
                  else {
                     ErrorLog().printf("Unable to open \"%s\" for appending.\n", path.c_str());
                     status = PV_FAILURE;
                     continue;
                  }
               }
            }
         }
      }

      // Prevent collisions if multiple root processes see the same filesystem
      MPI_Barrier(comm->globalCommunicator());
   }
   return status;
}

int recursiveCopy(std::string const &from, std::string const &to, Communicator *comm) {
   // copy a directory and its contents, allowing for the possibility that more than one process
   // might see the same file system, even if they are in different M-to-N blocks.
   int status = PV_SUCCESS;
   auto globalComm = comm->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   auto filestatus = std::filesystem::symlink_status(from);
   // will throw a filesystem_error on failure ("No such file or directory" is not a failure;
   // instead file_type::not_found is returned)
   int fromPathExists = (filestatus.type() == std::filesystem::file_type::not_found) ? 0 : 1;

   MPI_Allreduce(MPI_IN_PLACE, &fromPathExists, 1 /*count*/, MPI_INT, MPI_LOR, globalComm);
   if (!fromPathExists) {
      ErrorLog().printf(
            "recursiveCopy() called for source path \"%s\", but this path does not exist.\n",
            from.c_str());
      return PV_FAILURE;
   }

   status = recursiveDelete(to, comm, false /*warnIfAbsentFlag*/);
   if (status != PV_SUCCESS) {
      ErrorLog().printf(
            "recursiveCopy() failed to delete previously existing destination path \"%s\"\n",
            to.c_str());
      return PV_FAILURE;
   }

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank) {
         if (comm->getIOMPIBlock()->getRank() == 0) {
            auto filestatus = std::filesystem::symlink_status(to);
            if (filestatus.type() == std::filesystem::file_type::not_found) {
               std::error_code errorCode;
               std::filesystem::copy(from, to, std::filesystem::copy_options::recursive, errorCode);
               if (errorCode.value() != 0) { status = PV_FAILURE; }
            }
         }
      }

      // Prevent collisions if multiple processes on one filesystem try to delete the same file.
      MPI_Barrier(comm->globalCommunicator());
   }
   return status;
}

int recursiveDelete(std::string const &path, Communicator *comm, bool warnIfAbsentFlag) {
   // delete a directory recursively, allowing for the possibility that more than one process might
   // see the same file system, even if they are in different M-to-N blocks. We don't use the
   // FileManager routines because we sometimes delete a directory not in a tree controlled by
   // a FileManager object.
   int status = PV_SUCCESS;
   auto globalComm = comm->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   auto filestatus = std::filesystem::symlink_status(path);
   // will throw a filesystem_error on failure ("No such file or directory" is not a failure;
   // instead file_type::not_found is returned)
   int pathExists = (filestatus.type() == std::filesystem::file_type::not_found) ? 0 : 1;

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
            std::error_code errorCode;
            std::uintmax_t numRemoved = std::filesystem::remove_all(path, errorCode);
            if (errorCode.value() != 0) { status = PV_FAILURE; }
         }
      }

      // Prevent collisions if multiple processes on one filesystem try to delete the same file.
      MPI_Barrier(comm->globalCommunicator());
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

   auto globalComm = comm->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank) {
         bool processShouldRename = comm->getIOMPIBlock()->getRank() == 0;
         if (processShouldRename) {
            auto oldFileStatus = std::filesystem::symlink_status(oldpath);
            bool oldPathExists = oldFileStatus.type() != std::filesystem::file_type::not_found;
            auto newFileStatus = std::filesystem::symlink_status(newpath);
            bool newPathExists = newFileStatus.type() != std::filesystem::file_type::not_found;
            processShouldRename = oldPathExists and !newPathExists;
         }
         if (processShouldRename) {
            std::filesystem::rename(oldpath, newpath);
            processRenamedFile = 1;
         }
      }

      // Prevent collisions if multiple processes try to move the same file.
      MPI_Barrier(globalComm);
   }
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
