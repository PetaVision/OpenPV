#include "FileStreamBuilder.hpp"

namespace PV {

FileStreamBuilder::FileStreamBuilder(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      bool isText,
      bool readOnlyFlag,
      bool verifyWrites) {

   std::ios_base::openmode mode = std::ios_base::in;
   if (!readOnlyFlag) { mode |= std::ios_base::out; }
   if (!isText) { mode |= std::ios_base::binary; }

   // If the file is opened with mode out|in but the file does not exist, FileManager::open()
   // will throw an error. We must first create the file in write mode, then close and reopen it
   // in read/write mode.
   if (!readOnlyFlag) {
      struct stat statinfo;
      int result = fileManager->stat(path, statinfo);
      if (result != 0) {
          if (errno == ENOENT) {
             errno = 0;
             auto newFile = fileManager->open(path, std::ios_base::out, verifyWrites);
          }
          else {
             std::string errorMessage;
             errorMessage.append("Unable to open \"").append(path).append("\" in read/write mode");
             throw std::system_error(std::error_code(errno, std::generic_category()), errorMessage);
          }
      }
   }

   mFileStream = fileManager->open(path, mode, verifyWrites);
}

FileStreamBuilder::~FileStreamBuilder() {}

} // namespace PV
