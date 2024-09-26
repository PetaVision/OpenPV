#include "FileStreamBuilder.hpp"

namespace PV {

FileStreamBuilder::FileStreamBuilder(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      bool isTextFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWritesFlag) {

   std::ios_base::openmode mode = std::ios_base::in;
   if (!readOnlyFlag) { mode |= std::ios_base::out; }
   if (!isTextFlag) { mode |= std::ios_base::binary; }

   // If the file is opened with mode out|in but the file does not exist, FileManager::open()
   // will throw an error. We must first create the file in write mode, then close and reopen it
   // in read/write mode.
   // If however, the file is opened with mode out|in and the file does exist, FileManager::open()
   // will open it as is; however, we might want to delete existing contents. If the
   // clobber flag is true, we also create the file in write mode, which deletes any existing
   // file, and then close and reopen it in read/write mode.
   if (!readOnlyFlag) {
      struct stat statinfo;
      bool createNewFile = false;
      if (clobberFlag) {
         createNewFile = true;
      }
      else {
         // create the file only if it doesn't already exist
         int result = fileManager->stat(path, statinfo);
         if (result != 0) {
             if (errno == ENOENT) {
                errno = 0;
                createNewFile = true;
             }
             else {
                std::string errorMessage;
                errorMessage.append("Unable to open \"").append(path).append("\" in read/write mode");
                throw std::system_error(std::error_code(errno, std::generic_category()), errorMessage);
             }
         }
      }
      if (createNewFile) {
         mFileStream = fileManager->open(path, std::ios_base::out, verifyWritesFlag);
         mFileStream = nullptr; // Close the file; it will be reopened with the correct mode below
      }
   }

   mFileStream = fileManager->open(path, mode, verifyWritesFlag);
}

FileStreamBuilder::~FileStreamBuilder() {}

} // namespace PV
