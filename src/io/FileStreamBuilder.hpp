#ifndef FILESTREAMBUILDER_HPP_
#define FILESTREAMBUILDER_HPP_

#include "io/FileManager.hpp"
#include "io/FileStream.hpp"

#include <memory>

namespace PV {

class FileStreamBuilder {
  public:
   FileStreamBuilder(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      bool isText,
      bool readOnlyFlag,
      bool verifyWrites = false);

   ~FileStreamBuilder();

   std::shared_ptr<FileStream> get() { return mFileStream; }

  private:
   std::shared_ptr<FileStream> mFileStream;
};  // class FileStreamBuilder

} // namespace PV


#endif // FILESTREAMBUILDER_HPP_
