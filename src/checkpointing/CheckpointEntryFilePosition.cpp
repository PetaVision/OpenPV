#include "CheckpointEntryFilePosition.hpp"

#include "utils/PVAssert.hpp"

namespace PV {

void CheckpointEntryFilePosition::write(
      std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
   if (!fileManager->isRoot()) { return; }

   writeValueToBinAndTxt(
         fileManager,
         std::string("FileStreamRead"),
         mFileStream->getInPos(),
         verifyWritesFlag);

   writeValueToBinAndTxt(
         fileManager,
         std::string("FileStreamWrite"),
         mFileStream->getOutPos(),
         verifyWritesFlag);
}

void CheckpointEntryFilePosition::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   if (!fileManager->isRoot()) { return; }

   long readPosition;
   readValueFromBin(
         fileManager, std::string("FileStreamRead"), &readPosition);
   mFileStream->setInPos(readPosition, std::ios_base::beg);

   long writePosition;
   readValueFromBin(
         fileManager, std::string("FileStreamWrite"), &writePosition);
   mFileStream->setOutPos(writePosition, std::ios_base::beg);
}

void CheckpointEntryFilePosition::remove(std::shared_ptr<FileManager const> fileManager) const {
   std::string filenamebase(getName());
   filenamebase.append("_FileStream");
   fileManager->deleteFile(filenamebase + "Read.bin");
   fileManager->deleteFile(filenamebase + "Read.txt");
   fileManager->deleteFile(filenamebase + "Write.bin");
   fileManager->deleteFile(filenamebase + "Write.txt");
}

void CheckpointEntryFilePosition::readValueFromBin(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &label,
      long *value) const {
   pvAssert(fileManager->isRoot());
   std::string filenamebase(getName());
   filenamebase.append("_").append(label);
   std::string filename = filenamebase + ".bin";
   auto fileStream = fileManager->open(filename.c_str(), std::ios_base::in, false);
   fileStream->read(value, sizeof(*value));
}

void CheckpointEntryFilePosition::writeValueToBinAndTxt(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &label,
      long value,
      bool verifyWritesFlag) const {
   pvAssert(fileManager->isRoot());
   std::string filenamebase(getName());
   filenamebase.append("_").append(label);

   std::string filename = filenamebase + ".bin";
   auto fileStream = fileManager->open(filename.c_str(), std::ios_base::out, verifyWritesFlag);
   fileStream->write(&value, sizeof(value));

   filename = filenamebase + ".txt";
   fileStream = fileManager->open(filename.c_str(), std::ios_base::out, verifyWritesFlag);
   *fileStream << value << "\n";
}

} // namespace PV
