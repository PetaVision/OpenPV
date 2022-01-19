#include "CheckpointEntryMPIRecvStream.hpp"
#include <sys/stat.h>

namespace PV {

CheckpointEntryMPIRecvStream::CheckpointEntryMPIRecvStream(
      std::string const &name,
      MPIRecvStream &mpiRecvStream)
      : CheckpointEntry(name), mMPIRecvStream(&mpiRecvStream) {}

CheckpointEntryMPIRecvStream::CheckpointEntryMPIRecvStream(
      std::string const &objName,
      std::string const &dataName,
      MPIRecvStream &mpiRecvStream)
      : CheckpointEntry(objName, dataName), mMPIRecvStream(&mpiRecvStream) {}

void CheckpointEntryMPIRecvStream::write(
      std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
   if (fileManager->isRoot()) {
      long position;

      position = mMPIRecvStream->getOutPos();
      std::string writeposPath = getName() + "_FileStreamWrite.bin";
      auto writePosFileStream  =
            fileManager->open(writeposPath, std::ios_base::out, verifyWritesFlag);
      writePosFileStream->write(&position, static_cast<long>(sizeof(position)));

      std::string writeposPathTxt = getName() + "_FileStreamWrite.txt";
      auto writePosFileStreamTxt =
            fileManager->open(writeposPathTxt, std::ios_base::out, verifyWritesFlag);
      *writePosFileStreamTxt << position << "\n";

      position = mMPIRecvStream->getInPos();
      std::string readposPath = getName() + "_FileStreamRead.bin";
      auto readPosFileStream =
            fileManager->open(readposPath.c_str(), std::ios_base::out, verifyWritesFlag);
      readPosFileStream->write(&position, static_cast<long>(sizeof(position)));

      std::string readposPathTxt = getName() + "_FileStreamRead.txt";
      auto readPosFileStreamTxt =
            fileManager->open(readposPathTxt.c_str(), std::ios_base::out, verifyWritesFlag);
      *readPosFileStreamTxt << position << "\n";
   }
}

void CheckpointEntryMPIRecvStream::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   if (fileManager->isRoot()) {
      long position;

      std::string writeposPath = getName() + "_FileStreamWrite.bin";
      auto writePosFileStream =
            fileManager->open(writeposPath, std::ios_base::in, false);
      writePosFileStream->read(&position, static_cast<long>(sizeof(position)));
      mMPIRecvStream->setOutPos(position);

      std::string readposPath = getName() + "_FileStreamRead.bin";
      auto readPosFileStream =
            fileManager->open(readposPath, std::ios_base::in, false);
      readPosFileStream->read(&position, static_cast<long>(sizeof(position)));
      mMPIRecvStream->setInPos(position);
   }
}

void CheckpointEntryMPIRecvStream::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, getName() + "_FileStreamWrite.bin");
   deleteFile(fileManager, getName() + "_FileStreamWrite.txt");
   deleteFile(fileManager, getName() + "_FileStreamRead.bin");
   deleteFile(fileManager, getName() + "_FileStreamRead.bin");
}

} // namespace PV
