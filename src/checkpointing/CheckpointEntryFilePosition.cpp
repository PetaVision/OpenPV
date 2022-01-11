#include "CheckpointEntryFilePosition.hpp"

namespace PV {

void CheckpointEntryFilePosition::write(
      std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag) const {
   if (getMPIBlock()->getRank() != 0) { return; }
   std::string path;

   writeValueToBinAndTxt(
         checkpointDirectory,
         std::string("FileStreamRead"),
         mFileStream->getInPos(),
         verifyWritesFlag);

   writeValueToBinAndTxt(
         checkpointDirectory,
         std::string("FileStreamWrite"),
         mFileStream->getOutPos(),
         verifyWritesFlag);
}

void CheckpointEntryFilePosition::read(
      std::string const &checkpointDirectory, double *simTimePtr) const {
   if (getMPIBlock()->getRank() != 0) { return; }

   long readPosition;
   readValueFromBin(
         checkpointDirectory, std::string("FileStreamRead"), &readPosition);
   mFileStream->setInPos(readPosition, std::ios_base::beg);

   long writePosition;
   readValueFromBin(
         checkpointDirectory, std::string("FileStreamWrite"), &writePosition);
   mFileStream->setInPos(writePosition, std::ios_base::beg);
}

void CheckpointEntryFilePosition::remove(std::string const &checkpointDirectory) const {
   if (getMPIBlock()->getRank() != 0) { return; }

}

void CheckpointEntryFilePosition::readValueFromBin(
      std::string const &checkpointDirectory,
      std::string const &label,
      long *value) const {
   std::string pathBase(checkpointDirectory);
   pathBase.append("/").append(getName()).append("_").append(label);
   std::string path;
   path = pathBase + ".bin";
   FileStream binFileStream(path.c_str(), std::ios_base::in);
   binFileStream.read(value, sizeof(*value));
}

void CheckpointEntryFilePosition::writeValueToBinAndTxt(
      std::string const &checkpointDirectory,
      std::string const &label,
      long value,
      bool verifyWritesFlag) const {
   std::string pathBase(checkpointDirectory);
   pathBase.append("/").append(getName()).append("_").append(label);
   std::string path;
   path = pathBase + ".bin";
   FileStream binFileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
   binFileStream.write(&value, sizeof(value));
   path = pathBase + ".txt";
   FileStream txtFileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
   txtFileStream << value << "\n";
}

} // namespace PV
