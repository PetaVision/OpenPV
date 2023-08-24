/*
 * PvpListActivityBuffer.cpp
 *
 *  Created on: Aug 31, 2022
 *      Author: Pete Schultz
 */

#include "PvpListActivityBuffer.hpp"
#include <algorithm>

namespace PV {

PvpListActivityBuffer::PvpListActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PvpListActivityBuffer::~PvpListActivityBuffer() {}

void PvpListActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InputActivityBuffer::initialize(name, params, comm);
}

void PvpListActivityBuffer::setObjectType() { mObjectType = "PvpListActivityBuffer"; }

int PvpListActivityBuffer::countInputImages() {
   populateFileList();
   InfoLog() << "File " << getInputPath() << " contains " << mFileList.size() << " frames\n";
   return mFileList.size();
}

void PvpListActivityBuffer::populateFileList() {
   if (getCommunicator()->getIOMPIBlock()->getRank() == 0) {
      std::string line;
      mFileList.clear();
      InfoLog() << "Reading list: " << getInputPath() << "\n";
      std::ifstream infile(getInputPath(), std::ios_base::in);
      FatalIf(
            infile.fail(), "Unable to open \"%s\": %s\n", getInputPath().c_str(), strerror(errno));
      while (getline(infile, line, '\n')) {
         auto firstNonWhitespace = (std::string::size_type)0;
         while (firstNonWhitespace < line.size() and isspace(line[firstNonWhitespace])) {
            firstNonWhitespace++;
         }
         auto firstTrailingWhitespace = line.size();
         while (firstTrailingWhitespace > firstNonWhitespace
                and isspace(line[firstTrailingWhitespace - 1])) {
            firstTrailingWhitespace--;
         }
         if (firstTrailingWhitespace > firstNonWhitespace) {
            auto trimmedLength      = firstTrailingWhitespace - firstNonWhitespace;
            std::string trimmedLine = line.substr(firstNonWhitespace, trimmedLength);
            // Ignore any lines consisting of initial whitespace followed by '//'
            // Note: comments are only on lines to themselves. The line
            // "file.pvp // Description of the file" will add a file whose path ends with
            // " // Description of the file", which almost certainly isn't what you want.
            if (trimmedLine.length() >=2 and trimmedLine[0] == '/' and trimmedLine[1] == '/') {
               continue;
            }
            mFileList.push_back(trimmedLine);
         }
      }
      FatalIf(
            mFileList.empty(),
            "%s inputPath file list \"%s\" is empty.\n",
            getDescription_c(),
            getInputPath().c_str());
   }
}

std::string const &
PvpListActivityBuffer::getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const {
   int blockBatchIndex = localBatchIndex + getLayerLoc()->nbatch * mpiBatchIndex;
   int inputIndex      = mBatchIndexer->getIndex(blockBatchIndex);
   return mFileList.at(inputIndex);
}

std::string PvpListActivityBuffer::describeInput(int index) {
   std::string description = mFileList.at(index);
   return description;
}

Buffer<float> PvpListActivityBuffer::retrieveData(int inputIndex) {
   Buffer<float> result;
   std::string filename = mFileList.at(inputIndex);

   BufferUtils::readActivityFromPvp<float>(
         filename.c_str(), &result, 0 /*frame index*/, nullptr);
   return result;
}

} // namespace PV
