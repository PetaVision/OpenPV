#include "MPIRecvStream.hpp"

#include <cerrno>
#include <cstring>
#include <sys/stat.h>

namespace PV {

MPIRecvStream::MPIRecvStream(
      std::string &path, MPI_Comm mpi_comm, int sendRank, bool clobberFlag) {
   // If clobberFlag is true or the file does not exist, we need to create the
   // file in write mode, and then close it, so that the file exists with
   // length 0 and we can open it in out|in mode ("r+" in FILE*-speak).
   // If clobberFlag is false and the file exists, we open in out|in mode as is.
   // We set the file position to zero; it is up to the caller to move the file
   // position if needed (e.g. restoring from checkpoint).
   bool createFile = clobberFlag;
   if (!clobberFlag) {
      struct stat existingstat;
      int status = stat(path.c_str(), &existingstat);
      FatalIf(
            status != 0 and errno != ENOENT,
            "MPIRecvStream unable to check status of \"%s\": %s\n",
            path.c_str(), std::strerror(errno));
      if (errno == ENOENT) {
         createFile = true;
      }
   }
   if (createFile) {
      std::ofstream emptyFile(path.c_str());
   } // file closes when we exit the scope of emptyFile.

   auto mode = std::ios_base::out | std::ios_base::in;
   mFileStream = new FileStream(path.c_str(), mode, false /*do not verify writes*/);
   mFileStream->setInPos(0L, std::ios_base::beg);
   mFileStream->setOutPos(0L, std::ios_base::beg);
   mMPI_Comm = mpi_comm;
   mSendRank = sendRank;
}

MPIRecvStream::~MPIRecvStream() {
   delete mFileStream;
}

int MPIRecvStream::receive(int tag) {
   int msgPresent;
   MPI_Status probeStatus;
   MPI_Iprobe(mSendRank, tag, mMPI_Comm, &msgPresent, &probeStatus);
   int count = 0;
   if (msgPresent) {
      MPI_Get_count(&probeStatus, MPI_CHAR, &count);
      std::string recvString(1, '\0'); // Make sure string is not empty since we're using front()
      if (count > 0) { recvString.resize(count, '\0'); }
      char *recvBuffer = &recvString.front();
      MPI_Recv(recvBuffer, count, MPI_CHAR, mSendRank, tag, mMPI_Comm, MPI_STATUS_IGNORE);
      (*mFileStream) << recvString;
      mFileStream->flush();
   }
   return count;
}

} // namespace PV
