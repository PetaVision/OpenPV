#include "MPIRecvStream.hpp"

#include <cerrno>
#include <cstring>
#include <sys/stat.h>

namespace PV {

MPIRecvStream::MPIRecvStream(
      std::string &path, MPI_Comm mpi_comm, int sendRank) {
   // We don't want to clobber an existing file, in case we are restoring from checkpoint.
   // If we are, we need to be able to move the file pointer to earlier in the file.
   // Therefore, the open mode must be out|in ("r+" in FILE*-speak), but that means the file
   // must exist when we open it. Check if file exists; if not open it with mode "w" to
   // create an empty file and immediately close it. Then, open in read/write/no-append mode.
   // We set the output position to zero; if we are indeed restoring from checkpoint, the
   // CheckpointEntryMPIRecvStream object will move the output pointer to the correct position.
   struct stat existingstat;
   int status = stat(path.c_str(), &existingstat);
   FatalIf(
         status != 0 and errno != ENOENT,
         "Unable to check status of \"%s\": %s\n",
         path.c_str(), std::strerror(errno));
   if (errno == ENOENT) {
      std::ofstream emptyFile(path.c_str());
   }
   auto mode = std::ios_base::out | std::ios_base::in;
   mFileStream = new FileStream(path.c_str(), mode, false /*do not verify writes*/);
   mFileStream->setOutPos(0L, std::ios_base::beg);
   mMPI_Comm    = mpi_comm;
   mSendRank    = sendRank;
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
