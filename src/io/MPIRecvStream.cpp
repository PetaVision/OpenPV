#include "MPIRecvStream.hpp"

namespace PV {

MPIRecvStream::MPIRecvStream(
      std::string &path, MPI_Comm mpi_comm, int sendRank) {
   mFile        = new std::ofstream(path);
   mPrintStream = new PrintStream(*mFile);
   mMPI_Comm    = mpi_comm;
   mSendRank    = sendRank;
}

MPIRecvStream::~MPIRecvStream() {
   delete mPrintStream;
   delete mFile;
}

int MPIRecvStream::receive(int tag) {
   int msgPresent;
   MPI_Status probeStatus;
   MPI_Iprobe(mSendRank, tag, mMPI_Comm, &msgPresent, &probeStatus);
   int count = 0;
   if (msgPresent) {
      MPI_Get_count(&probeStatus, MPI_Datatype MPI_CHAR, &count);
      std::string recvString(1, '\0'); // Make sure string is not empty since we're using front()
      if (count > 0) { recvString.resize(count, '\0'); }
      char *recvBuffer = &recvString.front();
      MPI_Recv(recvBuffer, count, MPI_CHAR, mSendRank, tag, mMPI_Comm, MPI_STATUS_IGNORE);
      (*mPrintStream) << recvString;
   }
   return count;
}

} // namespace PV
