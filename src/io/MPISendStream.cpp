#include "MPISendStream.hpp"

namespace PV {

MPISendStream::MPISendStream(MPI_Comm mpi_comm, int receiveRank) {
   mMPI_Comm     = mpi_comm;
   mReceiveRank  = receiveRank;
   mStringStream = std::stringstream();
   initialize(mStringStream);
}

MPISendStream::~MPISendStream() {}

int MPISendStream::send(int tag) {
   int count = 0;
   auto zeropos = static_cast<std::stringstream::pos_type>(0);
   auto position = mStringStream.tellp();
   if (position != zeropos) {
      std::string sendString(position, '\0');
      mStringStream.read(&sendString.front(), position);
      char const *buffer = sendString.c_str();
      count = static_cast<int>(sendString.size());
      MPI_Send(buffer, count, MPI_CHAR, mReceiveRank, tag, mMPI_Comm);
   }
   mStringStream.seekg(zeropos);
   mStringStream.seekp(zeropos);
   return count;
}

} // namespace PV
