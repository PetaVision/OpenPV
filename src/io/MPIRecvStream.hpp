#ifndef MPIRECVSTREAM_HPP_
#define MPIRECVSTREAM_HPP_

#include "io/FileStream.hpp"
#include <fstream>
#include <memory>
#include <mpi.h>

namespace PV {

/**
 * A class that receives strings sent over MPI and prints them to a specified output stream.
 * The receive() function member uses a nonblocking MPI_Iprobe call to see if there is a message
 * with the given rank and tag, and if there is one, performs a blocking MPI_Recv call to
 * retrieve the data, which it then prints to the output stream specified in the constructor.
 * Returns the number of bytes printed (0 if there was no message sent).
 */
class MPIRecvStream {

  public:
   /**
    * The constructor for MPIRecvStream. Internally, it uses the path to create
    * FileStream object that the receive() function member uses to print the received data.
    */
   MPIRecvStream(std::string &path, MPI_Comm mpi_comm, int sendRank);
   ~MPIRecvStream();

   /**
    * Checks whether there is an MPI message with the given tag. If not, returns zero.
    * If there is a message, receives it with MPI_Recv, and prints it as a string to the
    * given output stream. It returns the number of characters received.
    * (The return value does not distinguish between there being no message and there being
    * a message with count zero.)
    */
   int receive(int tag);

   // Get-methods and set-methods that are passed on to the FileStream
   long getInPos() const { return mFileStream->getInPos(); }
   long getOutPos() const { return mFileStream->getOutPos(); }
   void setInPos(long p) { mFileStream->setInPos(p, std::ios_base::beg); }
   void setOutPos(long p) { mFileStream->setOutPos(p, std::ios_base::beg); }

  private:
   FileStream *mFileStream = nullptr;
   MPI_Comm mMPI_Comm;
   int mSendRank;

}; // class MPIRecvStream

} // namespace PV

#endif // MPIRECVSTREAM_HPP_
