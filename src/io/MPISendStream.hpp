#ifndef MPISENDSTREAM_HPP_
#define MPISENDSTREAM_HPP_

#include "io/PrintStream.hpp"
#include <memory>
#include <mpi.h>
#include <sstream>

namespace PV {

/**
 * A PrintStream-derived class that sends strings over MPI. For example, probes that live on
 * non-root processes may not have access to the file system, but can set up an MPISendStream
 * during initialization. Calling the MPISendStream's printf or operator<< member function
 * adds the string to an internal stringstream object. The send() member function then sends it
 * over MPI to the designated rank with the designated tag.
 * The root process must separately implement a way to monitor for messages sent to it over MPI.
 * For example, it could set up an MPIRecvStream object and periodically call its receive() member
 * function. This is what BaseProbe-derived objects do in the outputStateWrapper() member function.
 * The MPISendStream object does not monitor whether the messages it sends have been received.
 */
class MPISendStream : public PrintStream {

  public:
   /**
    * The constructor for MPISendStream. Internally, it creates a stringstream object that the
    * PrintStream::printf and PrintStream::operator<<() function members use to hold the output.
    * See the receive() function member to send the data over MPI, and the MPIRecvStream class
    * for the complementary receiving capability.
    */
   MPISendStream(MPI_Comm mpi_comm, int receiveRank);
   ~MPISendStream();

   /**
    * Sends the buffer over MPI with the indicated tag, and clears the buffer.
    */
   int send(int tag);

  private:
   MPI_Comm mMPI_Comm;
   int mReceiveRank;
   std::stringstream mStringStream;

}; // class MPISendStream

} // namespace PV

#endif // MPISENDSTREAM_HPP_
