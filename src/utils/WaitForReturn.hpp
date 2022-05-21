#ifndef WAITFORRETURN_HPP_
#define WAITFORRETURN_HPP_

#include "arch/mpi/mpi.h"

namespace PV {
/**
 * All processes in the given MPI communicator block until the
 * root process receives a newline character on standard input.
 */
void WaitForReturn(MPI_Comm comm);

} // namespace PV

#endif // WAITFORRETURN_HPP_
