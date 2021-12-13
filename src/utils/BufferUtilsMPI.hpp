#ifndef __BUFFERUTILSMPI_HPP_
#define __BUFFERUTILSMPI_HPP_

#include "structures/Buffer.hpp"
#include "structures/MPIBlock.hpp"
#include "structures/SparseList.hpp"

namespace PV {
namespace BufferUtils {
/**
 * Scatters a buffer from the specified process of an MPIBlock to the processes
 * in the block with the given batch index. On entry, the root process buffer
 * should be the size of the global extended buffer, and contain the data to
 * be scattered; the nonroot processes should be a buffer the size of the local
 * extended buffer. localWidth and localHeight give the size of the local
 * restricted buffer.
 *
 * On exit, each process's buffer is the size of the local extended buffer,
 * filled with the data from the root process.
 */
template <typename T>
void scatter(
      std::shared_ptr<MPIBlock const> mpiBlock,
      Buffer<T> &buffer,
      unsigned int localWidth, // These values should be the
      unsigned int localHeight, // layer's local nx and ny
      int mpiBatchIndex,
      int sourceProcess);

/**
 * Gathers buffers from each process with the specified batch index into a
 * global buffer in the specified process of the block. On entry, each process
 * contains the local extended buffer. localWidth and localHeight give the
 * size of the local restricted buffer. On exit, the root process returns the
 * gathered data as a global extended buffer, and the non-root processes
 * return their input buffers.
 */
template <typename T>
Buffer<T> gather(
      std::shared_ptr<MPIBlock const> mpiBlock,
      Buffer<T> buffer,
      unsigned int localWidth,
      unsigned int localHeight,
      int mpiBatchIndex,
      int destProcess);

template <typename T>
SparseList<T> gatherSparse(
      std::shared_ptr<MPIBlock const> mpiBlock,
      SparseList<T> list,
      int mpiBatchIndex,
      int rootProcess);

} // end namespace BufferUtils

} // end namespace PV

#include "BufferUtilsMPI.tpp" // template implementations file

#endif
