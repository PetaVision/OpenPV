/*
 * Checkpointer.tpp
 *
 *  Created on Dec 16, 2016
 *      Author: Pete Schultz
 *  template implementations for TextOutput::print<Checkpointer::TimeInfo>.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/PrintStream.hpp"

namespace PV {

namespace TextOutput {

template <>
inline void
print(Checkpointer::TimeInfo const *dataPointer, size_t numValues, PrintStream &stream) {
   for (size_t n = 0; n < numValues; n++) {
      stream << "time = " << dataPointer[n].mSimTime << "\n";
      stream << "timestep = " << dataPointer[n].mCurrentCheckpointStep << "\n";
   }
} // print()

} // namespace TextOutput

} // namespace PV
