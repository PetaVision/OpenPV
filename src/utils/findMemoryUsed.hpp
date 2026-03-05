#ifndef FINDMEMORYUSEDD_HPP_
#define FINDMEMORYUSEDD_HPP_

namespace PV {

/**
 * Returns the amount of resident memory currently used by the processs, in bytes.
 * Returns -1 if unable to find a value.
 */
long int findMemoryUsed();

} // namespace PV

#endif // FINDMEMORYUSEDD_HPP_
