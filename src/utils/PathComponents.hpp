#ifndef PATHCOMPONENTS_HPP_
#define PATHCOMPONENTS_HPP_

#include <string>

namespace PV {

/** Returns the directory part of a path (everything before the final slash)
 *  The returned string is that specified by the POSIX dirname() function.
 *  The input argument is not modified.
 */
std::string dirName(std::string const &path);

/** Returns the directory part of a path (everything before the final slash)
 *  The returned string is that specified by the POSIX dirname() function.
 *  The input argument is not modified.
 */
std::string dirName(char const *path);

/** Returns the file part of a path (everything after the final slash)
 *  The returned string is that specified by the POSIX dirname() function.
 *  The input argument is not modified.
 */
std::string baseName(std::string const &path);

/** Returns the file part of a path (everything after the final slash)
 *  The returned string is that specified by the POSIX dirname() function.
 *  The input argument is not modified.
 */
std::string baseName(char const *path);

/** Returns the part of the basename from the final dot onward
 *  If the basename has no dot, the empty string is returned.
 *  The input argument is not modified.
 */
std::string extension(std::string const &path);

/** Returns the part of the basename up to the final dot
 *  If the basename has no dot, the entire basename is returned.
 *  The input argument is not modified.
 */
std::string stripExtension(std::string const &path);

} // namespace PV

#endif // PATHCOMPONENTS_HPP_
