#ifndef PATHCOMPONENTS_HPP_
#define PATHCOMPONENTS_HPP_

#include <string>

namespace PV {

/** Returns the directory part of a path (everything before the final slash) */
std::string dirName(std::string const &path);

/** Returns the directory part of a path (everything before the final slash) */
std::string dirName(char const *path);

/** Returns the file part of a path (everything after the final slash) */
std::string baseName(std::string const &path);

/** Returns the file part of a path (everything after the final slash) */
std::string baseName(char const *path);

/** Returns the extension of a filename  (everything after the final dot)*/
std::string extension(std::string const &path);

/** Returns the extension of a filename (everything after the final dot) */
std::string extension(char const *path);

/** Returns the part of a filename from the final dot on (empty if there is no dot) */
std::string stripExtension(std::string const &path);

/** Returns the part of a filename from the final dot on (empty if there is no dot) */
std::string stripExtension(char const *path);

} // namespace PV

#endif // PATHCOMPONENTS_HPP_
