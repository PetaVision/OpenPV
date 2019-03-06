/*
 * expandLeadingTilde.hpp
 *
 *  Created on: Mar 6, 2019
 *      Author: peteschultz
 */

#ifndef EXPANDLEADINGTILDE_HPP_
#define EXPANDLEADINGTILDE_HPP_

#include <string>

namespace PV {

/** If a filename begins with "~/" or is "~", presume the user means the home directory.
 * The return value is the expanded path; e.g. if the home directory is /home/user1,
 * calling with the path "~/directory/file.name" returns "/home/user1/directory/file.name"
 * If the input filename is different from "~" and doesn't start with "~/", the return
 * value has the same contents as input (but is a different block of memory).
 * The calling routine has the responsibility for freeing the return value, and
 * if the input string needs to be free()'ed, the calling routine has that responsibility
 * as well.
 *
 * There is no checking whether the resulting path exists, or is writable, or anything else.
 */
std::string expandLeadingTilde(std::string const &path);

/** An overload of expandLeadingTilde to take a C-style string instead of a C++ std::string.
 */
std::string expandLeadingTilde(char const *path);

/**
 * Returns a string containing the value of the HOME environment variable;
 * returns the empty string if undefined.
 */
std::string const getHomeDirectory();

} // end namespace PV

#endif // EXPANDLEADINGTILDE_HPP_
