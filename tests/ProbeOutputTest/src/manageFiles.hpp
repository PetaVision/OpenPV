#ifndef MANAGEFILES_HPP_
#define MANAGEFILES_HPP_

#include <columns/Communicator.hpp>

#include <string>

using namespace PV;

int deleteLink(std::string const &path);
int deleteRegularFile(std::string const &path);
int recursiveDelete(std::string const &path);
int recursiveDelete(std::string const &path, Communicator *comm, bool warnIfAbsentFlag);
int recursiveDeleteDirectory(std::string const &path);
int renamePath(std::string const &oldpath, std::string const &newpath, Communicator *comm);

#endif // MANAGEFILES_HPP_
