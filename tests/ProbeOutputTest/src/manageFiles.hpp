#ifndef MANAGEFILES_HPP_
#define MANAGEFILES_HPP_

#include <columns/Communicator.hpp>

#include <filesystem>
#include <string>

using namespace PV;

int appendExtraneousData(std::filesystem::path const &path, Communicator *comm);
int recursiveCopy(std::string const &from, std::string const &to, Communicator *comm);
int recursiveDelete(std::string const &path, Communicator *comm, bool warnIfAbsentFlag);
int renamePath(std::string const &oldpath, std::string const &newpath, Communicator *comm);

#endif // MANAGEFILES_HPP_
