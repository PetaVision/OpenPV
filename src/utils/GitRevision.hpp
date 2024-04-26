#ifndef GITREVISION_HPP_
#define GITREVISION_HPP_

namespace PV {

/**
 * Retrieves the git revision string generated from git-status at compile time.
 * The purpose of this function is to decouple GitRevisionString.hpp from the rest of the code:
 * A change to GitRevisionString.hpp will trigger recompiling the very small GitRevision.cpp
 * file but no other translation units.
 */
char const *getGitRevision();

} //namespace PV

#endif // GITREVISION_HPP_
