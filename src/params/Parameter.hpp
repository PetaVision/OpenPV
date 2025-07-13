#ifndef PARAMETER_HPP_
#define PARAMETER_HPP_

#include <cassert>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace PV {

class Parameter {
  public:
   enum class Type { NotFound, Numeric, Array, String };
   Parameter(double value);
   Parameter(std::vector<double> const &arrayValues);
   Parameter(std::string const &stringValue);
   Parameter(char const *stringValue);

   bool operator==(Parameter const &rhs) const;

   void clearHasBeenReadFlag() { mHasBeenReadFlag = false; }

   template <typename T>
   T const *peek() const;

   template <typename T>
   T const *read() {
      T const *result = peek<T>();
      if (result) { mHasBeenReadFlag = true; }
      return result;
   }

   bool getHasBeenReadFlag() const { return mHasBeenReadFlag; }
   Type getType() const { return mType; }

  private:
   bool mHasBeenReadFlag = false;
   Type mType;
   std::vector<double> mValuesArray;
   std::string mValueString;

   static std::vector<double> const mDefaultArray;
   static double const mDefaultNumeric;
   static std::string const mDefaultString;
};

} // namespace PV

#endif // PARAMETER_HPP_
