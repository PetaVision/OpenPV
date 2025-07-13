#include "Parameter.hpp"
#include <cstdio>

namespace PV {

Parameter::Parameter(double value) {
   mType = Type::Numeric;
   mValuesArray = std::vector<double>{value};
}

Parameter::Parameter(std::vector<double> const &value) {
   mType = value.size() == 1UL ? Type::Numeric : Type::Array;
   mValuesArray = value;
   std::size_t N = value.size();
}

Parameter::Parameter(std::string const &stringValue) {
   mType = Type::String;
   mValueString = stringValue;
}

Parameter::Parameter(char const *stringValue) {
   mType = Type::String;
   mValueString = stringValue;
}

template <>
double const *Parameter::peek<double>() const {
   switch(mType) {
      case Type::Numeric:
      case Type::Array: // fall-through is intentional
         if (mValuesArray.size() == 1UL) {
            return &mValuesArray[0];
         }
         else {
            return nullptr;
         }
      case Type::String:
         return nullptr;
      default:
         // This shouldn't ever happen
         std::fprintf(
               stderr,
               "%s:%d Parameter::peek() with unknown parameter type %d\n",
               __FILE__, __LINE__, static_cast<int>(mType));
         std::exit(EXIT_FAILURE);
   }
}

template <>
std::vector<double> const *Parameter::peek<std::vector<double>>() const {
    switch(mType) {
       case Type::Numeric:
       case Type::Array: // fallthrough is intentional
          return &mValuesArray;
       case Type::String:
          return nullptr;
       default:
         // This shouldn't ever happen
         std::fprintf(
               stderr,
               "%s:%d Parameter::peek() with unknown parameter type %d\n",
               __FILE__, __LINE__, static_cast<int>(mType));
         std::exit(EXIT_FAILURE);
    }
}

template <>
std::string const *Parameter::peek<std::string>() const {
    switch(mType) {
       case Type::Numeric: // fallthrough is intentional
       case Type::Array:
          return nullptr;
      case Type::String:
         return &mValueString;
         break;
      default:
         // This shouldn't ever happen
         std::fprintf(
               stderr,
               "%s:%d Parameter::peek() with unknown parameter type %d\n",
               __FILE__, __LINE__, static_cast<int>(mType));
         std::exit(EXIT_FAILURE);
    }
}

bool Parameter::operator==(Parameter const &rhs) const {
   auto type1 = getType();
   auto type2 = rhs.getType();
   if (type1 != type2) { return false; }
   switch(type1) {
      case Type::Numeric:
         assert(mValuesArray.size() == 1UL);
         assert(rhs.mValuesArray.size() == 1UL);
         return mValuesArray[0] == rhs.mValuesArray[0];
      case Type::Array:
         return mValuesArray == rhs.mValuesArray;
      case Type::String:
         return mValueString == rhs.mValueString;
      default:
         // This shouldn't ever happen
         std::fprintf(
               stderr,
               "%s:%d Parameter::operator==() with unknown parameter type %d\n",
               __FILE__, __LINE__, static_cast<int>(mType));
         std::exit(EXIT_FAILURE);
   }
}

std::vector<double> const Parameter::mDefaultArray = {};
double const Parameter::mDefaultNumeric            = 0.0;
std::string const Parameter::mDefaultString        = "";

} // namespace PV
