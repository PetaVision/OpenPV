#ifndef __PRINTSTREAM_HPP_
#define __PRINTSTREAM_HPP_

extern "C" {
#include <unistd.h>
}

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "utils/PVAssert.hpp"

namespace PV {

class PrintStream {
  public:
   PrintStream(std::ostream &stream) { initialize(stream); }
   virtual ~PrintStream() {}

   int printf(const char *fmt, ...) {
      va_list args1, args2;
      va_start(args1, fmt);
      va_copy(args2, args1);
      char c;
      int chars_needed = vsnprintf(&c, 1, fmt, args1) + 1; // +1 for null terminator
      char output_string[chars_needed];
#ifdef NDEBUG
      vsnprintf(output_string, chars_needed, fmt, args2);
#else
      int chars_printed = vsnprintf(output_string, chars_needed, fmt, args2) + 1;
      pvAssert(chars_printed == chars_needed);
#endif // NDEBUG
      (*mOutStream) << std::string(output_string);
      va_end(args1);
      va_end(args2);
      return chars_needed;
   }

   void flush() { mOutStream->flush(); }

   // Operator overloads to allow using << like cout
   template <typename T>
   PrintStream &operator<<(const T &x) {
      (*mOutStream) << x;
      return *this;
   }
   PrintStream &operator<<(std::ostream &(*f)(std::ostream &)) {
      f(*mOutStream);
      return *this;
   }
   PrintStream &operator<<(std::ostream &(*f)(std::ios &)) {
      f(*mOutStream);
      return *this;
   }
   PrintStream &operator<<(std::ostream &(*f)(std::ios_base &)) {
      f(*mOutStream);
      return *this;
   }

  protected:
   PrintStream() {}
   void initialize(std::ostream &stream) { mOutStream = &stream; }

  private:
   std::ostream *mOutStream = nullptr;
};
} /* namespace PV */

#endif
