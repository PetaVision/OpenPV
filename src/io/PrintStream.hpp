#ifndef PRINTSTREAM_HPP_
#define PRINTSTREAM_HPP_

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
      int chars_needed = vsnprintf(nullptr, 0, fmt, args1) + 1; // +1 for null terminator
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
   PrintStream &operator<<(std::string &s) {
      (*mOutStream) << s;
      return *this;
   }
   PrintStream &operator<<(char c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(signed char c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(unsigned char c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(const char *c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(const signed char *c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(const unsigned char *c) { (*mOutStream) << c; return *this; }
   PrintStream &operator<<(short x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(unsigned short x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(int x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(unsigned int x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(long x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(unsigned long x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(long long x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(unsigned long long x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(float x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(double x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(long double x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(bool x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(void const *x) { (*mOutStream) << x; return *this; }
   PrintStream &operator<<(std::streambuf *x) { (*mOutStream) << x; return *this; }
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

#endif // PRINTSTREAM_HPP_
