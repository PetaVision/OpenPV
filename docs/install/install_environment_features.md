To build on a 64 bit architecture:
   1. Edit src/include/pv_arch.h
      a. Replace the line: #undef PV_ARCH_64 with: #define PV_ARCH_64

The default settings work for building using Eclipse.  To build and run
using the standard make files in the src directory, it will be convenient to:
   1. Edit src/include/pv_arch.h
      a. Replace the line: #define ECLIPSE with: #undef ECLIPSE