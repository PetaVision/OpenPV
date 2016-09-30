#!/bin/sh
find src/ -iname *.h* -o -iname *.c* | xargs clang-format -i -style=file
find tests/ -iname *.h* -o -iname *.c* | xargs clang-format -i -style=file
