#!/bin/sh
cp clang-format .clang-format
find src/ -iname *.h* -o -iname *.c* -o -iname *.tpp | xargs clang-format -i -style=file
find tests/ -iname *.h* -o -iname *.c* -o -iname *.tpp | xargs clang-format -i -style=file
rm .clang-format
