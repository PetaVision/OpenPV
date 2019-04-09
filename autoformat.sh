#!/bin/sh
cp clang-format .clang-format
find src tests -iname '*.h*' -o -iname '*.c*' -o -iname '*.kpp' -o -iname '*.tpp' |
xargs clang-format-3.8 -i -style=file
rm .clang-format
