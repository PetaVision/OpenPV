#! /usr/bin/env bash
# Usage: ChangeParamsGroupName oldname newname filename
#
# Checks if the string "newname" (including double quotes) appears in the file
# and if not, changes appearances of the string "oldname" to "newname" (where
# the quotes are part of the string being searched for, but not part of the
# argument to the command).
#
# The script runs sed, editing filename in place with the suffix '.bak',
# and then deletes the .bak file.
#
# The motivation is to quickly change the name of a layer or connection in
# all its occurrences in a params file.

progname="$(dirname $0)"
usage="Usage: $progname oldname newname filename
oldname can only contain letters, numbers, underscores or spaces.
newname can only contain letters, numbers, or underscores and must start with
        a letter or a number.
Neither oldname nor newname can be the empty string.
Only a single filename can be processed."

if test $# -eq 0
then
    echo "$usage"
    exit 0
fi

if test $# -ne 3
then
    >&2 echo "$usage" 
    exit 1
fi

oldname="$1"
newname="$2"
filename="$3"

cleanold="${oldname//[^0-9A-Za-z_ ]/}"
cleannew="${newname//[^0-9A-Za-z_]/}"
if test "$oldname" != "$cleanold"
then
   >&2 echo "$progname: oldname argument has an invalid character."
   exit 1
fi

if test "$newname" != "$cleannew"
then
   >&2 echo "$progname: newname argument has an invalid character."
   exit 1
fi

if test "$(echo -n "$oldname" | wc -c)" -eq 0
then
    >&2 echo "$progname: oldname argument cannot be empty"
    exit 1
fi

if test "$(echo -n "$newname" | wc -c)" -eq 0
then
    >&2 echo "$progname: newname argument cannot be empty"
    exit 1
fi

if ! test -f "$filename"
then
    >&2 echo "$progname: filename must exist as a regular file (or a symlink to a regular file)."
    exit 1
fi

if test -n "$(echo "$filename" | fgrep \""$newname"\")"
then
    >&2 echo "$progname: newname argument already exists in the file."
    exit 1
fi

if test -z "$(fgrep \""$oldname"\" "$filename")"
then
    >&2 echo "$progname: oldname argument does not exist in the file."
    exit 1
fi

sed -e '1,$s/"'"$oldname"'"/"'"$newname"'"/g' -i.bak "$filename"
rm "$filename.bak"
