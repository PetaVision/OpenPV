#!/usr/bin/env bash
#
# This script monitors the inputImages folder for files and prints the paths
# to the files as they are detected.
#
# It was written to be used on a remote machine while upload_to_aws.bash
# (in the demo-localsize.tgz archive) is running on a local machine;
# the output can then be piped to the PASCAL_Classification executable.
#
# The script watches for a file inputImages/.uploadedfile, and
# when such a file is noticed, it prints the contents of .uploadedfile
# and then deletes .uploadedfile.  It then watches for
# inputImages/.uploadedfile again, and repeats the process.
#
# If the file inputImages/.uploadfinished appears, the script breaks out
# of the .uploadedfile loop, deletes the .uploadfinished file, and exits.
#
# The assumption behind this script is that the local side would upload
# an image into the inputImages folder, and on completion, write the
# path to the uploaded file into inputImages/.uploadedfile.
# The local script would then watch for .uploadedfile to be deleted before
# uploading the next file.  When the last file had been uploaded and
# the corresponding .uploadedfile deleted, the local script would create
# the .uploadfinished file and then exit.
#
# The reason for the .uploadedfile file is so that monitorInputImagesDir.bash
# does not report the file until the file has finished uploading, so the
# local side should wait for uploading to finish before creating this file.
#
until test -f inputImages/.uploadfinished || test -f inputImages/.uploadedfile
do
    sleep 1
    if test -f inputImages/.uploadedfile
    then
        cat inputImages/.uploadedfile
        rm inputImages/.uploadedfile
    fi
done
rm inputImages/.uploadfinished
