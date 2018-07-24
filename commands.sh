This documents serves as refernce for useful commands that are used in while am doing my thesis.

#moving or copying the files from one folder randmoly to another folder

shuf -zen200 source/* | xargs -0 mv -t dest

#If you don't have shuf but have a sort that takes -R, this should work

find source -type f -print0 | sort -Rz | cut -d $'\0' -f-200 | xargs -0 mv -t dest
