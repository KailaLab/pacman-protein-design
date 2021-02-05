molecule load pdb {pdb}
set everyone [atomselect top all]
set mm [measure minmax $everyone]
set ce [measure center $everyone]
set of [open {outfile} w]
puts $of $mm
puts $of $ce
close $of
quit

