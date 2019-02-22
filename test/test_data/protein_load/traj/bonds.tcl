set sel [atomselect 0 "index 0 1"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 1]
if { $id == -1 } {
lappend bond1 1
}
set id [lsearch -exact $bond2 0]
if { $id == -1 } {
lappend bond2 0
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 1 2"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 2]
if { $id == -1 } {
lappend bond1 2
}
set id [lsearch -exact $bond2 1]
if { $id == -1 } {
lappend bond2 1
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 1 3"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 3]
if { $id == -1 } {
lappend bond1 3
}
set id [lsearch -exact $bond2 1]
if { $id == -1 } {
lappend bond2 1
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 3 4"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 4]
if { $id == -1 } {
lappend bond1 4
}
set id [lsearch -exact $bond2 3]
if { $id == -1 } {
lappend bond2 3
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 3 5"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 5]
if { $id == -1 } {
lappend bond1 5
}
set id [lsearch -exact $bond2 3]
if { $id == -1 } {
lappend bond2 3
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 5 6"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 6]
if { $id == -1 } {
lappend bond1 6
}
set id [lsearch -exact $bond2 5]
if { $id == -1 } {
lappend bond2 5
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 5 7"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 7]
if { $id == -1 } {
lappend bond1 7
}
set id [lsearch -exact $bond2 5]
if { $id == -1 } {
lappend bond2 5
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 7 8"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 8]
if { $id == -1 } {
lappend bond1 8
}
set id [lsearch -exact $bond2 7]
if { $id == -1 } {
lappend bond2 7
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 7 9"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 9]
if { $id == -1 } {
lappend bond1 9
}
set id [lsearch -exact $bond2 7]
if { $id == -1 } {
lappend bond2 7
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 9 10"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 10]
if { $id == -1 } {
lappend bond1 10
}
set id [lsearch -exact $bond2 9]
if { $id == -1 } {
lappend bond2 9
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 9 11"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 11]
if { $id == -1 } {
lappend bond1 11
}
set id [lsearch -exact $bond2 9]
if { $id == -1 } {
lappend bond2 9
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 11 12"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 12]
if { $id == -1 } {
lappend bond1 12
}
set id [lsearch -exact $bond2 11]
if { $id == -1 } {
lappend bond2 11
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 12 13"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 13]
if { $id == -1 } {
lappend bond1 13
}
set id [lsearch -exact $bond2 12]
if { $id == -1 } {
lappend bond2 12
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 12 14"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 14]
if { $id == -1 } {
lappend bond1 14
}
set id [lsearch -exact $bond2 12]
if { $id == -1 } {
lappend bond2 12
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 14 15"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 15]
if { $id == -1 } {
lappend bond1 15
}
set id [lsearch -exact $bond2 14]
if { $id == -1 } {
lappend bond2 14
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 14 16"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 16]
if { $id == -1 } {
lappend bond1 16
}
set id [lsearch -exact $bond2 14]
if { $id == -1 } {
lappend bond2 14
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 16 17"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 17]
if { $id == -1 } {
lappend bond1 17
}
set id [lsearch -exact $bond2 16]
if { $id == -1 } {
lappend bond2 16
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 16 18"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 18]
if { $id == -1 } {
lappend bond1 18
}
set id [lsearch -exact $bond2 16]
if { $id == -1 } {
lappend bond2 16
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 18 19"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 19]
if { $id == -1 } {
lappend bond1 19
}
set id [lsearch -exact $bond2 18]
if { $id == -1 } {
lappend bond2 18
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 18 20"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 20]
if { $id == -1 } {
lappend bond1 20
}
set id [lsearch -exact $bond2 18]
if { $id == -1 } {
lappend bond2 18
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 20 21"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 21]
if { $id == -1 } {
lappend bond1 21
}
set id [lsearch -exact $bond2 20]
if { $id == -1 } {
lappend bond2 20
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 20 22"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 22]
if { $id == -1 } {
lappend bond1 22
}
set id [lsearch -exact $bond2 20]
if { $id == -1 } {
lappend bond2 20
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 22 23"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 23]
if { $id == -1 } {
lappend bond1 23
}
set id [lsearch -exact $bond2 22]
if { $id == -1 } {
lappend bond2 22
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 22 24"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 24]
if { $id == -1 } {
lappend bond1 24
}
set id [lsearch -exact $bond2 22]
if { $id == -1 } {
lappend bond2 22
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 24 25"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 25]
if { $id == -1 } {
lappend bond1 25
}
set id [lsearch -exact $bond2 24]
if { $id == -1 } {
lappend bond2 24
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 24 26"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 26]
if { $id == -1 } {
lappend bond1 26
}
set id [lsearch -exact $bond2 24]
if { $id == -1 } {
lappend bond2 24
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 26 27"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 27]
if { $id == -1 } {
lappend bond1 27
}
set id [lsearch -exact $bond2 26]
if { $id == -1 } {
lappend bond2 26
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 26 28"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 28]
if { $id == -1 } {
lappend bond1 28
}
set id [lsearch -exact $bond2 26]
if { $id == -1 } {
lappend bond2 26
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 28 29"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 29]
if { $id == -1 } {
lappend bond1 29
}
set id [lsearch -exact $bond2 28]
if { $id == -1 } {
lappend bond2 28
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 29 30"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 30]
if { $id == -1 } {
lappend bond1 30
}
set id [lsearch -exact $bond2 29]
if { $id == -1 } {
lappend bond2 29
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 29 31"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 31]
if { $id == -1 } {
lappend bond1 31
}
set id [lsearch -exact $bond2 29]
if { $id == -1 } {
lappend bond2 29
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 31 32"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 32]
if { $id == -1 } {
lappend bond1 32
}
set id [lsearch -exact $bond2 31]
if { $id == -1 } {
lappend bond2 31
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 31 33"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 33]
if { $id == -1 } {
lappend bond1 33
}
set id [lsearch -exact $bond2 31]
if { $id == -1 } {
lappend bond2 31
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 33 34"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 34]
if { $id == -1 } {
lappend bond1 34
}
set id [lsearch -exact $bond2 33]
if { $id == -1 } {
lappend bond2 33
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 33 35"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 35]
if { $id == -1 } {
lappend bond1 35
}
set id [lsearch -exact $bond2 33]
if { $id == -1 } {
lappend bond2 33
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 35 36"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 36]
if { $id == -1 } {
lappend bond1 36
}
set id [lsearch -exact $bond2 35]
if { $id == -1 } {
lappend bond2 35
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 35 37"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 37]
if { $id == -1 } {
lappend bond1 37
}
set id [lsearch -exact $bond2 35]
if { $id == -1 } {
lappend bond2 35
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 37 38"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 38]
if { $id == -1 } {
lappend bond1 38
}
set id [lsearch -exact $bond2 37]
if { $id == -1 } {
lappend bond2 37
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 37 39"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 39]
if { $id == -1 } {
lappend bond1 39
}
set id [lsearch -exact $bond2 37]
if { $id == -1 } {
lappend bond2 37
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 39 40"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 40]
if { $id == -1 } {
lappend bond1 40
}
set id [lsearch -exact $bond2 39]
if { $id == -1 } {
lappend bond2 39
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 39 41"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 41]
if { $id == -1 } {
lappend bond1 41
}
set id [lsearch -exact $bond2 39]
if { $id == -1 } {
lappend bond2 39
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 41 42"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 42]
if { $id == -1 } {
lappend bond1 42
}
set id [lsearch -exact $bond2 41]
if { $id == -1 } {
lappend bond2 41
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 41 43"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 43]
if { $id == -1 } {
lappend bond1 43
}
set id [lsearch -exact $bond2 41]
if { $id == -1 } {
lappend bond2 41
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 43 44"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 44]
if { $id == -1 } {
lappend bond1 44
}
set id [lsearch -exact $bond2 43]
if { $id == -1 } {
lappend bond2 43
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 43 45"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 45]
if { $id == -1 } {
lappend bond1 45
}
set id [lsearch -exact $bond2 43]
if { $id == -1 } {
lappend bond2 43
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 45 46"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 46]
if { $id == -1 } {
lappend bond1 46
}
set id [lsearch -exact $bond2 45]
if { $id == -1 } {
lappend bond2 45
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 45 47"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 47]
if { $id == -1 } {
lappend bond1 47
}
set id [lsearch -exact $bond2 45]
if { $id == -1 } {
lappend bond2 45
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 47 48"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 48]
if { $id == -1 } {
lappend bond1 48
}
set id [lsearch -exact $bond2 47]
if { $id == -1 } {
lappend bond2 47
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 47 49"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 49]
if { $id == -1 } {
lappend bond1 49
}
set id [lsearch -exact $bond2 47]
if { $id == -1 } {
lappend bond2 47
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 49 50"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 50]
if { $id == -1 } {
lappend bond1 50
}
set id [lsearch -exact $bond2 49]
if { $id == -1 } {
lappend bond2 49
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 49 51"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 51]
if { $id == -1 } {
lappend bond1 51
}
set id [lsearch -exact $bond2 49]
if { $id == -1 } {
lappend bond2 49
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 51 52"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 52]
if { $id == -1 } {
lappend bond1 52
}
set id [lsearch -exact $bond2 51]
if { $id == -1 } {
lappend bond2 51
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 51 53"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 53]
if { $id == -1 } {
lappend bond1 53
}
set id [lsearch -exact $bond2 51]
if { $id == -1 } {
lappend bond2 51
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 53 54"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 54]
if { $id == -1 } {
lappend bond1 54
}
set id [lsearch -exact $bond2 53]
if { $id == -1 } {
lappend bond2 53
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 53 55"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 55]
if { $id == -1 } {
lappend bond1 55
}
set id [lsearch -exact $bond2 53]
if { $id == -1 } {
lappend bond2 53
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 55 56"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 56]
if { $id == -1 } {
lappend bond1 56
}
set id [lsearch -exact $bond2 55]
if { $id == -1 } {
lappend bond2 55
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 55 57"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 57]
if { $id == -1 } {
lappend bond1 57
}
set id [lsearch -exact $bond2 55]
if { $id == -1 } {
lappend bond2 55
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 57 58"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 58]
if { $id == -1 } {
lappend bond1 58
}
set id [lsearch -exact $bond2 57]
if { $id == -1 } {
lappend bond2 57
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 57 59"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 59]
if { $id == -1 } {
lappend bond1 59
}
set id [lsearch -exact $bond2 57]
if { $id == -1 } {
lappend bond2 57
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 59 60"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 60]
if { $id == -1 } {
lappend bond1 60
}
set id [lsearch -exact $bond2 59]
if { $id == -1 } {
lappend bond2 59
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 59 61"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 61]
if { $id == -1 } {
lappend bond1 61
}
set id [lsearch -exact $bond2 59]
if { $id == -1 } {
lappend bond2 59
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 61 62"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 62]
if { $id == -1 } {
lappend bond1 62
}
set id [lsearch -exact $bond2 61]
if { $id == -1 } {
lappend bond2 61
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 61 63"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 63]
if { $id == -1 } {
lappend bond1 63
}
set id [lsearch -exact $bond2 61]
if { $id == -1 } {
lappend bond2 61
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 63 64"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 64]
if { $id == -1 } {
lappend bond1 64
}
set id [lsearch -exact $bond2 63]
if { $id == -1 } {
lappend bond2 63
}
$sel setbonds [list $bond1 $bond2]
$sel delete
set sel [atomselect 0 "index 63 65"]
lassign [$sel getbonds] bond1 bond2
set id [lsearch -exact $bond1 65]
if { $id == -1 } {
lappend bond1 65
}
set id [lsearch -exact $bond2 63]
if { $id == -1 } {
lappend bond2 63
}
$sel setbonds [list $bond1 $bond2]
$sel delete
