======================================================================
H3: Description of the files containing the H3 energy values of BKMP2:
======================================================================

The files described below contain the BKMP2 H3 energies used and described in:
A. I. Boothroyd, W. J. Keogh, P. G. Martin, and M. R. Peterson (1996),
"A Refined H3 Potential Energy Surface", J. Chem. Phys. 104, 7139-7152.

     Archive Name      Size                    Description
   ---------------- ---------- ------------------------------------------------
   cih3usen.tar.gz  (0.32 Mb): gzip-compressed tar-file archive containing the
                                 8 files 1 through 8 below (total size 1.4 Mb):
                                 all H3 ground state energies used in the BKMP2
                                 H3 surface fit.

   cih3ean.tar.gz    (1.0 Mb): gzip-compressed tar-file archive containing the
                                 6 files 8 through 13 below (total size 5 Mb):
                                 all our own H3 ab initio energies (including
                                 lowest 4 excited states, at some geometries).

To extract files requires the gzip utility to decompress the archive, and the
tar utility to extract the files.  For example, on a Unix or Linux system, to
extract all the files from an archive called ARCHIVENAME.tar.gz and delete it:
  gunzip -v ARCHIVENAME.tar.gz
  tar xvf ARCHIVENAME.tar
  rm ARCHIVENAME.tar

Contents of the archives:
-------------------------
Files 1-6 contain the energies fitted by the BKMP2 surface.  For files 1, 2, 3,
and 7, the energy value for any given geometry is generally a weighted average
of the various ab initio energies available for that geometry (with a common
basis set; note however that Liu's Slater-basis is combined with the [4s3p1d]
basis).  The corresponding files 9 through 12 (plus file 13 of test geometries)
contain all the energies used in these averages, plus (in files 11 and 13) some
energies corresponding to excited states.

       File Name    Size(bytes)                 Description
   ----------------- --------- ------------------------------------------------
1. cih3allpts.usen   (167029): 1235 geometries: the old BKMP grid, including
                                 the Slater-basis ab initio H3 energies of Liu
                                 1973, and the (9s3p1d)/[4s3p1d] ab initio
                                 energies of Liu 1973, Siegbahn and Liu 1978,
                                 Blomberg and Liu 1985, and BKMP 1991; plus
                                 some geometries added by BKMP2
2. cih3farpts.usen    (74594): 540 geometries: (9s3p1d)/[4s3p1d] ab initio
                                 energies of the preliminary grid of BKMP2
3. cih3newpts.usen   (873654): 6548 geometries: (9s3p2d)/[4s3p2d] ab initio
                                 energies of the comprehensive grid of BKMP2
4. cih3partriab.usen  (69673): 503 geometries: (11s5p3d1f)/[6s5p3d1f] ab initio
                                 energies of Partridge et al. 1993 (and of
                                 Partridge 1994)
5. cih3partrmtt.usen (131518): 968 geometries: energies obtained via the MTT
                                 fit to the ab initio energies of Partridge et
                                 al. 1993 to constrain the van der Waals region
6. cih3closlond.min   (26048): 175 geometries: low-accuracy very-short-distance
                                 energies obtained from the London equation, to
                                 constrain the extrapolation to short distances

7. cih3sp.usen         (6548): 28 geometries: (9s3p)/[4s3p] ab initio energies
                                 to test the basis correction; NOT USED IN FIT

8. h3ptsREADME.txt    (18469): This README file describing the H3 points.

9. cih3allpts.ean    (503785): 1235 geometries, 3767 energies: [4s3p1d]
10. cih3farpts.ean   (256538): 540 geometries, 1908 energies: [4s3p1d]
11. cih3newpts.ean  (4322610): 6548 geometries, 32480 energies: [4s3p2d]: with
                                 multiple MRD-CI roots (i.e., excited states)
                                 for 1809 of these geometries

12. cih3sp.ean        (10139): 28 geometries, 55 energies: [4s3p]

13. cih3varchk.ean   (137768): 174 geometries, 1015 energies: [4s3p2d]: some
                                 Varandas-style test geometries at short range
                                 near the conical intersection, with multiple
                                 MRD-CI roots (these excited states were used
                                 to show that the conical intersection is no
                                 longer an equilateral-triangle-H3 geometry at
                                 short enough distances)

Description of the format of the files:
---------------------------------------

Each file has a number of header lines giving a brief description of the
contents, followed by the data; except for some header lines, the line length
is 132 characters.  The first 72 characters of the last two header lines and
of the first four data lines in file 1 (cih3allpts.usen) are:

  Nabs ijk lmn   A/2       Y3       Z3       X4        Y4        Z4    S
 ----- -=-==-= -------- -------- -------- -------- --------- --------- -
 77006 404     0.868500 0.000000 2.645500                              t
 77016 404     0.878500 0.175407 2.626722                              t
 81853 204     0.600000 0.000000 2.500000                              P
 81864 205     0.600000 0.000000 2.600000                              P

and characters 73 through 132 of these six lines are:

 f(N) sumC*C    Eex   dEl34 dE(T)  DTS  |Ddc| DbasL  Efinal
 --- ------- -------- ----- ----- ----- ----- ----- --------
 .00 .999999-.1575456     0     0     0     0  1642-.1591876
 .00 .999999 -.157452     0     0     0     0  1643 -.159095
 .01 .985185 -.134380   -77   279    42   631  1750 -.136094
 .01 .984948 -.138062   -42   249    23   642  1757 -.139802

NOTE THAT THE LAST HEADER LINE MAY BE FOUND BY checking the first 6 characters
in each line; it is the only line whose first 6 characters are ' -----'.  The
data lines (following until the end of the file) contain the values:

 Nabs, ijk, lmn, A/2, Y3, Z3, X4, Y4, Z4, S, f(N), sumC*C, Eex, dEl34, dE(T),
   DTS, |Ddc|, DbasL, Efinal

in the FORMAT (I6,1X,A3,A4,4F9.6,2F10.6,1X,A1,F4.2,F8.6,F9.6,5I6,F9.6)

where the geometry (A/2, Y3, Z3) and the final energy (Efinal) are the most
important quantities; see the following description of the data lines:

     Nabs  (I6) is 5-digit integer, a unique identifier for each geometry; note
             that the same geometry may appear in several different files.
  ijk,lmn  (1X,A3,A4) are (non-unique) geometry indices, and can be ignored;
             for H3, ijk is a 3-digit integer, and lmn is blank.
      A/2  (F9.6) gives the position (in bohrs) of the first two H-atoms (in
             Cartesian coordinates): x1 = y1 = 0 , z1 = -A/2 , x2 = y2 = 0 ,
             z2 = A/2 ; note that A/2 is half the shortest interatomic distance
             r1 between atoms 1 and 2.
    Y3,Z3  (2F9.6) give the position (in bohrs) of the third H-atom: x3 = 0 ,
             y3 = Y3 , z3 = Z3 ; note that r2 (the distance between atoms 2
             and 3) is never longer than r3 (between atoms 1 and 3).
 X4,Y4,Z4  (F9.6,2F10.6); for H3, these are blank (as there is no fourth atom;
             they would be non-zero only for H4).
        S  (1X,A1) is a single-character code identifying the type of energy
             (see full description below of its possible meanings).
     f(N)  (F4.2) is the multiplier lambdaDC:H3 used to obtain the correction
             to full CI from the Davidson correction (see Efinal below).
   sumC*C  (F8.6) is the sum of the reference C-squared values in the
             multiple-reference CI calculation.
      Eex  (F9.6) is Elambda(T) (in hartrees), the energy extrapolated (using
             Buenker's method) to zero threshold from the truncated-CI energies
             E(T) and E(2T) [where only configurations contributing more than
             the threshold T (or 2T) were included in the truncated-CI
             calculations]; note that by definition Elambda(T) = Elambda(2T).
    dEl34  (I6) is Elambda(3T) + Elambda(4T) - 2 * Eex (in microhartrees), a
             measure of the energy extrapolation error at higher thresholds.
    dE(T)  (I6) is E(T) - Eex = E(T) - Elambda(T) (in microhartrees), the size
             of the extrapolation to zero threshold from the lowest-threshold
             truncated-CI energy.
      DTS  (I6) is a small further threshold correction, namely, -0.25 * dEl34
             (in microhartrees); for files 1, 2, 3, and 7, the value of DTS is
             the sum of two corrections: ( -0.25 * dEl34 ), plus the difference
             between the weighted-average Efinal value and the best of the
             individual Efinal values from corresponding files 9 through 12.
    |Ddc|  (I6) is the absolute value (in microhartrees) of the "bare" Davidson
             correction supplied by Buenker's MRD-CI program, namely,
             abs{ ( 1 - sumC*C ) * [ E(MRDCI) - Ereference ] } .
    DbasL  (I6) is the size (in microhartrees) of the modified London-type
             basis-set correction.
   Efinal  (F9.6) (in hartrees) is Eex + DTS - f(N) * |Ddc| / sumC*C - DbasL ,
             the best final corrected ab initio energy value (for files 1, 2,
             3, and 7, Efinal is the weighted-average); the BKMP2 surface was
             fitted to the Efinal values in files 1 through 6.

Note that for any given geometry in files 9 through 12, ALL the coordinates
A/2 through Z4 are left blank except for the first energy at that geometry;
the last energy value for each geometry is usually the best one.  Note that in
files 1, 2, 3, and 7, the quantities S through |Ddc| refer to the best of the
individual energy values from the corresponding files 9 through 12.  Note
that the quantites f(N) through |Ddc| (or their equivalents) were not in
general available for the energies supplied by other authors.

Meanings of the single-character type-code S (identifying the type of energy):
------------------------------------------------------------------------------

               for ab initio energies of other authors:
               ----------------------------------------
 "S" = H3 ab initio energies using Slater-basis (linear geometries only), from
         B. Liu (1973), J. Chem. Phys., 58, 1925.
 "e" = H3 ab initio energies using the (9s3p1d)/[4s3p1d] basis set, from
         P. Siegbahn and B. Liu (1978), J. Chem. Phys., 68, 2457.
 "m" = H3 ab initio energies using the (9s3p1d)/[4s3p1d] basis set, from
         M. R. A. Blomberg and B. Liu (1985), J. Chem. Phys. (Notes), 82, 1050.
 "a" = H3 ab initio energies from C. W. Bauschlicher, S. R. Langhoff, and
         H. Partridge (1990), Chem. Phys. Lett., 170, 345.  (Note that only a
         very approximate basis correction is available for these energies, as
         discussed in BKMP, and they were thus given very low weight in BKMP2.)
 "-" = H3 ab initio energies using the (11s5p3d1f)/[6s5p3d1f] basis set, from
          H. Partridge, C. W. Bauschlicher, J. R. Stallcop, and E. Levin
          (1993), J. Chem. Phys., 699, 5951.  Note that a few additional and
          corrected energy values were supplied by H. Partridge (1994), private
          communication; the H2-basis-correction for this basis was supplied by
          H. Partridge (1992), private communication.

               for energies generated to constrain the fit:
               --------------------------------------------
 "Z" = van der Waals region H3 energies computed from the modified Tang-Toenies
          (MTT) formulae fit to the data of Partridge et al. above (see BKMP2).
 "z" = the same as "Z", but with more extreme (large or small) values of r1.
 "t" = two H3 energy values generated from the saddle-point energy (and given
         high weight) to constrain the asymmetric-stretch and bending force
         constants, as was done by D. G. Truhlar and C. J. Horowitz (1978),
         J. Chem. Phys., 68, 2466; J. Chem. Phys. (Errata), 71, 1514 (1979).
          (Note that the first of these two energies requires seven decimal
         places of precision to set the force constant accurately enough, and
         thus has format F9.7 in the data file 1 [cih3allpts.usen].)
 "X" = very compact H3 energies generated from the London formula, to constrain
         extrapolation of the surface to short distances (i.e., high energies).

               for ab initio energies computed by BKMP and BKMP2:
               --------------------------------------------------
 "C", "L", "d", "c" = single-root MRD-CI energies with extrapolation threshold
                        T = 10.0, 2.0, 0.4, or 0.0 microhartrees, respectively,
                        with the molecular orbitals for the CI calculation
                        obtained from closed-shell SCF (2 closed shells)
 "o", "p", "P", "O" = molecular orbitals obtained from open-shell SCF (3 open
                        shells); otherwise same as above
 "b", "B", "h", "H" = molecular orbitals from mixed-shell SCF (1 closed shell,
                        1 open shell); otherwise same as above

 "D", "E", "l", "1" = the lowest root (ground state energy) from multiple-root
                        MRD-CI calculation (same T = 10.0, 2.0, 0.4, or 0.0
                        microhartrees, respectively), with molecular orbitals
                        obtained from closed-shell SCF
 "I", "J", "[", "2" = the second root (first excited state energy); otherwise
                        same as above
 "i", "j", "]", "3" = the third root; otherwise same as above
 "A", "<", "!", "4" = the fourth root; otherwise same as above
 "#", ">", "|", "5" = the fifth root; otherwise same as above

 "Q", "q", "0", "6" = the lowest root (ground state energy) from multiple-root
                        MRD-CI calculation (same T = 10.0, 2.0, 0.4, or 0.0
                        microhartrees, respectively), with molecular orbitals
                        obtained from open-shell SCF
 "N", "G", "(", "7" = the second root (first excited state energy); otherwise
                        same as above
 "r", "s", ")", "8" = the third root; otherwise same as above
 "@", "&", "+", "9" = the fourth root; otherwise same as above
 "$", "^", "=", "_" = the fifth root; otherwise same as above

 "M", "T", "x", ";" = the lowest root (ground state energy) from multiple-root
                        MRD-CI calculation (same T = 10.0, 2.0, 0.4, or 0.0
                        microhartrees, respectively), with molecular orbitals
                        obtained from mixed-shell SCF
 "V", "u", "{", "`" = the second root (first excited state energy); otherwise
                        same as above
 "v", "w", "}", "," = the third root; otherwise same as above
 "y", "*", "/", ":" = the fourth root; otherwise same as above
 "%", "?", "~", "." = the fifth root; otherwise same as above

Note that other type-codes (namely, "W", "K", "k", "F", "f", "Y", "U", "n",
and "R") are used to identify H4 energies of other authors; "g" is used to
identify H3 van der Waals energies generated from the old Gengenbach formula
(see BKMP), now superceded by the MTT formulae fit to Partridge energies (see
BKMP2 and above).  These type-codes are not found in any of the above files.

Note that, when an N-root MRD-CI calculation is performed, the choice of which
N roots will have lowest energy is only made using an approximate criterion;
thus when levels are closely spaced, the MRD-CI calculation may miss an energy
level that is actually slightly lower in energy than the Nth, or even (N-1)th,
reported root.  Similarly, the single-root MRD-CI calculation very occasionally
misses the ground state energy, yielding the first excited state instead; in
such a case, calculations with molecular orbitals from a different SCF-type, or
with multiple roots, has yielded the correct ground state energy for use in the
weighted average energies of files 1, 2, 3, and 7.

Description of weighted-average (to get files 1, 2, 3, 7 from files 9 - 12):
----------------------------------------------------------------------------

For geometries where several ground-state energy values were available, the
weighted average reported in files 1, 2, 3, and 7 as obtained as follows from
the individual energy values in files 9, 10, 11, and 12:

An energy cutoff was defined, as the minimum over all energy values of

  Ecut1 = min{ [ E(T) - DbasL ] , [ Efinal + 0.5 millihartree ] } ,

i.e., a cutoff at the lowest truncated-CI energy, at most 0.5 millihartree
above the lowest Efinal value.  If an first-excited-state energy E1 was present
and lay more than 0.3 millihartree above the corresponding ground-state energy
(i.e., a non-degenerate case), then another energy cutoff was defined, namely

  Ecut2 = Efinal + max{ min[ ( E1 - Efinal ) / 10 , 0.5 millihartree ] ,
                         min[ ( E1 - Efinal ) / 2 , 0.3 millihartree ] } .

Any Efinal values above the above the cutoff(s) were discarded.

The remaining Efinal values were presumed to have a sigma^2 uncertainty (note
"^2" means "to the power of 2") given by the following formula:

 Usq = Ufac * { [ dEl34 ]^2 + [ |Ddc| / sumC*C ]^2 + [ 0.5 * dE(T) ]^2
                   + [ 50 microhartrees ]^2 }

where Ufac was a factor of order unity that slightly increased the Usq value
of cases with higher threshold values T, and of single-root cases, and reduced
the uncertainty of Liu's energies; as a function of the type-code S (see above)
Ufac had the following values:

                        (open-shell)    (mixed-shell)   (closed-shell)
            (Liu)    T=10  2  0.4  0    10  2  0.4  0    10  2  0.4  0 microhar
         -----------  ---------------  ---------------  ---------------
type S = "S" "e" "m"  "o" "p" "P" "O"  "b" "B" "h" "H"  "C" "L" "d" "c"
  Ufac = 0.4 0.5 0.5  3.0 1.3 1.0 0.5  4.0 1.6 1.3 0.8  4.0 1.6 1.3 0.8

 multiple-root:   S = "Q" "q" "0" "6"  "M" "T" "x" ";"  "D" "E" "l" "1"
               Ufac = 2.0 1.0 .75 0.4  2.6 1.3 1.0 0.7  2.6 1.3 1.0 0.7

Each individual Efinal was then weighted by the inverse of its Usq value, to
yield the weighted average for file 1, 2, 3, or 7.
