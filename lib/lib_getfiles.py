# -*- coding: utf-8 -*-
# !/usr/bin/env python
# coding: utf-8

'''
#=============================================================================
#   lib_getfiles.py
#-----------------------------------------------------------------------------
    Library of Functions for processing data 
    @author: Daniel.Collins <Daniel.Collins@Wisk.Aero>
    Revised: 2023-09-29-T15:27-0700

    Created: Mo Apr 18 2022 1556 T
    
#-----------------------------------------------------------------------------
'''

## ======================================================================== ##
######   import packages  /////////////////////////////////////////
import os;
from   os import listdir;
import string ;
import math ;
import re ;
import linecache ;
import argparse ;
from   datetime import datetime ;
import statistics ;
import pandas as pd ;
import numpy as np ;
import itertools ;

######   import packages - custom lib /////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
## ======================================================================== ##

#-----------------------------------------------------------------------------
SETDEF_PATH  = r'G:\\' ;
KWBEG = '';
#########[  0  1  2  3  4  5  6  7  8  9 ]##---------------------------------
SHOW      =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # toggle: sys prints
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];


#-----------------------------------------------------------------------------
# Function : getFigname
#-----------------------------------------------------------------------------
def getFigname(ipic, outpath):
    if ipic < 1000:
        jpic = '0' + str(ipic)
    if ipic < 100:
        jpic = '00' + str(ipic)
    if ipic < 10:
        jpic = '000' + str(ipic)
    figname = outpath + jpic + '.png'
    ipic += 1
    print('saving figure as: {0} '.format(figname))
    return (ipic, figname)

#----------------------------------------------------------

#-----------------------------------------------------------------------------
# Function : getFiles
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



def getFile_begins(inppath, begins):
    filepaths = [inppath + f \
              for f in listdir(inppath) \
                if f.endswith('.csv') and f.startswith(begins) ];
    return filepaths;
#-----------------------------------------------------------------------------

def getFiles_last(inppath, last_nfiles, begins ):
    '''
    Selectfiles=  getFiles(inppath, last_nfiles)

    '''
    nfdef = 1;
    filepaths = [inppath + f \
                 for f in listdir(inppath) \
                   if f.endswith('.csv') and f.startswith(begins) ]
    NFILES = len(filepaths)
    # Selectfiles = filepaths[NFILES-nfdef:NFILES]
    Selectfiles = filepaths[ -last_nfiles : ]
    print(Selectfiles)
    return Selectfiles
#-----------------------------------------------------------------------------

def getFiles_first(inppath, first_nfiles, begins ):
    '''
    Selectfiles=  getFiles(inppath, last_nfiles)

    '''
    nfdef = 1
    ## archetype filename:
    '  ES-612s-0#-2022-03-18-12-06.csv '
    # begins='ES-612s-0#'
    filepaths = [inppath + f \
                 for f in listdir(inppath) \
                   if f.endswith('.csv') and f.startswith(begins) ]
    NFILES = len(filepaths)
    # Selectfiles = filepaths[ -last_nfiles : ]
    Selectfiles = filepaths[0:first_nfiles]
    print(Selectfiles)
    return Selectfiles
#-----------------------------------------------------------------------------

def getFiles_all(inppath, begins):
    '''
    Selectfiles=  getFiles(inppath )

    '''

    ## archetype filename:
    '  ES-612s-0#-2022-03-18-12-06.csv '
    # begins='ES-612s-0#'
    filepaths = [inppath + f \
                 for f in listdir(inppath) \
                   if f.endswith('.csv') and f.startswith(begins) ]
    NFILES = len(filepaths)
    # Selectfiles = filepaths[NFILES-nfdef:NFILES]
    Selectfiles = filepaths[ : ]
    print(Selectfiles)
    return Selectfiles
#-----------------------------------------------------------------------------


def getFiles0(inppath):
    nfdef = 1
    filepaths = [inppath + f for f in listdir(inppath) if f.endswith('.csv')]
    NFILES = len(filepaths)
    Selectfiles = filepaths[NFILES-nfdef:NFILES]
    print(Selectfiles)
    return Selectfiles
#-----------------------------------------------------------------------------

def getFiles(inppath):
    nfdef = 1

    # '  ES-612s-0#-2022-03-18-12-06.csv '
#    filepaths = [inppath + f for f in listdir(inppath) if f.endswith('.csv')]
    beginstr=[ 'ES-612s-0#'] ; ## archetype filename
    filepaths = [inppath + f \
                 for f in listdir(inppath) \
                   if f.endswith('.csv') and f.startswith(beginstr[0]) ];
    NFILES = len(filepaths);
    Selectfiles = filepaths[NFILES-nfdef:NFILES];
    print(Selectfiles);
    return Selectfiles;
#-----------------------------------------------------------------------------


def getFileFav(inppath, favfile):
    filepaths = [inppath + f \
              for f in listdir(inppath) \
                if f.endswith('.csv') and f.startswith(favfile) ];
    return filepaths;
#-----------------------------------------------------------------------------

def getFilesSel(inppath, opts):
    nfdef = 1
    if len(inppath) > 0:
        mypath = inppath
    else:
        mypath = r'Z:\bms-host\\'

    filepaths = [inppath + f for f in listdir(mypath) if f.endswith('.csv')]
    NFILES = len(filepaths)
    Selectfiles = filepaths[NFILES-nfdef:NFILES]
    sel = int(input(''' Enter
          3     select file range
          2     use last two
          0     use last one [default]
          :  '''))
    if sel == 3:
        print(' {0:>s} --- {1:<s} \n'.format('File#', ' File'))
        for NFILE, f in enumerate(filepaths):
            print(' {0:>d} --- {1:<s} '.format(NFILE, f))
        NF1 = int(input('First File# in range: '))
        NF2 = int(input('Last  File# in range: '))
        Selectfiles = filepaths[NF1:NF2]
    if sel == 2:
       nfdef = 2
       Selectfiles = filepaths[NFILES-nfdef:NFILES]
    else:
       nfdef = 1
       Selectfiles = filepaths[NFILES-nfdef:NFILES]
#    else:
    # Selectfiles = filepaths[NFILES-nfdef:NFILES]
    filepaths = Selectfiles
    print(filepaths)
    return filepaths
#-----------------------------------------------------------------------------

def getFilesOpts(inppath, GetFileOpt):
    '''
    ------------------------------------------------------------------------
    filepaths < return >=
    getFilesOpts(inppath, GetFileOpt)

    ------------------------------------------------------------------------
    Parameters
    inppath <str>: long path name to search for files
    GetFileOpt <int>: option to decide number of files to process in inppath
      0   : process ALL FILES
      2   : process Last ONE file
      1   : process Last TWO files
      N   : processes the last N files

    ------------------------------------------------------------------------

    '''
    print(f' filerange = {GetFileOpt}')
    nfdef = 1
    if len(inppath) > 0:
        mypath = inppath ;
    else:
        mypath = SETDEF_PATH ;
    #filepaths = [inppath + f for f in listdir(mypath) if f.endswith('.csv')]
    begins='ES-612s-0#' ; ## archetype filename
    filepaths = [inppath + f \
                 for f in listdir(inppath) \
                   if f.endswith('.csv') and f.startswith(begins) ] ;

    NFILES = len(filepaths)
    if GetFileOpt == 0 :
        nfdef = NFILES
    if GetFileOpt == 1 :
        nfdef = 1
    if GetFileOpt == 2 :
        nfdef = 2
    if int(GetFileOpt) > 0:
        nfdef = int(GetFileOpt) ;
    Selectfiles = filepaths[NFILES-nfdef:NFILES]
#    else:
#        Selectfiles = filepaths[NFILES-nfdef:NFILES]
    filepaths = Selectfiles
    print(filepaths)
    return filepaths
#-----------------------------------------------------------------------------


def getFilesOpts_Str(inppath, GetFileOpt):
    '''
    ------------------------------------------------------------------------
    filepaths < return >=
    getFilesOpts_Str(inppath, GetFileOpt)
    ------------------------------------------------------------------------
    Parameters
    inppath <str>: long path name to search for files
    GetFileOpt <str>: option to decide number of files to process in inppath
      "allfiles"  : process ALL FILES
      "last"      : process Last ONE file
      "two"       : process Last TWO files
      <int>       : process the last N files

    ------------------------------------------------------------------------

    '''
    print(f' filerange = {GetFileOpt}')
    nfdef = 1
    if len(inppath) > 0:
        mypath = inppath
    else:
        mypath = r'Z:\bms-host\\'
    filepaths = [inppath + f for f in listdir(mypath) if f.endswith('.csv')]
    NFILES = len(filepaths)
    if GetFileOpt == 'allfiles':
        nfdef = NFILES
    if GetFileOpt =='last' :
        nfdef = 1
    if GetFileOpt == 'two' :
        nfdef = 2
    if int(GetFileOpt) > 0:
        nfdef = int(GetFileOpt) ;
    Selectfiles = filepaths[NFILES-nfdef:NFILES]
#    else:
#        Selectfiles = filepaths[NFILES-nfdef:NFILES]
    filepaths = Selectfiles
    print(filepaths)
    return filepaths
#-----------------------------------------------------------------------------



def getdf(Selectfiles):
    if SHOW[ 2 ]: print (f'nr of files as input: {len(Selectfiles)}');
    if len(Selectfiles) == 1:
      if SHOW[ 2 ]: print(f' Executing...  pd.read_csv(Selectfiles) ... ');
      if SHOW[ 2 ]: print(f' Selectfiles = {Selectfiles}');
      df = pd.read_csv( str(Selectfiles[0]) );
    if len(Selectfiles) > 1:
      if SHOW[ 2 ]: print(f' Executing...  pd.concat(map(pd.read_csv, Selectfiles)) ...');
      if SHOW[ 2 ]: print(f' Selectfiles = {Selectfiles}');
      df = pd.concat(map(pd.read_csv, Selectfiles));
    return df;

#-----------------------------------------------------------------------------


def SelectFilesGen( inppath, GetFileOpt, begins='' ):
    '''
    Selectfiles = \
       SelectFilesGen ( inppath, GetFileOpt [ beg, end ] )
    '''
    Selectfiles = '';
    begins = '';
    if SHOW[10]:
      print(f' filerange = {GetFileOpt}')
      print (
       'searching for files that start with: {}'.format(
        begins));
    filepaths = [inppath + f \
                  for f in listdir(inppath) \
                  if f.endswith('.csv') and f.startswith(begins) ];
    NFILES = len( filepaths ) ;
    if (type (GetFileOpt) == list ):
      if len(GetFileOpt) > 1:
          Selectfiles = filepaths[ GetFileOpt[0]:GetFileOpt[1] ];
    else:
    #if len(GetFileOpt) == 1:
        if GetFileOpt <   0 : ## last_nfiles
            Selectfiles = filepaths[ -GetFileOpt :             ];
        if GetFileOpt == -1 : ## last_nfiles
            Selectfiles = filepaths[ -GetFileOpt :             ];
            Selectfiles = [ filepaths[ -1 ]] ;

        if GetFileOpt ==  1 : ## last_nfiles
            Selectfiles = filepaths[ -GetFileOpt :             ];
            Selectfiles = [ filepaths[ 0 ]] ;

        if GetFileOpt >   0 : ## first_nfiles
            Selectfiles = filepaths[           0 : GetFileOpt  ];
        if GetFileOpt ==  0 : ## all_files
            Selectfiles = filepaths[ : ];


    if SHOW[11]: print (
                 'files that will be included : {}'.format(
                  Selectfiles ));
    ##if SHOW[11]:
    print (
      'nr files found and thereby in Selectfiles: {}'.format(
        len(Selectfiles) ));
    return Selectfiles;
#-----------------------------------------------------------------------------

def SelectfilesBay( inppath, CaseNr, GetFileOpt, bmsmark=4, XSHOW=False ):
    '''
    Selectfiles = \
       SelectfilesBay2( inppath, CaseNr, GetFileOpt [ beg, end ] )
    '''
    # XSHOW=XSHOW 
    if XSHOW: print(f' filerange = {GetFileOpt}')

    Selectfiles = '';
    
    begins = KWBEG + str(min(1, int(CaseNr - 0) )) ;#+ '#';
    begins = KWBEG;
    # if (bmsmark == 3):
    #   begins = KWBEG + str(CaseNr - 0) ;#+ '#';
    # if (bmsmark == 4):
    #   begins = KWBEG + str(CaseNr - 0) ;#+ '#';
    
    if XSHOW: print (
      'searching for files that start with: {}'.format(
        begins));
    filepaths = [inppath + f \
                  for f in listdir(inppath) \
                  if f.endswith('.csv') and f.startswith(begins) ];
    NFILES = len( filepaths ) ;
    if (type (GetFileOpt) == list ):
      if len(GetFileOpt) == 2:
            Selectfiles =  filepaths[ GetFileOpt[0]:GetFileOpt[1] ];
      if len(GetFileOpt)  > 2:
            Selectfiles =  filepaths[ GetFileOpt ];
    else: #if len(GetFileOpt) == 1:
        if GetFileOpt <   0 : ## last_nfiles
            Selectfiles = filepaths[ NFILES + GetFileOpt : NFILES ];
        if GetFileOpt == -1 : ## last_nfiles
            # Selectfiles = filepaths[ -GetFileOpt :           ];
            Selectfiles = [ filepaths[ -1 ]] ;

        if GetFileOpt ==  1 : ## last_nfiles
            # Selectfiles = filepaths[ -GetFileOpt :           ];
            Selectfiles = [ filepaths[ 0 ]] ;

        if GetFileOpt >   0 : ## first_nfiles
            Selectfiles = filepaths[           0 : GetFileOpt  ];
        if GetFileOpt ==  0 : ## all_files
            Selectfiles = filepaths[ : ];
            Selectfiles = filepaths[ : ];

    if XSHOW: print (
       'files that will be included : {}'.format(
                  Selectfiles ));
    if XSHOW: print (
        'nr files found and thereby in Selectfiles: {}'.format(
                  len(Selectfiles) ));
    if 1==1 : print (
        '#f {}'.format(
                  len(Selectfiles) ));
    return Selectfiles;
#-----------------------------------------------------------------------------

def SelectfilesBay1( inppath, CaseNr, GetFileOpt ):
    '''
    Selectfiles = \
    SelectfilesBay1( inppath, CaseNr, GetFileOpt )
    '''
    print(f' filerange = {GetFileOpt}')
    Selectfiles = '';

    begins = KWBEG + str(CaseNr - 0) + '#';
    #if SHOW[10]:
    print (
      'searching for files that start with: {}'.format(
        begins));
    filepaths = [inppath + f \
                  for f in listdir(inppath) \
                  if f.endswith('.csv') and f.startswith(begins) ];
    NFILES = len( filepaths ) ;
    if GetFileOpt <   0 : ## last_nfiles
        #Selectfiles = getfiles.getFiles_last(  inppath , GetFileOpt, begins  );
        Selectfiles = filepaths[ -GetFileOpt :             ];
    if GetFileOpt >   0 : ## first_nfiles
        #Selectfiles = getfiles.getFiles_first( inppath , GetFileOpt, begins );
        Selectfiles = filepaths[           0 : GetFileOpt  ];
    if GetFileOpt ==  0 : ## all_files
        Selectfiles = filepaths[ : ];
    # if DEB[1]: print (
    #              'files that will be included : {}'.format(
    #               Selectfiles ));
    ##if SHOW[11]:
    print (
    'File count registered for Selectfiles: {}'.format( len(Selectfiles) ));
    return Selectfiles;
#-----------------------------------------------------------------------------


# In[1]: timetemp