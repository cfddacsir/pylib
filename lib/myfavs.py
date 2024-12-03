##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
myfavs.py

Created on Tue Oct  8 09:47:01 2024

@author: dac
"""

##############################################################################
# In[0]: Load packages
#------------------------------------------------------------------------------
####  Import Packages
import re ;
from   os import listdir;
import os;
import string ;
import math ;
import linecache ;
import argparse ;
import statistics ;
import pandas as pd ;
import numpy as np ;
import matplotlib as mpl ;
import matplotlib.pyplot as plt ;
import matplotlib.font_manager as fontmgr ;
from   scipy import signal;
from   datetime import datetime ;

#import requests
#import bs4 
#import seaborn as sn;
#-----------------------------------------------------------------------------#
###############################################################################
def loadimports():
    """
    for r in loadimports.impload(): exec(r);
    loaded = pd.DataFrame( loadimports.impload(), columns=['load'] ); print(loaded);
    """
    import loadimports;
    for r in loadimports.impload(): exec(r);
    loaded = pd.DataFrame( loadimports.impload(), columns=['load'] ); print(loaded);
    hint="""
    for r in loadimports.impload(): exec(r);
    loaded = pd.DataFrame( loadimports.impload(), columns=['load'] ); print(loaded);
    """
    print(hint);
    return loaded;

###############################################################################
EXE = {
       'inp':   None, # (os.getcwd() ),
       'v':     False,
      };
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import myfavs as my;
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hr = 60*'==';
br = 60*'--';
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pbr(i=1,n=60,):
    symbols= '''#=-~''';
    br = n*symbols[i] ;
    print (br);
    return br;

def pbs(i=0,n=60,prefix="# "):
    prefix = "" + ""*(prefix==None)
    symbols = [chr(i) for i in range(32, 64) ];
    br = prefix + (n-len(prefix))*symbols[i] ;
    print (br);
    return br;

def brsep(pr=False, c='*', n=80, a=0, e=0):
    br = ( a*'\n' + n*f'{c}' +  1*e*'\n');
    #br = ( a*'\n' +   '{}'  +  1*e*'\n').format( c * n);
    if pr: print(br,  end='' ); #+ e*'\n');
    return br;

def alphabet():
    alphabet = [chr(i) for i in range(65, 65+26) ]
    return alphabet;

def smybols():
    alphabet = [chr(i) for i in range(32, 64) ]
    return alphabet;

def pcenter( s, spad="#",sp = " ",n=60, pr=True):
    e = (int(n/2) - int((len(s)+len(sp))/2) )//len(spad);
    R = str( (e)*spad + (sp)+(s)+(sp) +(e)*spad) ;
    if pr: print(f"{R}");
    return R

def pleft( s, spad="#",sp = " ",n=60, rpad="-", pr=True):
    jle =  4;
    jre =  max(0,n - (jle*len(spad) + len(s) + 5*len(sp)));
    #if jre <1: jre=0;
    R = str( (jle)*spad + (1*sp)+(s)+(4*sp) +(jre)*rpad);
    if pr: print(f"{R}");
    return R

######################################################################
  
def whoami( x ):
    print( type( x) )
    #if type(x) == 
    print( )
    return None;



######################################################################

###############################################################################
# Months to Months
#------------------------------------------------------------------------------
month2mon={
        'Jan' :  1 ,
        'Feb' :  2 ,
        'Mar' :  3 ,
        'Apr' :  4 ,
        'May' :  5 ,
        'Jun' :  6 , 
        'Jul' :  7 ,
        'Aug' :  8 ,
        'Sep' :  9 ,
        'Oct' :  10 ,
        'Nov' :  11 ,
        'Dec' :  12 ,
}
mon2month={
        1: 'Jan', 
        2: 'Feb', 
        3: 'Mar', 
        4: 'Apr', 
        5: 'May', 
        6: 'Jun', 
        7: 'Jul', 
        8: 'Aug', 
        9: 'Sep', 
        10: 'Oct', 
        11: 'Nov', 
        12: 'Dec',
    }
def get_mon2month():
    return mon2month;

def get_month2mon():
    return month2mon;

def get_months( out='s' ):
    '''
    set out to s for making Jan..Dec
    set out to i for making 01.. 12
    '''
    if out=='i':
        return month2mon;
    if out=='s':
        return month2mon;

def monthsordered( rettype='list' ):    
    pass;

def monthstr_2_monint(dfa, colmonth='Month'):
    MON=list();
    #colmonth='Month'
    for d in dfa[colmonth]:
        m = 0 +\
        1*(d=='Jan') +\
        2*(d=='Feb') +\
        3*(d=='Mar') +\
        4*(d=='Apr') +\
        5*(d=='May') +\
        6*(d=='Jun') +\
        7*(d=='Jul') +\
        8*(d=='Aug') +\
        9*(d=='Sep') +\
       10*(d=='Oct') +\
       11*(d=='Nov') +\
       12*(d=='Dec') ;
        m = int(m);
        #print( d, m);
        MON.append(m);
    dfa['Mon'] = pd.DataFrame( MON );
    return dfa;
###############################################################################


######################################################################
# WRANGLE AND CLEAN
######################################################################

def reportaction( action, do=True ):
    print( pcenter( action,n=80,sp="  ") );
    if do:
     eval( 'print( {} ) '.format( action ));
    print( pcenter( "#",n=80,sp="") );
#=====================================================================
def dfcorr( df ):
    " look for not floats and drop or convert with dummies"
    
    pass;
#=====================================================================
# made special for reporting df characterisitics:dfcharacter(df) ----
def dfcharacter( df ):
    do=True;
    LACTIONS=[
        'df.shape',
        'df.columns',
        'df.head()',
        'df.info()',
        'df.describe()',
        #'df.corr()',
        #'my.report_missing(df)'
        'report_missing(df)'
    ]
    for action in ( LACTIONS ):
        #reportaction( h, do =False );
        print( pcenter( action,n=80,sp="  ") );
        #if do:
        eval( 'print( {} ) '.format( action ));
        print( pcenter( "#",n=80,sp="") );
#=====================================================================
##  test for missing
def missing(df):
    missing_data = df.isnull();
    print( missing_data.head(5) );
    for column in missing_data.columns.values.tolist():
        print(column);
        print (missing_data[column].value_counts());
        print("") ;
#=====================================================================
def report_missing( df , showreport=False ):
    missing_data = df.isnull() ;
    #missing_data.head(5) 
    report=list();
    for column in missing_data.columns.values.tolist():
        ct = missing_data[column].value_counts()[0] ;
        gr = missing_data[column].shape[0] ;
        null = gr - ct;
        report.append( [ column, ct, null, gr ] );
        #print (missing_data[column].value_counts())
        #print("")  
    rp = pd.DataFrame( data=report, index=None, columns=['col','nonull','Nulls', 'Size']);
    if showreport:
        pcenter( f" Showing report for df ");
        print(rp);
        pbr();
    return rp;
#=====================================================================
# Cleaning
## replace duds 
def cleanval( df ):
    s = '?'
    df.replace( s, np.nan, inplace=True);
    return df;
#=====================================================================
## repalce with mean
def replacemean(df, xlab ):
    pass   
    mu = df[ xlab ].astype("float").mean(axis=0);
    #mu
    df[ xlab ].replace( np.nan, mu, inplace=True) ;
    return df;

## replace with top frequent
def replacefreq( df, xlab, show=True, replace=True) :
    x1  = df[ xlab ].value_counts();
    top = vct = df[ xlab ].value_counts().idxmax(); 
    #print( "counts ", x1, "idmax: ", x2 )
    #top = vct
    if show: print ("top count: ", top)
    if replace: df[ xlab ].replace( np.nan, top, inplace=True);
    return df ;


###############################################################################
def osfss( ):   
    '''
    arg:  None
    ret: FSS <str> file-separator based on the operating system
    '''
    import platform; p = platform.uname();
    #import unicode; 
    import platform; p = platform.uname();
    if p.system == 'Windows': FS  =  str('\u005c') ;  ## == "\" #if system is Windows:
    else: FS  =  str('\u002f') ;  ## == "/"  # if NOT windows
    FSS =  str( 1*FS ) ;
    return FSS;

FSS = osfss();

def cwd():
    """ returns the current directory
    """
    cwd = os.getcwd();
    return cwd; 


###############################################################################
def initsetfile ( SETFILE, ofile=None, setkey=None ):
    """
    initialize ofile based on SETFILE
    return ofile[key]
    """
    if ofile==None: ofile = {} ;    
    thisfile = SETFILE 
    if 1:        
      if type(setkey) == str or int: k=setkey
      else: k=0;
      #if PSH[0]:
      #print('setkey', end='..', sep='.. ')
      #print( setkey )
    if 1:        
      ofile[k] =(open( thisfile, 'w',encoding='utf-8',newline=None, )).write('' );
      ofile[k] =(open( thisfile, 'a',encoding='utf-8',newline='\n', ));
    return ofile ;

def showheader( listparm, wf=None, ):
    for k, p in enumerate( listparm  ) : 
        if k < len( listparm  ) -1: 
              print(('{}'+1*','+0*' ').format( str( (p)) ), sep='', end=''  ,file=wf )
        else: print(('{}'+0*','+0*' ').format( str( (p)) ), sep='', end='\n',file=wf )
def showkeys( DICT, wf=None, ):
    for i, k in enumerate( DICT) : 
        p = DICT.get(k)
        print(('{:<3s} : {:<5s} : {}').format( str(i),str(k),str( p) ),
              sep='',end='\n',file=wf );
###############################################################################

def getdf(Selectfiles, PSH):
    if PSH[2] : print (f'nr of files as input: {len(Selectfiles)}');
    if len(Selectfiles) == 1:
      if PSH[2] : print(f' Executing...  pd.read_csv(Selectfiles) ... ');
      if PSH[2] : print(f' Selectfiles = {Selectfiles}');
      df = pd.read_csv( str(Selectfiles[0]) );
    if len(Selectfiles) > 1:
      if PSH[2] : print(f' Executing...  pd.concat(map(pd.read_csv, Selectfiles)) ...');
      if PSH[2] : print(f' Selectfiles = {Selectfiles}');
      df = pd.concat(map(pd.read_csv, Selectfiles));
    return df;

def get_date ( timestamp):
    gs  = datetime.fromisoformat(timestamp);
    ss  = '' ;
    su  = '_' ;
    Y=str(gs.year - 2000)
    A=(gs.month <10)*'0'+str(gs.month )
    D=(gs.day   <10)*'0'+str(gs.day   )
    H=(gs.hour  <10)*'0'+str(gs.hour  )
    I=(gs.minute<10)*'0'+str(gs.minute)
    fig_date = Y + ss + A + ss + D + su + H + ss + I ;
    return fig_date ;
###############################################################################

def getArgs( EXE= {} ):
    '''
    return EXE = getArgs() ;
    EXE = {
          };
    '''
  ##-----------------------------------------------------------    
    ##INARGS={};
    __doc__=''' '''
    __version__=''' '''
    __usage__=''' '''
    parser = argparse.ArgumentParser();
    parser.add_argument("-inp",
            nargs=1, type=str, default='' ,
            help="__<str:>: path of inputh file ",
            action='store',);
    args   = parser.parse_args();
    if (args.inp)!=None: EXE['inp'] = args.inp; print("will do inp: {}".format(args.inp));

    if EXE['v']: print( __doc__, __version__, __usage__ , end='\n\n' );
    return EXE ;

###############################################################################


def normalize( y, opt=2 ):
    """ y: dataframe-series 
    opt==1: y/ymax
    opt==2: y-ymin / ymax-ymin >def<
    """
    if opt==1:
        ynorm = (y )/ (y.max());
    if opt==2:
        ynorm = (y - y.min())/(y.max() - y.min());
    return ynorm;


##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<##


#################################################################################
if '__main__'==__name__: ## for testing purposes
 pass;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 if 0: ## TEST SAMPLE
    import sample_df;
    df = sample_df.sample_df();    
    print ( type (df))
    #my.dfcharacter(df);
    print( df.head() );
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 if 1: ## testing dfcharacter(df)
    import sample_df;
    df = sample_df.sample_df();
    dfcharacter(df);
#################################################################################
