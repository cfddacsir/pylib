## loadimports.py
'''
import loadimports; ### load imports...
for r in loadimports.impload(): exec(r);
'''


notvalid='''import ;
for r in loadimports.impload(): exec(r);
 partially initialized module 'myimports' has no attribute 'impload' (most likely due to a circular import)
 '''
##############################################################################
# In[0]: 
def impload( show=False ):
    impfile = '/Users/dac/pylab/lib/imports.py';
    myimp = list();
    ignore="#"; 
    with open( impfile ,'r',newline='\n') as inp:
        for row in inp:  #.readline():
            #row = inp.read();
            row = row.strip('\n');
            if len(ignore)>0:
                ##print(f"testing {ignore}")
                if show: print (row);
                if row.startswith( ignore ):
                    pass;
                else: myimp.append(row);
            else:     myimp.append(row);
    inp.close();
    return myimp; ## list of imports
def imprecall( myimp ):
    if len(myimp)<1:
        myimp = impload();
    f=" {} \t {} "
    for i,r in enumerate( myimp):
        print( (f).format( str(i+1), str(r) ));
    return myimp;

def impshow( ):
    myimp = impload();
    f="{1}"
    for i,r in enumerate( myimp):
        print( (f).format( str(i+1), str(r) ));
    return myimp;
    
def impexec( myimp=[], ii=0, ):
    if len(myimp)<1:
        myimp = impload();
    if ii==0: ## importing all
        print("doing all! ");
        for each in myimp:
            print(f"doing .. {each}");
            exec( each );
    else:
        each = myimp[ii-1];
        print(f"doing .. {each}");
        exec( each );

def hint():
    hint='''
    # load imports:
    for r in loadimports.impload(): exec(r);
    '''
    print( hint );
    return hint;
def loadall():
    """
    for r in loadimports.impload(): exec(r);"""
    for r in loadimports.impload(): exec(r);
# hint() ;
##############################################################################
if 0: 
    for r in loadimports.impload(): exec(r);
    h = loadimports.hint(); print( h );
if __name__=='__main__':
    hint() ;
    #myimp = impload( show=False);
    #print( myimp );
    #imprecall( myimp) ;
    #importsrecall( myimp );
    impexec( [], ii=0);
##############################################################################

