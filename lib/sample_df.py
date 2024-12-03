#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sample df 
samplepath = '/Users/dac/pylab/lib/'
def sample_df(samplefile='sample.csv'):
    samplepath = '/Users/dac/pylab/lib/';
    samplefile = samplepath + samplefile;
    import pandas as pd;
    print( f"trying to read file :{samplefile} " );
    df = pd.read_csv( samplefile );
    df.drop(columns='Unnamed: 0',inplace=True);
    df.rename({'No_of_Children':'nchild'},axis=1,inplace=True);
    return df;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~