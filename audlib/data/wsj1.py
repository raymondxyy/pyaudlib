#
# !!pyaudiolib is required for this script to work!!
# This file implements a class of functions that manipulate the WSJ1 dataset.
# Use this script to:
#   1. Convert .wv (sphere files) to .wav files
#   2. Prepare train.fileids and test.fileids for WSJ1
#
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)
#
# Updates:
#
#

# Define global variables for your purpose
CONFIG = { # configuration for WSJ1
    # SPHINX-related parameters
    #'EXP_BASEDIR': "/home/airbus/cmusphinx/wsj1", #SPHINX experiment base dir.
    'EXP_BASEDIR': "/home/xyy/local/tutorial/wsj1", # on ubuntu
    ## every path below is relative to basedir
    'EXP_WAV': 'wav', # path to all audio files
    'EXP_ETC': 'etc', # path to user input files
    # WSJ1-related parameters
    #'WSJ1_BASEDIR': '/Volumes/Data/csr_2_comp/', # WSJ1 dataset base dir.
    'WSJ1_BASEDIR': '/home/xyy/data/csr_2_comp/', # on ubuntu
    ## every path below is relative to basedir
    'TRAIN_NDX': '13-34.1/wsj1/doc/indices/si_tr_s.ndx', # relative path
                                                          # to training index
    'TEST_NDX': '13-34.1/wsj1/doc/indices/h1_p0.ndx', # rel. path to test index
    'TRAIN_TRANS': '13-34.1/wsj1/trans/wsj1/si_tr_s', # train transcription dir
    'TEST_TRANS': '13-34.1/wsj1/trans/wsj1/si_dt_20', # test transcription dir
    'DICT': 'cmudict.0.6d', # dictionary path
    'FILLER': 'wsj1.filler', # filler dictionary path
    'STSEG-READ': 'stseg-read' # binary executable for stseg-read
}

from subprocess import check_output
import os
from pyaudiolib.io.audio_io import sph2wav
from pyaudiolib.sphinx.sphinx_io import read_sphinx
import numpy as np

def prepare_train(verbose=False):
    """
    This function reads in a list of training files defined in the training
    index file, and converts all .wv (sphere) files to .wav (wave) files and
    save to wav/ directory in experiment directory. In addition, it creates a
    train.fileid file and save to etc/.
    """
    train_idx = os.path.join(CONFIG['WSJ1_BASEDIR'],CONFIG['TRAIN_NDX'])
    outdir_wav = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_WAV'])
    outdir_etc = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    assert os.path.exists(train_idx)

    with open(train_idx) as fp:
        lines = fp.readlines()
    processed = 0 # number of existing training files
    skipped = 0 # number of skipped files
    wfp = open(os.path.join(outdir_etc,'wsj1_train.fileids'),'w')
    for line in lines:
        # Template: 13_11_1: /wsj1/si_tr_s/4a8/4a8c0201.wv1
        line = line.strip()
        if line[0] == ';': continue # header
        disc, fpath = line.split()
        n1,n2,n3 = disc[:-1].split('_')
        fpath = "{}-{}.{}{}".format(n1,n2,n3,fpath)
        ffpath = os.path.join(CONFIG['WSJ1_BASEDIR'],fpath) # full file path
        if not os.path.exists(ffpath):
            print("SKIPPED [{}]: file does not exist.".format(ffpath))
            skipped += 1
        else:
            # output path - keep the si_tr_s directory
            wpath = '/'.join(fpath.split('/')[-3:]).split('.')[0]
            # convert audio
            sph2wav(ffpath,os.path.join(outdir_wav,wpath+'.wav'),
                    verbose=verbose)
            # write index file
            wfp.write("{}\n".format(wpath))
            processed += 1
    print("Total training file: [{}]".format(processed+skipped))
    print("Processed training file: [{}]".format(processed))
    print("Skipped training file: [{}]".format(skipped))
    return

def prepare_test(verbose=False):
    """
    This function reads in a list of training files defined in the test
    index file, and converts all .wv (sphere) files to .wav (wave) files and
    save to wav/ directory in experiment directory. In addition, it creates a
    test.fileid file and save to etc/.
    """
    test_idx = os.path.join(CONFIG['WSJ1_BASEDIR'],CONFIG['TEST_NDX'])
    outdir_wav = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_WAV'])
    outdir_etc = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    assert os.path.exists(test_idx)

    with open(test_idx) as fp:
        lines = fp.readlines()
    processed = 0 # number of existing training files
    skipped = 0 # number of skipped files
    wfp = open(os.path.join(outdir_etc,'wsj1_test.fileids'),'w')
    for line in lines:
        # Template: 13_16_1:wsj1/si_dt_20/4k0/4k0c0301.wv1
        line = line.strip()
        if line[0] == ';': continue # header
        disc, fpath = line.split(':')
        n1,n2,n3 = disc.split('_')
        fpath = "{}-{}.{}/{}".format(n1,n2,n3,fpath)
        ffpath = os.path.join(CONFIG['WSJ1_BASEDIR'],fpath) # full file path
        if not os.path.exists(ffpath):
            print("SKIPPED [{}]: file does not exist.".format(ffpath))
            skipped += 1
        else:
            # output path - keep the si_tr_s directory
            wpath = '/'.join(fpath.split('/')[-3:]).split('.')[0]
            # convert audio
            sph2wav(ffpath,os.path.join(outdir_wav,wpath+'.wav'),
                    verbose=verbose)
            # write index file
            wfp.write("{}\n".format(wpath))
            processed += 1
    print("Total test file: [{}]".format(processed+skipped))
    print("Processed test file: [{}]".format(processed))
    print("Skipped test file: [{}]".format(skipped))
    return

def prepare_trans(option,verbose=False):
    """
    Collate all transcription files to one file according to S3 format. This
    function assumes "common read" transcripts.
    """
    etcdir = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    if option == 'train':
        fileid_path = os.path.join(etcdir,'wsj1_train.fileids')
        trans_path = os.path.join(CONFIG['WSJ1_BASEDIR'],CONFIG['TRAIN_TRANS'])
        outpath = os.path.join(etcdir,'wsj1_train.transcription')
    elif option == 'test':
        fileid_path = os.path.join(etcdir,'wsj1_test.fileids')
        trans_path = os.path.join(CONFIG['WSJ1_BASEDIR'],CONFIG['TEST_TRANS'])
        outpath = os.path.join(etcdir,'wsj1_test.transcription')
    else:
        raise ValueError('Must be one of train/test.')

    # Real all used files and save file ids
    fids = []
    with open(fileid_path) as fp:
        lines = fp.readlines()
    for line in lines:
        fids.append(line.strip().split('/')[-1])

    wfp = open(outpath,'w')
    counter = 0 # number of transcripts used
    # Walk through all .lsn files. Append transcripts to file as needed.
    for root, dirs, files in os.walk(trans_path):
        for fname in filter(lambda s: s.endswith('.lsn'), files):
            with open(os.path.join(root,fname)) as fp:
                lines = fp.readlines()
            for line in lines:
                content = line.strip().split()
                trans,fid = content[:-1],content[-1][1:-1].lower()
                if fid in fids: # transcript has audio in fileids
                    wfp.write("<s> {} </s> ({})\n".format(' '.join(trans),fid))
                    counter += 1
                    if verbose:
                        print("[{}] files processed.".format(counter))
                else:
                    if verbose:
                        print("[{}] is not in [{}]".format(fid,fileid_path))
    wfp.close()

    # Print summary
    print("######### Summary of prepare_trans in wsj1.py ############")
    print("[{}] files in [{}].".format(len(fids),fileid_path))
    print("[{}] transcripts found in [{}].".format(counter,trans_path))
    print("Writing all transcripts to [{}].".format(outpath))
    if counter != len(fids):
        print("ERROR: Some files are missing transcriptions!")
    return

def check_trans(option,verbose=False):
    """
    Check all transcripts and make sure that all words appear in dictionary.
    If not, remove the file from fileids and transcript.
    """
    etcdir = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    dicpath = os.path.join(etcdir,CONFIG['DICT'])
    fillpath = os.path.join(etcdir,CONFIG['FILLER'])
    if option == 'train':
        infileid = os.path.join(etcdir,'wsj1_train.fileids')
        intrans = os.path.join(etcdir,'wsj1_train.transcription')
        outfileid = os.path.join(etcdir,'wsj1_train_check.fileids')
        outtrans = os.path.join(etcdir,'wsj1_train_check.transcription')
    elif option == 'test':
        infileid = os.path.join(etcdir,'wsj1_test.fileids')
        intrans = os.path.join(etcdir,'wsj1_test.transcription')
        outfileid = os.path.join(etcdir,'wsj1_test_check.fileids')
        outtrans = os.path.join(etcdir,'wsj1_test_check.transcription')
    else:
        raise ValueError('Must be one of train/test.')

    # Read in dictionaries
    with open(dicpath) as fp:
        lines = fp.readlines()
    dic = [line.split()[0] for line in lines]
    with open(fillpath) as fp:
        lines = fp.readlines()
    dic.extend([line.split()[0] for line in lines])
    print("Collected [{}] words in dictionaries".format(len(dic)))

    exclude = [] # a list of fids to exclude
    wfp = open(outtrans,'w')
    with open(intrans) as fp:
        lines = fp.readlines()
    processed = 0
    for line in lines:
        content = line.strip().split()
        content, fid = content[:-1],content[-1][1:-1]
        good = 1 # is current file good for processing?
        for word in content:
            if word not in dic: # word is not defined in dictionary
                if verbose:
                    print("[{}] does not exist in dictionary. Skipping [{}].".format(word,fid))
                good = 0
                break
        if not good:
            exclude.append(fid)
        else: # Keep this transcript
            wfp.write(line)
        processed += 1
        if not (processed % 1000) and verbose:
            print("[{}] files processed.".format(processed))
    wfp.close()

    # Exclude audio files from fileids
    wfp = open(outfileid,'w')
    with open(infileid) as fp:
        lines = fp.readlines()
    for line in lines:
        fid = line.strip().split('/')[-1]
        if fid not in exclude:
            wfp.write(line)
    wfp.close()

    # Print summary
    print("Excluding [{}] files.".format(len(exclude)))
    if verbose:
        print("Excluding file list: [{}]".format(','.join(exclude)))
    print("New fileids file is written to [{}].".format(outfileid))
    print("New transcript file is written to [{}].".format(outtrans))

    return

def sort_trans(option,verbose=False):
    """
    Sorting fileids and transcription files such that each row corresponds to
    the same utterance.
    """
    etcdir = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    if option == 'train':
        infileid = os.path.join(etcdir,'wsj1_train.fileids')
        intrans = os.path.join(etcdir,'wsj1_train.transcription')
        outfileid = os.path.join(etcdir,'wsj1_train.fileids.sorted')
        outtrans = os.path.join(etcdir,'wsj1_train.transcription.sorted')
    elif option == 'test':
        infileid = os.path.join(etcdir,'wsj1_test.fileids')
        intrans = os.path.join(etcdir,'wsj1_test.transcription')
        outfileid = os.path.join(etcdir,'wsj1_test.fileids.sorted')
        outtrans = os.path.join(etcdir,'wsj1_test.transcription.sorted')
    else:
        raise ValueError('Must be one of train/test.')

    with open(outfileid,'w') as wfp:
        wfp.write(''.join(sorted(open(infileid),key=lambda l: l.strip().split('/')[-1])))
    with open(outtrans,'w') as wfp:
        wfp.write(''.join(sorted(open(intrans),key=lambda l: l.strip().split()[-1])))
    return

def prepare_feat(aligndir, outpath, verbose=False):
    """
    Take feature files from `featdir` and alignment files from `aligndir` and
    produce feature,label pairs and store in `outpath`.
    """
    featdim = 40
    featdir = os.path.join(CONFIG['EXP_BASEDIR'],'feat')
    with open(os.path.join(aligndir,'wsj1.alignedfiles')) as fp:
        featpaths = fp.readlines()

    # Walk through all feature files; find its alignment file and append to out
    counter = 0
    out = []
    for fpath in featpaths:
        if verbose: print("Processing [{}]".format(fpath.strip()))
        counter += 1
        print("Processing [{}]/[{}].".format(counter,len(featpaths)))
        fid = fpath.strip().split('/')[-1]
        featpath = os.path.join(featdir,fpath.strip()+'.logspec')
        feat = read_sphinx(featpath, featdim) # TODO: bad style...
        out.append(feat)

    # Write to outpath
    print("Saving [{}] processed feature files to [{}]".format(len(out),outpath))
    np.save(outpath,out)

def prepare_label(aligndir, phonedicpath, outpath, verbose=False):
    """
    Take feature files from `featdir` and alignment files from `aligndir` and
    produce feature,label pairs and store in `outpath`.
    """
    # load subphoneme list from model architecture
    phonedic = load_phonedic(phonedicpath)
    with open(os.path.join(aligndir,'wsj1.alignedfiles')) as fp:
        featpaths = fp.readlines()

    # Walk through all feature files; find its alignment file and append to out
    out = [] # array of training data to be stored
    counter = 0
    for fpath in featpaths:
        if verbose: print("Processing [{}]".format(fpath.strip()))
        counter += 1
        print("Processing [{}]/[{}].".format(counter,len(featpaths)))
        fid = fpath.strip().split('/')[-1]
        alignpath = os.path.join(aligndir,'stseg/{}.stseg'.format(fid))
        align = parse_align(alignpath, phonedic)
        out.append(align)

    # Write to outpath
    print("Saving [{}] processed files to [{}]".format(len(out),outpath))
    np.save(outpath,out)

def load_phonedic(phonedicpath):
    """
    Load phoneme list to find labels.
    """
    with open(phonedicpath) as fp:
        lines = fp.readlines()[11:]
    dic = {}
    for line in lines:
        bs,lf,rt,p,attr,snum,sid1,sid2,sid3,_ = line.strip().split()
        if '-' in (lf+rt+p):
            statestr = bs
        else:
            statestr = "{} {} {} {}".format(bs,lf,rt,p)
        assert statestr not in dic
        dic[statestr] = [int(sid1),int(sid2),int(sid3)]

    return dic

def parse_align(inpath, phonedic):
    """
    Read the alignment file and return a list of subphoneme labels.
    """
    aligntxt = check_output("cat {} | {}".format(inpath,CONFIG['STSEG-READ']),
                            shell=True).strip().split('\n')[2:]
    label = [] # hold output label
    for line in aligntxt:
        line = line.split()
        framenum,llk,sid,statestr = line[0],line[1],line[2],line[3:]
        statestr = ' '.join(statestr)
        label.append(phonedic[str(statestr)][int(sid)])

    return label

def prepare_feat_test(outpath, verbose=False):
    """
    Prepare feature files for test.
    """
    featdim = 40
    featdir = os.path.join(CONFIG['EXP_BASEDIR'],'feat/si_dt_20') # test directory

    # Walk through all feature files; find its alignment file and append to out
    out = []
    for root, dirs, files in os.walk(featdir):
        for fname in files:
            if not fname.endswith('.logspec'): continue
            fpath = os.path.join(root,fname)
            if verbose: print("Processing [{}]".format(fpath))
            feat = read_sphinx(fpath, featdim) # TODO: bad style...
            out.append(feat)

    # Write to outpath
    print("Saving [{}] processed feature files to [{}]".format(len(out),outpath))
    np.save(outpath,out)

def contains_only(seq, cset, verbose=False):
    """
    Check if sequence `seq` contains only characters in `cset`.
    """
    for c in seq:
        if c not in cset:
            if verbose: print('[{}] is not valid!'.format(c))
            return False
    return True


def prepare_char_feats(fileid, outpath, verbose=False):
    featdim = 40
    with open(fileid) as fp:
        lines = fp.readlines()
    out = []
    for line in lines:
        fpath = os.path.join(CONFIG['EXP_BASEDIR'],'feat/{}.logspec'.format(line.strip()))
        if verbose: print("Processing [{}]".format(fpath))
        feat = read_sphinx(fpath, featdim) # TODO: bad style...
        out.append(feat)

    # Write to outpath
    print("Saving [{}] processed feature files to [{}]".format(len(out),outpath))
    np.save(outpath,out)

def prepare_char_transcripts(inpath, outpath, verbose=False):
    """
    Takes transcription from `inpath` and output tranascripts that have only
    valid characters.
    """
    cset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz #'-/@_")
    with open(inpath) as fp:
        lines = fp.readlines()
    out = []
    for ii,line in enumerate(lines):
        strlist = line.strip().split()
        lineout, fid = ' '.join(strlist[1:-2]), strlist[-1]

        # Remove special characters here
        chars_to_remove = [':','.','(',')','"']
        lineout = lineout.translate(None, "".join(chars_to_remove))

        if not contains_only(lineout, cset, verbose):
            print("File [{}] Line [{}] contains illegal character.".format(fid,lineout))
            exit(-1)
        out.append(lineout)
        print("[{}/{}] transcripts processed.".format(ii+1,len(lines)))
    np.save(outpath, out)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-td',
        help='Prepare training data and fileids for WSJ1', required=False,
        action='store_true',default=False)
    parser.add_argument('-ed',
        help='Prepare test data and fileids for WSJ1', required=False,
        action='store_true',default=False)
    parser.add_argument('-tt',
        help='Prepare training transcripts for WSJ1. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-et',
        help='Prepare test transcripts for WSJ1. REQUIRES existing fileids.',
        required=False,action='store_true',default=False)
    parser.add_argument('-tc',
        help='Check training transcripts/fileids for WSJ1. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-ec',
        help='Check test transcripts/fileids for WSJ1. REQUIRES existing fileids.',
        required=False,action='store_true',default=False)
    parser.add_argument('-ts',
        help='Sort training transcripts/fileids for WSJ1. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-es',
        help='Sort test transcripts/fileids for WSJ1. REQUIRES existing fileids.',
        required=False,action='store_true',default=False)
    parser.add_argument('-pf',
        help='Prepare features. !!Make sure you configure paths correctly!!',
        required=False,action='store_true',default=False)
    parser.add_argument('-pl',
        help='Prepare labels. !!Make sure you configure paths correctly!!',
        required=False,action='store_true',default=False)
    parser.add_argument('-ef',
        help='Prepare features for test. !!Make sure you configure paths correctly!!',
        required=False,action='store_true',default=False)
    parser.add_argument('-chartrans',
        help='Prepare character transcripts for CTC/attention models. !!Make sure you configure paths correctly!!',
        required=False,action='store_true',default=False)
    parser.add_argument('-charfeats',
        help='Prepare character features for CTC/attention models. !!Make sure you configure paths correctly!!',
        required=False,action='store_true',default=False)
    parser.add_argument('-v',
        help='Enable verbose', required=False,action='store_true',default=False)
    args = parser.parse_args()

    if args.td:
        prepare_train(verbose=args.v)
    if args.ed:
        prepare_test(verbose=args.v)
    if args.tt:
        prepare_trans('train',verbose=args.v)
    if args.et:
        prepare_trans('test',verbose=args.v)
    if args.tc:
        check_trans('train',verbose=args.v)
    if args.ec:
        check_trans('test',verbose=args.v)
    if args.ts:
        sort_trans('train',verbose=args.v)
    if args.es:
        sort_trans('test',verbose=args.v)
    if args.pf:
        # Uncomment for CI-model feature preparation
        mdefpath = '{}/model_architecture/wsj1.falign_ci.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        aligndir = '{}/falignout_ci_test'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        # Uncomment for CD-model feature preparation
        #mdefpath = '{}/model_architecture/wsj1.3000.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        #aligndir = '{}/falignout_cd_3000'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        outpath = 'feat_ci_test.npy' # output file
        prepare_feat(aligndir,outpath,verbose=args.v)
    if args.pl:
        mdefpath = '{}/model_architecture/wsj1.falign_ci.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        aligndir = '{}/falignout_ci_test'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        #mdefpath = '{}/model_architecture/wsj1.3000.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        #aligndir = '{}/falignout_cd_3000'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        outpath = 'label_ci_test.npy' # output file
        prepare_label(aligndir,mdefpath,outpath,verbose=args.v)
    if args.ef:
        outpath = 'feat_test.npy'
        prepare_feat_test(outpath,args.v)

    if args.chartrans:
        inpath = '/home/airbus/cmusphinx/wsj1/etc/wsj1_train.transcription'
        outpath = 'wsj1_train_transcripts.npy'
        prepare_char_transcripts(inpath, outpath, args.v)
    if args.charfeats:
        fileid = '/home/airbus/cmusphinx/wsj1/etc/wsj1_train.fileids'
        outpath = 'wsj1_train_feats.npy'
        prepare_char_feats(fileid, outpath, args.v)
