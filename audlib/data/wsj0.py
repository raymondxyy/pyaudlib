#
# !!pyaudiolib is required for this script to work!!
# This file implements a class of functions that manipulate the WSJ0 dataset.
# Use this script to:
#   1. Convert .wv (sphere files) to .wav files
#   2. Prepare train.fileids and test.fileids for SPHINX
#
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)
#
# Updates:
#
#

# Define global variables for your purpose
CONFIG = { # configuration for WSJ0
    # SPHINX-related parameters
    #'EXP_BASEDIR': "/home/airbus/cmusphinx/wsj0", #SPHINX experiment base dir.
    'EXP_BASEDIR': "/home/xyy/local/tutorial/wsj0", # on ubuntu
    ## every path below is relative to basedir
    'EXP_WAV': 'wav', # path to all audio files
    'EXP_ETC': 'etc', # path to user input files
    # WSJ0-related parameters
    #'WSJ0_BASEDIR': '/Volumes/Data/csr_2_comp/', # WSJ0 dataset base dir.
    'WSJ0_BASEDIR': '/home/xyy/data/csr_1/', # on ubuntu
    ## every path below is relative to basedir
    'TRAIN_NDX': '11-13.1/wsj0/doc/indices/train/tr_all_wv1.ndx', # relative path
                                                          # to training index
    'TEST_NDX': '11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx', # SI-20k NVP read
    'TRAIN_TRANS': '11-4.1/wsj0/transcrp/dots', # train transcription dir
    'TEST_TRANS': '11-14.1/wsj0/si_et_20', # test transcription dir
    'DICT': 'cmudict.0.7a', # dictionary path
    'FILLER': 'wsj0.filler', # filler dictionary path
    'STSEG-READ': 'stseg-read' # binary executable for stseg-read
}

CONVERT_AUDIO = False

from subprocess import check_output
import os
from pyaudiolib.io.audio_io import sph2wav, audioread
from pyaudiolib.sphinx.sphinx_io import read_sphinx
from pyaudiolib.analysis.transform import stft, audspec, logspec, magphase
import numpy as np
import re
from pdb import set_trace

def prepare_train(convert=True, verbose=False):
    """
    This function reads in a list of training files defined in the training
    index file, and converts all .wv (sphere) files to .wav (wave) files and
    save to wav/ directory in experiment directory. In addition, it creates a
    train.fileid file and save to etc/.
    """
    train_idx = os.path.join(CONFIG['WSJ0_BASEDIR'],CONFIG['TRAIN_NDX'])
    outdir_wav = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_WAV'])
    outdir_etc = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    assert os.path.exists(train_idx)

    with open(train_idx) as fp:
        lines = fp.readlines()
    processed = 0 # number of existing training files
    skipped = 0 # number of skipped files
    wfp = open(os.path.join(outdir_etc,'wsj0_train.fileids'),'w')
    fids = []
    for line in lines:
        # Template: 11_10_1:wsj0/sd_tr_s/00f/00fc0301.wv2
        line = line.strip()
        if line[0] == ';': continue # header
        disc, fpath = line.split(":")
        n1,n2,n3 = disc.split('_')
        fpath = "{}-{}.{}/{}".format(n1,n2,n3,fpath)
        ffpath = os.path.join(CONFIG['WSJ0_BASEDIR'],fpath) # full file path
        if not os.path.exists(ffpath):
            print("SKIPPED [{}]: file does not exist.".format(ffpath))
            skipped += 1
        else:
            fid = ffpath.split('/')[-1].split('.')[0]
            if fid in fids: continue
            fids.append(fid)
            # output path - keep the si_tr_s directory
            wpath = '/'.join(fpath.split('/')[-3:]).split('.')[0]
            # convert audio
            if convert:
                sph2wav(ffpath,os.path.join(outdir_wav,wpath+'.wav'),
                    verbose=verbose)
            # write index file
            wfp.write("{}\n".format(wpath))
            processed += 1
    print("Total training file: [{}]".format(processed+skipped))
    print("Processed training file: [{}]".format(processed))
    print("Skipped training file: [{}]".format(skipped))
    return

def prepare_test(convert=True, verbose=False):
    """
    This function reads in a list of training files defined in the test
    index file, and converts all .wv (sphere) files to .wav (wave) files and
    save to wav/ directory in experiment directory. In addition, it creates a
    test.fileid file and save to etc/.
    """
    test_idx = os.path.join(CONFIG['WSJ0_BASEDIR'],CONFIG['TEST_NDX'])
    outdir_wav = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_WAV'])
    outdir_etc = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    assert os.path.exists(test_idx)

    with open(test_idx) as fp:
        lines = fp.readlines()
    processed = 0 # number of existing training files
    skipped = 0 # number of skipped files
    wfp = open(os.path.join(outdir_etc,'wsj0_test.fileids'),'w')
    for line in lines:
        # Template: 13_16_1:wsj0/si_dt_20/4k0/4k0c0301.wv1
        line = line.strip()
        if line[0] == ';': continue # header
        disc, fpath = line.split(':')
        n1,n2,n3 = disc.split('_')
        fpath = "{}-{}.{}/{}".format(n1,n2,n3,fpath)
        ffpath = os.path.join(CONFIG['WSJ0_BASEDIR'],fpath) # full file path
        if os.path.exists(ffpath+'.wv1'):
            ffpath += '.wv1'
        elif os.path.exists(ffpath+'.wv2'):
            ffpath += '.wv2'
        else:
            print("SKIPPED [{}]: file does not exist.".format(ffpath))
            skipped += 1
            continue
        # output path - keep the si_et_20 directory
        wpath = '/'.join(fpath.split('/')[-3:]).split('.')[0]
        # convert audio
        if convert:
            sph2wav(ffpath,os.path.join(outdir_wav,wpath+'.wav'),
                verbose=verbose)
        # write index file
        wfp.write("{}\n".format(wpath))
        processed += 1
    print("Total test file: [{}]".format(processed+skipped))
    print("Processed test file: [{}]".format(processed))
    print("Skipped test file: [{}]".format(skipped))
    return

def format_dot(s):
    """
    Process a transcription string `s` in .dot format and output standard .lsn
    format for sphinx.
    """
    # template: ...transcript... (id)
    content = s.strip().split()
    trans,fid = content[:-1], content[-1][1:-1].lower()
    trans = filter(lambda item: ('[' not in item) and (item != '.') and (item != '~'), trans) # remove special chars
    trans = ' '.join(trans).upper()
    trans = re.sub('[\\\!:*]','',trans)
    trans = "<s> {} </s> ({})\n".format(trans,fid)
    return fid, trans

def prepare_trans(option, verbose=False):
    """
    Collate all transcription files to one file according to S3 format. This
    function assumes "common read" transcripts.
    """
    etcdir = os.path.join(CONFIG['EXP_BASEDIR'],CONFIG['EXP_ETC'])
    if option == 'train':
        fileid_path = os.path.join(etcdir,'wsj0_train.fileids')
        trans_path = os.path.join(CONFIG['WSJ0_BASEDIR'],CONFIG['TRAIN_TRANS'])
        outpath = os.path.join(etcdir,'wsj0_train.transcription')
    elif option == 'test':
        fileid_path = os.path.join(etcdir,'wsj0_test.fileids')
        trans_path = os.path.join(CONFIG['WSJ0_BASEDIR'],CONFIG['TEST_TRANS'])
        outpath = os.path.join(etcdir,'wsj0_test.transcription')
    else:
        raise ValueError('Must be one of train/test.')

    # Real all used files and save file ids
    fids = []
    with open(fileid_path) as fp:
        lines = fp.readlines()
    for line in lines:
        fids.append(line.strip().split('/')[-1])
    tot = len(fids)
    wfp = open(outpath,'w')
    counter = 0 # number of transcripts used
    # Walk through all .lsn files. Append transcripts to file as needed.
    for root, dirs, files in os.walk(trans_path):
        for fname in filter(lambda s: s.endswith('.dot'), files):
            #if fname == '001c0l00.dot': set_trace()
            with open(os.path.join(root,fname)) as fp:
                lines = fp.readlines()
            for line in lines:
                fid, trans = format_dot(line)
                if fid in fids: # transcript has audio in fileids
                    wfp.write(trans)
                    counter += 1
                    fids.remove(fid)
                    if verbose:
                        print("[{}] files processed.".format(counter))
    wfp.close()

    # Print summary
    print("######### Summary of prepare_trans in wsj0.py ############")
    print("[{}] files in [{}].".format(tot,fileid_path))
    print("[{}] transcripts found in [{}].".format(counter,trans_path))
    print("[{}] transcripts NOT found in [{}].".format(len(fids),trans_path))
    print("Writing all transcripts to [{}].".format(outpath))
    if counter != tot:
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
        infileid = os.path.join(etcdir,'wsj0_train.fileids')
        intrans = os.path.join(etcdir,'wsj0_train.transcription')
        outfileid = os.path.join(etcdir,'wsj0_train.fileids.checked')
        outtrans = os.path.join(etcdir,'wsj0_train.transcription.checked')
    elif option == 'test':
        infileid = os.path.join(etcdir,'wsj0_test.fileids')
        intrans = os.path.join(etcdir,'wsj0_test.transcription')
        outfileid = os.path.join(etcdir,'wsj0_test.fileids.checked')
        outtrans = os.path.join(etcdir,'wsj0_test.transcription.checked')
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
        infileid = os.path.join(etcdir,'wsj0_train.fileids')
        intrans = os.path.join(etcdir,'wsj0_train.transcription')
        outfileid = os.path.join(etcdir,'wsj0_train.fileids.sorted')
        outtrans = os.path.join(etcdir,'wsj0_train.transcription.sorted')
    elif option == 'test':
        infileid = os.path.join(etcdir,'wsj0_test.fileids')
        intrans = os.path.join(etcdir,'wsj0_test.transcription')
        outfileid = os.path.join(etcdir,'wsj0_test.fileids.sorted')
        outtrans = os.path.join(etcdir,'wsj0_test.transcription.sorted')
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
    with open(os.path.join(aligndir,'wsj0.alignedfiles')) as fp:
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
    with open(os.path.join(aligndir,'wsj0.alignedfiles')) as fp:
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
        chars_to_remove = ':.()",%&?;{}'
        lineout = lineout.translate(None, chars_to_remove)

        if not contains_only(lineout, cset, verbose):
            print("File [{}] Line [{}] contains illegal character.".format(fid,lineout))
            exit(-1)
        out.append(lineout)
        print("[{}/{}] transcripts processed.".format(ii+1,len(lines)))
    print("Saving [{}] processed transcripts to [{}]".format(len(out),outpath))
    np.save(outpath, out)

def dic2phone(dpath, outpath, verbose=False):
    plist = set([])
    wc = 0
    with open(dpath) as fp:
        lines = fp.readlines()
    for line in lines:
        content = line.strip().split()
        word, phones = content[0],set(content[1:])
        plist |= phones
        wc += 1
    if verbose:
        print("Summary of dictionary [{}]:".format(dpath))
        print("Total number of words = [{}]".format(wc))
        print("Total number of unique phones = [{}]".format(plist))
    print("Writing phone list to [{}]".format(outpath))
    # Extending plist to include SIL and garbage
    plist |= set(['+BREATH+','+GARBAGE+','+NOISE+','+SMACK+','SIL'])
    with open(outpath,'w') as fp:
        fp.write('\n'.join(sorted(list(plist))))
        fp.write('\n')
    return

def feat_dnn(fpath, verbose=False):
    """
    Prepare features for NN.
    """
    wlen = 0.025625 # SPHINX's default
    hop_fraction = .25
    nfft = 512
    nfilts = 40
    pwr_floor = 1e-10 # power floor below which frame will be rounded off
    x, xsr = audioread(fpath, verbose=verbose)
    tmap, fmap, X = stft(x, xsr,
            window_length=wlen,
            #w_start = 0,
            hop_fraction=hop_fraction,
            nfft=nfft,
            stft_truncate=True)
    Xmag, _ = magphase(X)

    # CMN
    
    Xmel, _ = audspec(Xmag**2, nfft=nfft, sr=xsr, nfilts=nfilts)

    Xmel_power = np.sum(Xmel,axis=1)
    # Normalize and append power
    Xmel_norm = np.zeros((Xmel.shape[0],Xmel.shape[1]+1))
    for tt in range(Xmel_norm.shape[0]):
        if Xmel_power[tt] < pwr_floor: # 0 power and uniform distribution
            Xmel_norm[tt,:Xmel.shape[1]] = 1./nfilts
            Xmel_norm[tt, Xmel.shape[1]] = np.log10(pwr_floor)
        else:
            Xmel_norm[tt,:Xmel.shape[1]] = Xmel[tt,:] / Xmel_power[tt]
            Xmel_norm[tt, Xmel.shape[1]] = np.log10(Xmel_power[tt])

    return Xmel_norm

def prepare_feat_dnn(fileid, outpath, verbose=False):
    with open(fileid) as fp:
        lines = fp.readlines()
    out = []
    count, tot = 0, len(lines)
    for line in lines:
        fpath = os.path.join(CONFIG['EXP_BASEDIR'],'wav/{}.wav'.format(line.strip()))
        count += 1
        if verbose: print("Processing [{}/{}][{}].".format(count,tot,fpath))
        feat =feat_dnn(fpath, verbose=verbose)
        out.append(feat)

    # Write to outpath
    print("Saving [{}] processed feature files to [{}]".format(len(out),outpath))
    np.save(outpath,out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-td',
        help='Prepare training data and fileids for WSJ0', required=False,
        action='store_true',default=False)
    parser.add_argument('-ed',
        help='Prepare test data and fileids for WSJ0', required=False,
        action='store_true',default=False)
    parser.add_argument('-tt',
        help='Prepare training transcripts for WSJ0. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-et',
        help='Prepare test transcripts for WSJ0. REQUIRES existing fileids.',
        required=False,action='store_true',default=False)
    parser.add_argument('-tc',
        help='Check training transcripts/fileids for WSJ0. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-ec',
        help='Check test transcripts/fileids for WSJ0. REQUIRES existing fileids.',
        required=False,action='store_true',default=False)
    parser.add_argument('-ts',
        help='Sort training transcripts/fileids for WSJ0. REQUIRES existing fileids.',
        required=False, action='store_true',default=False)
    parser.add_argument('-es',
        help='Sort test transcripts/fileids for WSJ0. REQUIRES existing fileids.',
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
    parser.add_argument('-phone',
        help='Create phone list from dictionary.',
        required=False,action='store_true',default=False)
    parser.add_argument('-dnnfeat',
        help='Prepare DNN features',
        required=False,action='store_true',default=False)
    parser.add_argument('-v',
        help='Enable verbose', required=False,action='store_true',default=False)
    args = parser.parse_args()

    if args.td:
        prepare_train(convert=CONVERT_AUDIO, verbose=args.v)
    if args.ed:
        prepare_test(convert=CONVERT_AUDIO, verbose=args.v)
    if args.tt:
        prepare_trans('train', verbose=args.v)
    if args.et:
        prepare_trans('test', verbose=args.v)
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
        mdefpath = '{}/model_architecture/wsj0.falign_ci.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        aligndir = '{}/falignout_ci_test'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        # Uncomment for CD-model feature preparation
        #mdefpath = '{}/model_architecture/wsj0.3000.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        #aligndir = '{}/falignout_cd_3000'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        outpath = 'feat_ci_test.npy' # output file
        prepare_feat(aligndir,outpath,verbose=args.v)
    if args.pl:
        mdefpath = '{}/model_architecture/wsj0.falign_ci.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        aligndir = '{}/falignout_ci_test'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        #mdefpath = '{}/model_architecture/wsj0.3000.mdef'.format(CONFIG['EXP_BASEDIR']) # .mdef file used to define HMM
        #aligndir = '{}/falignout_cd_3000'.format(CONFIG['EXP_BASEDIR']) # alignment directory
        outpath = 'label_ci_test.npy' # output file
        prepare_label(aligndir,mdefpath,outpath,verbose=args.v)
    if args.ef:
        outpath = 'feat_test.npy'
        prepare_feat_test(outpath,args.v)

    if args.chartrans:
        inpath = '/home/xyy/local/tutorial/wsj0/etc/wsj0_train.transcription'
        outpath = '/home/xyy/local/airbus-asr/wsj0_train_logmel40_trans.npy'
        prepare_char_transcripts(inpath, outpath, args.v)

    if args.charfeats:
        fileid = '/home/airbus/cmusphinx/wsj0/etc/wsj0_train.fileids'
        outpath = 'wsj0_train_feats.npy'
        prepare_char_feats(fileid, outpath, args.v)

    if args.phone:
        outpath = '/home/xyy/local/tutorial/wsj0/etc/wsj0.phone'
        dicpath = '/home/xyy/local/tutorial/wsj0/etc/cmudict.0.7a'
        dic2phone(dicpath, outpath, args.v)

    if args.dnnfeat:
        fileid = '/home/xyy/local/tutorial/wsj0/etc/wsj0_train.fileids'
        outpath = '/home/xyy/local/airbus-asr/wsj0_train_norm_logmel41.npy'
        prepare_feat_dnn(fileid, outpath, args.v)
