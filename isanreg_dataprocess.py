import argparse
from pyfaidx import Fasta
import os
import errno
import random
import subprocess

def generate_seq(data_dir,args,ref_genome):
    input_file = data_dir+args.input_file
    with open(input_file,'r') as f:
        file = f.readlines()
    fimo_input = data_dir + args.tf_name + "fimo_seq.fna"
    with open(fimo_input, 'w') as output:
        for i in range(len(file)):
            file[i] = file[i].strip().split("\t")[0:3]
            chrom = file[i][0]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            left_nuc, right_nuc  = int(file[i][1]), int(file[i][2])
            if chrom == "chrx":
                chrom = "chrX"
            elif chrom == "chry":
                chrom = "chrY"
            else:
                pass
            input_seq = ref_genome[chrom] [left_nuc:right_nuc].seq
            output.write(">" + chrom + ":" + str(left_nuc) + "-" + str(right_nuc) + "\n" + input_seq + "\n")


def fimo_search(data_dir,args):
    pos_dir = data_dir + args.tf_name + "_fimo_out"
    meme_file = data_dir + args.tf_motif
    seq_file = data_dir + args.tf_name + "fimo_seq.fna"
    command = "fimo --max-strand --oc " + pos_dir + " --parse-genomic-coord " + meme_file + " " + seq_file
    result = subprocess.Popen(command, stdout=None, stderr=None, shell=True)
    result.communicate()
    pos_file = pos_dir + "/fimo.gff"
    neg_dir = data_dir + args.tf_name + "_fimo_ref_genome"
    command = "fimo --max-strand --max-stored-scores 10000000 --oc " + neg_dir + " --parse-genomic-coord " + meme_file + " " + args.ref_fasta
    result = subprocess.Popen(command, stdout=None, stderr=None, shell=True)
    result.communicate()
    neg_file = neg_dir + "/fimo.gff"
    return pos_file, neg_file


def extract_peaks(pos_file, args):
    pos_bins = []
    with open(pos_file,'r') as f:
        file_out = f.readlines()
    file_out = file_out[1:]
    for i in range(len(file_out)):
        data = file_out[i].strip().split("\t")[0:5]
        chrom = data[0]
        if (len(chrom.split("_")) == 1 and chrom != "chrM"):
            middle_nuc = (int(data[3]) + int(data[4])) // 2
            left_nuc = middle_nuc - (args.seq_len // 2)
            right_nuc = middle_nuc + (args.seq_len // 2)
            pos_bins.append([chrom, str(left_nuc), str(right_nuc)])
        else:
            pass
    return pos_bins


#function for converting bed into dict object

def bedto_dict(bed_file,bed_dict):
    with open(bed_file, 'r') as f:
        file_out = f.readlines()
    for i in file_out:
        bed_region = i.strip().split("\t")[0:3]
        if bed_region[0] not in bed_dict:
            bed_dict[bed_region[0]] = []
        bed_dict[bed_region[0]].append([bed_region[1],bed_region[2]])
    return bed_dict


#remove excluded and low mappability regions from rep_list

def exclude_regions(pos_bins, bed_dict):
    input_list = []
    for x in pos_bins:
        chr_name = x[0]
        if chr_name in bed_dict:
            for y in bed_dict[chr_name]:
                if (int(x[1]) < int(y[1]) and int(x[2]) > int(y[0])):
                    break
                else:
                    pass
            else:
                input_list.append(x) 
        else:
            input_list.append(x)  
    return input_list


def combined_dict(bed_dict, input_list):
    for i in input_list:
        if i[0] not in bed_dict:
            bed_dict[i[0]] = []
        bed_dict[i[0]].append([i[1],i[2]])
    return bed_dict


def separate_peaks(input_list):
    train_pos = []
    val_pos = []
    test_pos = []
    for i in input_list:
        chrom = i[0]
        if (chrom == 'chr6' or chrom == 'chr7'):
            test_pos.append(i)
        elif chrom == 'chr8':
            val_pos.append(i)
        else:
            train_pos.append(i)
    return train_pos, val_pos, test_pos

def rev_comp(peak_seq):
    nuc_dict = {"A":"T","T":"A","G":"C","C":"G","[":"]","]":"[","N":"N"}
    rec_seq = ""
    for i in peak_seq[::-1]:
        rec_seq += nuc_dict[i]
    return rec_seq


def create_input(data_dir,train_pos, train_neg, args, ref_genome, data_mode="training"):
    input_bins = data_dir + args.tf_name
    seq_len = args.seq_len
    pos_size = len(train_pos)
    if data_mode == "training":
        out_file = input_bins + "_train.txt"
    elif data_mode == "validation":
        out_file = input_bins + "_val.txt"
    else:
        out_file = input_bins + "_test.txt"
    neg_size = pos_size
    with open(out_file, "w") as output:
        for data in train_pos:
            peak_seq = ref_genome[data[0]] [int(data[1]) : int(data[2])].seq.upper()
            if len(peak_seq) < seq_len:
                diff = seq_len - len(peak_seq)
                add_seq = 'N' * diff
                if diff == 1:
                    peak_seq = add_seq + peak_seq
                else:
                    half_nuc = diff//2
                    peak_seq = add_seq[:half_nuc] + peak_seq
                    peak_seq += add_seq[half_nuc:]
            else:
                pass
            if data_mode == "training":
                rec_seq = rev_comp(peak_seq)
                output.write(rec_seq + "\t" + "1" + "\n")
            else:
                pass
            output.write(peak_seq + "\t" + "1" + "\n")
        random.Random(1).shuffle(train_neg)
        train_neg = train_neg[:neg_size]
        for ele in train_neg:
            neg_seq = ref_genome[ele[0]] [int(ele[1]) : int(ele[2])].seq.upper()
            if len(neg_seq) < seq_len:
                diff = seq_len - len(neg_seq)
                add_seq = 'N' * diff
                if diff == 1:
                    neg_seq = add_seq + neg_seq
                else:
                    half_nuc = diff//2
                    neg_seq = add_seq[:half_nuc] + neg_seq
                    neg_seq += add_seq[half_nuc:]
            else:
                pass
            if data_mode == "training":
                rec_seq = rev_comp(neg_seq)
                output.write(rec_seq + "\t" + "0" + "\n")
            else:
                pass
            output.write(neg_seq + "\t" + "0" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ISANREG Data processing')
    parser.add_argument('--data_dir', default="data_files", help='name of folder having data files')
    parser.add_argument('-r', '--ref_fasta', default = "hg38.fa", help='specify the reference genome fasta file')
    parser.add_argument('--excl_files', default=["GRCh38_unified_blacklist.bed","dukeExcludeRegions.bed"], help='path of bed files having exclusion regions')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking sequence length for each input')
    parser.add_argument('-i','--input_file', default="ESR1_MCF-7_ENCFF138XTJ_.bed", help='path of input file having chip_seq peak regions')
    parser.add_argument('--tf_name', default="ESR1", help='specify the name of the TF under study')
    parser.add_argument('--tf_motif', default="ESR1_MA0112.1.meme", help='specify the TF motif meme file')
    args = parser.parse_args()
    
    data_dir = args.data_dir + "/"
    
    #raises exception if the input file does not exist
    
    if not os.path.isfile(data_dir+args.input_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),args.input_file)
    
    ref_fasta = args.ref_fasta
    
    #raises exception if the reference genome fasta file does not exist
    if not os.path.isfile(data_dir+ref_fasta):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ref_fasta)
        
    #raises exception if the meme file does not exist
    if not os.path.isfile(data_dir+args.tf_motif):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.tf_motif)
    
    print("Saving reference genome to variable...")
    ref_genome = Fasta(data_dir+ref_fasta)
    
    print("Writing fasta sequences to file...")
    generate_seq(data_dir,args,ref_genome)
    
    print("Running FIMO for motif scanning")
    pos_file, neg_file = fimo_search(data_dir,args)
    
    print("Saving positive input data into list...")
    pos_bins = extract_peaks(pos_file, args)
    
    print("Saving negative input data into list...")
    neg_bins = extract_peaks(neg_file, args)
    
    #load bed file of excluded regions
    bed_dict = {}
    for bed_path in args.excl_files:
        bed_dict = bedto_dict(data_dir+bed_path,bed_dict)
    
    print("Removing excluded regions from input positive regions...")
    input_list = exclude_regions(pos_bins, bed_dict)
    
    print("Creating combined dict of positive sample and exclusion regions...")
    bed_dict = combined_dict(bed_dict, input_list)
    
    print("Filtering negative input data...")
    #Filters excluded and positive sample regions
    neg_list = exclude_regions(neg_bins, bed_dict)
     
    print("Extracting separate datasets...")
    train_pos, val_pos, test_pos = separate_peaks(input_list)
    train_neg, val_neg, test_neg = separate_peaks(neg_list)
    
    print("Writing training data to file...")
    create_input(data_dir,train_pos, train_neg, args, ref_genome, data_mode="training")
    
    print("Writing validation data to file...")
    create_input(data_dir,val_pos, val_neg, args, ref_genome, data_mode="validation")
    
    print("Writing testing data to file...")
    create_input(data_dir,test_pos, test_neg, args, ref_genome, data_mode="testing")

    
