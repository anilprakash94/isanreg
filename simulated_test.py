import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import os
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
from simulated_train import *

def test_run(batch_test, args):
    output_array = [int(i[1]) for i in batch_test]
    y_test = tf.stack(output_array, axis=0)
    y_test = tf.expand_dims(y_test, axis=-1)
    seq_data = [i[0] for i in batch_test]
    input_array = vectorization(seq_data)
    x_test = tf.stack(input_array)
    y_pred, att_weight = test_model(input_data=x_test, attr=True) 
    return y_test, y_pred, att_weight, x_test, seq_data


def plot_tensor(k_tensor, motif1_idx, motif2_idx, sample_num, args):
    left_nuc = min(motif1_idx[0],motif2_idx[0])
    right_nuc = max(motif1_idx[1],motif2_idx[1])
    nuc_gap = right_nuc - left_nuc
    if left_nuc < 10:
        x = np.arange(0,right_nuc+10)
        plot_attr = k_tensor[0:right_nuc+10]
    elif (args.seq_len - right_nuc) < 10:
        x = np.arange(left_nuc-10,args.seq_len)
        plot_attr = k_tensor[left_nuc-10:args.seq_len]
    else:
        x = np.arange(left_nuc-10,right_nuc+10)
        plot_attr = k_tensor[left_nuc-10:right_nuc+10]
 
    f = plt.figure()
    f.set_figwidth(25)
    f.set_figheight(15)
    
    markerline, stemlines, baseline = plt.stem(x, plot_attr)
    plt.setp(stemlines, linestyle="-", color="magenta", linewidth=0.5 )
    plt.setp(markerline, marker='o', markersize=5, markerfacecolor = "cornflowerblue", markeredgecolor="blue", markeredgewidth=0.5)
    plt.setp(baseline, linestyle="-", color="red", linewidth=2)
    
    plt.xlabel('Position in' + ' ' + str(args.seq_len) + 'bp input sequence', labelpad=15,fontsize=25,weight='bold')
    plt.ylabel('Attribution_score',labelpad=15,fontsize=25,weight='bold')
    
    plot_title = "Attention-Attribution plot" +"\n"
    plt.title(plot_title, fontsize=35,weight='bold')
    tick_step = (nuc_gap * 10) // 100
    plt.xticks(np.arange(min(x), max(x), tick_step), fontsize=21)
    plt.yticks(fontsize=21)
    plot_name = weights_filepath(args.model_dir+"/simulated/images/")
    plot_name += str(sample_num) + ".png"    
    
    min_value = np.min(plot_attr)
    max_value = np.max(plot_attr)
    
    max_ylim = max_value + 0.15
   
    plt.ylim([min_value, max_ylim])
        
    m1_values = k_tensor[motif1_idx[0]:motif1_idx[1]]
    m1_indices = np.arange(motif1_idx[0],motif1_idx[1])
    
    plt.plot(m1_indices, m1_values, color='green', linestyle='dashed', linewidth=3, label='CTCT-Motif') 
    plt.legend(loc="upper right", prop={'size': 25, 'weight': 'bold'})
    
    m2_values = k_tensor[motif2_idx[0]:motif2_idx[1]]
    m2_indices = np.arange(motif2_idx[0],motif2_idx[1])
    plt.plot(m2_indices, m2_values, color='red', linestyle='dashed', linewidth=3, label='E2F1-Motif')
    plt.legend(loc="upper right", prop={'size': 25, 'weight': 'bold'})
    plt.savefig(plot_name,dpi=400)


def find_inter(output, top_attridx, k_tensor, batch_seq, true_pos, inter_pos, args, sample_num, left_nuc, right_nuc, num_tf1=0, num_tf2=0, attr_ind1=None, attr_ind2=None):
    motif1_idx , motif2_idx = None, None
    for index_num in top_attridx:
        if (num_tf1 >= 1 and num_tf2 >= 1):
            inter_pos += 1
            #print('Percent of TF motifs found : ', (inter_pos / (true_pos)) * 100)
            break
        if attr_ind1 != None:
            if (index_num > attr_ind1-15 and index_num < attr_ind2+15):
                adj_idx = True
            elif (num_tf2 == 1 and index_num > left_nuc and index_num < right_nuc):
                adj_idx = True
            else:
                adj_idx = False
        else:
            adj_idx = False
        if adj_idx == True:
            pass
        else:
            if num_tf1 == 0:
                motif1 = args.motifs[0]
                motif1_len = len(motif1)
                flank_left = index_num - (motif1_len-1)
                flank_right = index_num + (motif1_len)
                attr_seq = batch_seq[flank_left:flank_right]
                regex_res = re.search(motif1,attr_seq)
                if bool(regex_res):
                    match_seq = regex_res.group(0)
                    match_span = regex_res.span(0)
                    attr_ind1 = flank_left+match_span[0]
                    attr_ind2 = flank_left+match_span[1]
                    mot_attr = k_tensor[attr_ind1:attr_ind2]
                    mot_attr = list(np.around(np.array(mot_attr),2))
                    num_tf1 += 1
                    #print("Success : TF1 found")
                    #print("TF1 position index:",(attr_ind1,attr_ind2))
                    motif1_idx = (attr_ind1,attr_ind2)
                    output.write(str(sample_num) + "\t" + "Motif-1" + "\t" + str(attr_ind1) + "-" + str(attr_ind2) + "\t" + str(mot_attr) + "\t" + match_seq + "\n")          
            if num_tf2 == 0:
                motif2 = args.motifs[1]
                motif2_len = len(motif2)
                flank_left = index_num - (motif2_len-1)
                flank_right = index_num + (motif2_len)
                attr_seq = batch_seq[flank_left:flank_right]
                regex_res = re.search(motif2,attr_seq)
                if bool(regex_res):
                    match_seq = regex_res.group(0)
                    match_span = regex_res.span(0)
                    attr_ind1 = flank_left+match_span[0]
                    attr_ind2 = flank_left+match_span[1]
                    mot_attr = k_tensor[attr_ind1:attr_ind2]
                    mot_attr = list(np.around(np.array(mot_attr),2))
                    num_tf2 += 1
                    #print("Success : TF2 found")
                    #print("TF2 position index:",(attr_ind1,attr_ind2))
                    motif2_idx = (attr_ind1,attr_ind2)
                    output.write(str(sample_num) + "\t" + "Motif-2" + "\t" + str(attr_ind1) + "-" + str(attr_ind2) + "\t" + str(mot_attr) + "\t" + match_seq + "\n")    
    return inter_pos, motif1_idx, motif2_idx


def model_testing(test_peaks, args):
    batch_size = args.batch_size
    m = args.m
    top_len = sum([len(re.sub(r"\[.*?\]", r"N", mot)) for mot in args.motifs])
    left_nuc = (args.seq_len//2) - 100
    right_nuc = (args.seq_len//2) + 100
    test_steps = len(test_peaks) // batch_size
    acc = tf.keras.metrics.BinaryAccuracy()
    plot_num = 0
    output_path = args.model_dir+"/simulated/"+args.attr_output
    with open(output_path, "w") as output:
        output.write("Sample_num" + "\t" + "TF_motif" + "\t" + "Motif_position" + "\t" + "Attribution_score" + "\t" + "Motif-seq" + "\n")
        true_pos = 0
        inter_pos = 0
        for sample in range(test_steps):
            batch_test = test_peaks[sample*batch_size : (sample+1)*batch_size]
            y_test, y_pred, att_weight, x_test, seq_data = test_run(batch_test, args)
            acc.update_state(y_test, y_pred)
            #print('Test accuracy : ', acc.result().numpy())
            for i in range(K.shape(x_test)[0]):
                sample_num = (sample * batch_size) + i
                #print(f'Running attribution sample no.: {sample_num}')
                input_test = x_test[i:i+1]
                label_test = y_test[i:i+1]
                batch_pred = y_pred[i:i+1]
                batch_att = att_weight[i:i+1]
                batch_seq = seq_data[i:i+1][0]
                if (label_test == 1 and batch_pred > 0.7):
                    true_pos += 1
                    grad_tensor = tf.zeros(K.shape(batch_att), dtype=tf.dtypes.float32)
                    for k in range(m):
                        baseline = tf.zeros(K.shape(batch_att), dtype=tf.dtypes.float32)
                        input_att = baseline + ( (k/m) * (batch_att - baseline) )
                        with tf.GradientTape() as tape:
                            tape.watch(input_att)
                            pos_pred = test_model(input_test, attr= None, att_weight = input_att) 
                            grad_tensor = tf.math.add(grad_tensor, tape.gradient(pos_pred, input_att))
                    attr_tensor = tf.math.multiply(grad_tensor, (batch_att - baseline) / m )
                    #sum of top 10 feature attention attribution score
                    attr_k, _ = tf.math.top_k(attr_tensor, k=10)
                    k_tensor = tf.math.reduce_sum(attr_k, axis =-1)
                    k_tensor = tf.squeeze(k_tensor).numpy()
                    #top k indices (k = total length of motifs)
                    _, indices = tf.math.top_k(k_tensor, k=top_len)
                    top_attridx = indices.numpy().tolist()
                    k_tensor = k_tensor.tolist()
                    k_tensor = np.cbrt(k_tensor)
                    inter_pos, motif1_idx, motif2_idx = find_inter(output, top_attridx, k_tensor, batch_seq, true_pos, inter_pos, args, sample_num, left_nuc, right_nuc)
                    if (motif1_idx and motif2_idx) != None and plot_num < 5:
                        plot_tensor(k_tensor, motif1_idx, motif2_idx, sample_num, args)
                        plot_num += 1
                else:
                    pass
    test_acc = acc.result().numpy()
    inter_percent = (inter_pos / (true_pos)) * 100
    return test_acc, inter_percent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulated Testing')
    parser.add_argument("--epoch_num", type=int, help="epoch number of weights file having minimum validation loss")
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding')
    parser.add_argument('--batch_size', type=int, default=20, help='specify the batch_size needed for testing')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input') 
    parser.add_argument('--block_num', type=int, default=200, help='number of blocks into which the input sequence should be split, seq_len should be divisible by block_num')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('--dir', dest = "model_dir", default = "Models", help='specify the output folder name')
    parser.add_argument('-att', '--attention', dest = "attention", default = "multi_head", help='specify the Intra-block and Inter-block self-attention method, "dot_product" can be used to reduce computation time')
    parser.add_argument('-m', '--m', dest = "m",type=int, default=20, help='number of approximation steps in calculating integrated gradients')
    parser.add_argument('-attr', '--attr_output', dest = "attr_output" , default="simtest_out.txt", help='specify output text file for writing attribution results')
    parser.add_argument('--test_file', default = "data_files/simulated_test2k.txt", help='specify the input data for testing')
    parser.add_argument('--motifs', default = ["AGCGCCACCTAGTGGTA","ATTGGCGCCAAA"] , help='specify the sequence of motifs as a list')
    args, unknown = parser.parse_known_args()
    
    #raises exception if seq_len is not divisible by block_num
    
    if args.seq_len % args.block_num != 0:
        raise ValueError("seq_len is not divisible by block_num")   
    
    print("Extracting testing data from file...")
    test_peaks = extract_peaks(args.test_file)
    
    print("Initializing the model...")
    test_model = Modelsubclass(args)
    
    #create one testing batch for running model once
    x_single, _ = one_train_batch(args, test_peaks)
    
    test_model(input_data=x_single)
    
    print("Loading weights from file...")
    model_dir = args.model_dir
    dir_path = weights_filepath(model_dir+"/simulated")
    file_names = os.listdir(dir_path)
    idx_files = []
    for names in file_names:
        if names.endswith(".index"):
            idx_files.append(names)
    
    for idx in idx_files:
        if int(re.search('weights.(.+?)-', idx).group(1)) == args.epoch_num:
            weights_file = idx[0:-6]
    
    weights_path = model_dir + "/simulated/" + weights_file
    status = test_model.load_weights(weights_path).expect_partial() 
    
    print("Asserting matching of weights...")
    status.assert_existing_objects_matched()
    
    print("Testing model...")
    test_acc, inter_percent = model_testing(test_peaks, args)
    
    print('Final Test accuracy : ', test_acc)
    
    print('Percent of TF motifs found : ', inter_percent)
    
