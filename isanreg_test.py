import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import os
import argparse
import re

#numpy version > 1.20.0 is needed
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import fisher_exact

from isanreg_train import *


def test_run(batch_test, args):
    output_array = [int(i[1]) for i in batch_test]
    y_test = tf.stack(output_array, axis=0)
    y_test = tf.expand_dims(y_test, axis=-1)
    seq_data = [i[0] for i in batch_test]
    input_array = vectorization(seq_data)
    x_test = tf.stack(input_array)
    y_pred, att_weight = test_model(input_data=x_test, attr=True) 
    return y_test, y_pred, att_weight, x_test, seq_data

def top_motifs(attr_tensor, top_num, window_len):
    window_tensor = sliding_window_view(attr_tensor, window_len)
    win_avg = window_tensor.mean(axis=-1)
    #extract top k values and their indices from attribution score windows
    _, indices = tf.math.top_k(win_avg, k=top_num)
    sorted_attridx = indices.numpy().tolist()
    return sorted_attridx

def model_testing(test_peaks, args):
    batch_size = args.batch_size
    m = args.m
    top_dict = {}
    rest_dict = {}
    test_steps = len(test_peaks) // batch_size
    acc = tf.keras.metrics.BinaryAccuracy()
    core_len = args.core_len
    mid_nuc = args.seq_len//2
    left_len = -(-core_len//2)
    left_core = mid_nuc - (left_len-1)
    right_len = core_len - left_len
    right_core = mid_nuc + (right_len+1)
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
            batch_seq = seq_data[i]
            if (label_test == 1 and batch_pred > 0.7):
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
                #sliding window length
                window_len = args.window_len
                win_num = args.seq_len-(window_len-1)
                sorted_attridx = top_motifs(k_tensor, win_num, window_len)
                #top 1% indices are considered
                top_num = (win_num*1)//100
                seq_num = 0
                #loop through the top scoring sorted nucleotide windows
                for index_num in sorted_attridx:
                    if index_num in range(left_core-(window_len-1),right_core):
                        pass
                    else:
                        win_seq = batch_seq[index_num:index_num+window_len]
                        if seq_num < top_num:
                            if win_seq not in top_dict:
                                top_dict[win_seq] = 1
                            else:
                                top_dict[win_seq] += 1
                        else:
                            if win_seq not in rest_dict:
                                rest_dict[win_seq] = 1
                            else:
                                rest_dict[win_seq] += 1
                        seq_num += 1
    top_total = sum(top_dict.values())
    num_top = len(top_dict)
    rest_total = sum(rest_dict.values())
    res_dict = {}
    for x,y in top_dict.items():
        if x not in rest_dict:
            rest_value = 0
        else:
            rest_value = rest_dict[x]
        top_others = top_total - y
        rest_others = rest_total - rest_value
        oddsratio, pvalue = fisher_exact([[y , rest_value], [top_others , rest_others]], alternative='greater')
        res_dict[x+":"+str(pvalue)] = oddsratio
    top_sort = sorted(res_dict.items(), key=lambda motif: motif[1], reverse=True)
    output_path = os.getcwd() + "/" +  args.model_dir + "/" + args.tf_name + "/" + args.tf_name + "_out.txt"
    with open(output_path, "w") as file_out:
        file_out.write("k-mer" + "\t" + "odds_ratio" + "\t" + "p_value" + "\t" + "corrected_p_value" + "\n")
        for k_mer in top_sort:
            mer_name, p_value = k_mer[0].split(":")
            adj_pval = float(p_value) * num_top
            file_out.write(mer_name + "\t" + str(k_mer[1]) + "\t" + p_value + "\t" + str(adj_pval) + "\n")
    test_acc = acc.result().numpy()
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ISANREG Testing')
    parser.add_argument("--epoch_num", type=int, help="epoch number of weights file having minimum validation loss")
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding')
    parser.add_argument('--batch_size', type=int, default=20, help='specify the batch_size needed for testing')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input') 
    parser.add_argument('--block_num', type=int, default=200, help='number of blocks into which the input sequence should be split, seq_len should be divisible by block_num')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('-att', '--attention', dest = "attention", default = "multi_head", help='specify the Intra-block and Inter-block self-attention method, "dot_product" can be used to reduce computation time')
    parser.add_argument('-m', '--m', dest = "m",type=int, default=20, help='number of approximation steps in calculating integrated gradients')
    parser.add_argument('--model_dir', dest = "model_dir", default = "Models", help='specify the output folder name for saving the model')
    parser.add_argument('--data_dir', default="data_files", help='name of folder having data files')
    parser.add_argument('--window_len', type=int, default=5, help='specify the length of sliding window')
    parser.add_argument('--core_len', type=int, default=18, help='specify the core motif length of the TF under study')
    parser.add_argument('--tf_name', default="ESR1", help='specify the name of the TF under study')
    args, unknown = parser.parse_known_args()
    
    #raises exception if seq_len is not divisible by block_num
    
    if args.seq_len % args.block_num != 0:
        raise ValueError("seq_len is not divisible by block_num")   
    
    test_file = args.data_dir + "/" + args.tf_name + "_test.txt"
    print("Extracting testing data from file...")
    test_peaks = extract_peaks(test_file)
    
    print("Initializing the model...")
    test_model = Modelsubclass(args)
    
    #create one testing batch for running model once
    x_single, _ = one_train_batch(args, test_peaks)
    
    test_model(input_data=x_single)
    
    print("Loading weights from file...")
    model_dir = args.model_dir
    dir_path = weights_filepath(model_dir+"/"+args.tf_name)
    file_names = os.listdir(dir_path)
    idx_files = []
    for names in file_names:
        if names.endswith(".index"):
            idx_files.append(names)
    
    for idx in idx_files:
        if int(re.search('weights.(.+?)-', idx).group(1)) == args.epoch_num:
            weights_file = idx[0:-6]
    
    weights_path = dir_path + "/" + weights_file
    status = test_model.load_weights(weights_path).expect_partial() 
    
    print("Asserting matching of weights...")
    status.assert_existing_objects_matched()
    
    print("Testing model...")
    test_acc = model_testing(test_peaks, args)
    
    print('Final Test accuracy : ', test_acc)
    test_metrics = os.getcwd() + "/" +  args.model_dir + "/" + args.tf_name + "/" + args.tf_name + "_metric.txt"
    with open(test_metrics, "w") as file_out:
        file_out.write("Testing_Accuracy:" + "\t" + str(test_acc) + "\n")
        
