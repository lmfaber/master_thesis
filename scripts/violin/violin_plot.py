import os
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib 
def subfolders(path):
    """ Returns all first level directories in a given directory.
    
    Arguments:
        path {str} -- Path of the directory.
    """
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            yield(name)

def create_length_dict(assembly):
    return [(key, len(entry.seq)) for key, entry in assembly.items()]

def sort_by_length(assembly):
    return sorted(assembly, key=lambda x: x[1])

def get_all_sequence_lengths(assembly):
    return [length for seq, length in assembly]



for folder in subfolders('.'):
    plt.clf()
    print('===================')
    print(folder)
    all_all_lengths = []
    plt.xticks(rotation=45, ha='right')		# Rotate the xaxis at 45 degree

    if folder == 'eco_SRR4255368' or folder == 'cel_ERR2886543':
        file_names = ['rnaSPAdes.fasta', 'SOAPdenovo-Trans.fasta', 'Trinity.fasta', 'combined.fasta', 'cd-hit-est-100.fasta', 'cd-hit-est-90.fasta', 'Linclust.fasta', 'MeShClust.fasta', 'MeShClust2.fasta', 'Grouper.fasta', 'Grouper_p.fasta', 'karma.fasta', 'karma_p.fasta']
        real_names = ['rnaSPAdes', 'SOAPdenovo-Trans', 'Trinity', 'Combined', 'cd-hit-est-100', 'cd-hit-est-90', 'Linclust', 'MeShClust', 'MeShClust2', 'Grouper', 'Grouper*', 'karma', 'karma*']
        a, b, c, *_ = sns.color_palette()
        pal = [a, a, a, b, c,c,c,c,c,c,c,c,c]
    else:
        file_names = ['rnaSPAdes.fasta', 'SOAPdenovo-Trans.fasta', 'Trinity.fasta', 'combined.fasta', 'cd-hit-est-100.fasta', 'cd-hit-est-90.fasta', 'Linclust.fasta', 'MeShClust.fasta', 'MeShClust2.fasta', 'Grouper.fasta', 'Grouper_p.fasta', 'karma.fasta', 'karma_p.fasta', 'OysterRiverProtocol.fasta']
        real_names = ['rnaSPAdes', 'SOAPdenovo-Trans', 'Trinity', 'Combined', 'cd-hit-est-100', 'cd-hit-est-90', 'Linclust', 'MeShClust', 'MeShClust2', 'Grouper', 'Grouper*', 'karma', 'karma*', 'OysterRiverProtocol']
        a, b, c, *_ = sns.color_palette()
        pal = [a, a, a, b, c,c,c,c,c,c,c,c,c,c]    
    for f in file_names:
        assembly_file = f"{folder}/{f}"

        # Open file and read all sequences as dict
        with open(assembly_file, 'r') as fastaReader:
            original_fasta_sequences = SeqIO.to_dict(SeqIO.parse(fastaReader, 'fasta'))

        len_dict = create_length_dict(original_fasta_sequences)
        len_dict = sort_by_length(len_dict)

        number_of_sequences = len(original_fasta_sequences)
        shortest_seq = len_dict[0][1]
        longest_seq = len_dict[-1][1]
        all_sequence_length = get_all_sequence_lengths(len_dict)
        all_sequence_length = [math.log10(x) for x in all_sequence_length]
        # print(all_sequence_length)
        all_all_lengths.append(all_sequence_length)
        print(f"{assembly_file}\t\t{number_of_sequences}\t{shortest_seq}\t{longest_seq}")
        # sns.distplot(all_sequence_length, bins=250, rug=True, hist=True)
        # plt.show()

    data = pd.DataFrame.from_dict({a: b for a,b in zip(real_names, all_all_lengths)}, orient='index' ).transpose()

    # # print(pal)


    # sns.violinplot(data=all_all_lengths, orient='v', cut=0, scalue_hue=True).set_title(folder)
    sns.violinplot(data=data, orient='v', scale='width', inner='quartile', palette=pal)
    ax = plt.gca()
    ax.set_ylabel('log10(sequence length)')
    plt.tight_layout()
    # sns.boxplot(data=all_all_lengths)
    plt.savefig(f"{folder}.svg", format='svg')


    # print(number_of_sequences, shortest_seq, longest_seq)


