
from argparse import ArgumentParser
import pandas as pd
import os
import subprocess
import concurrent.futures

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"Command: {cmd}\nOutput: {result.stdout}\nError: {result.stderr}"
    except Exception as e:
        return f"Error executing {cmd}: {str(e)}"
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_list_file", type=str, default="../data/train_50_indivs.txt",
                        help="The sample list to be used for consensus generation.")

    parser.add_argument("--reference",type=str,default="../data/hs37d5.fa",help="")
    parser.add_argument("--vcf_root",type=str,default="../data/vcf_files",help="")
    parser.add_argument("--consensus_root",type=str,default="../data/consensus",help="")



    args = parser.parse_args()
    if(not os.path.exists(args.consensus_root)):
        os.mkdir(args.consensus_root)

    all_samples = pd.read_csv(args.sample_list_file, names=["samples"])
    #print(all_samples)

    all_cmds = []
    vcf_root = args.vcf_root


    # generate consensus sequences for each sample and chromosome
    for cur_sample_name in all_samples["samples"]:
        
        for chr_num in range(1,23):
            consensus_cmd = f"samtools faidx {args.reference} {chr_num} | \
        bcftools consensus -s {cur_sample_name} -H 1pIu {args.vcf_root}/GEUVADIS.chr{chr_num}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz\
        -i \"type='snp'\" > \"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_1_forward.fa\""
            
            consensus_cmd_2 = f"samtools faidx {args.reference} {chr_num} | \
        bcftools consensus -s {cur_sample_name} -H 2pIu {args.vcf_root}/GEUVADIS.chr{chr_num}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz\
        -i \"type='snp'\" > \"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_2_forward.fa\""

            all_cmds.append(consensus_cmd)
            all_cmds.append(consensus_cmd_2)


    print("\n".join(all_cmds))


    
    # the these commands in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_command, all_cmds))
     
    for res in results:
        print(res)
    

    cat_cmds = []
    for cur_sample_name in all_samples["samples"]:
        save_sample_fn = f'"{args.consensus_root}/{cur_sample_name}_allele_1_forward.fa"'
        save_sample_fn_2 = f'"{args.consensus_root}/{cur_sample_name}_allele_2_forward.fa"'
        cur_cmd = ""

        for chr_num in range(1,23):

            consensus_cmd = f"cat \"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_1_forward.fa\" >> {save_sample_fn};"
            print(consensus_cmd)
            os.system(consensus_cmd)

            os.remove(f"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_1_forward.fa")
            
         
            consensus_cmd_2 = f"cat \"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_2_forward.fa\" >> {save_sample_fn_2};"
            print(consensus_cmd_2)
            os.system(consensus_cmd_2)
            # remove the intermediate file
            os.remove(f"{args.consensus_root}/{cur_sample_name}_chr{chr_num}_allele_2_forward.fa")
            



    