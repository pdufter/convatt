import subprocess
import time
import os
import json
import argparse


def get_ud_metadata(filename):
    infile = json.load(open(filename, "r"))
    metadata = {}
    for line in infile:
        metadata[line['name']] = line
    return metadata


ADD_COMMANDS = ["""python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN" \
     --logging true \
     --ex_id {seed},{lan},sa \
     --log_dir {logdir} \
     --model selfattention \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --seed {seed}""",

                """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+PE[con]" \
     --logging true \
     --ex_id {seed},{lan},sapeconc \
     --log_dir {logdir} \
     --model selfattention \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --positions embeddings \
     --position_embed_dim 64 \
     --positions_mode concatenate \
     --seed {seed}""",

                """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+P" \
     --logging true \
     --ex_id {seed},{lan},sap \
     --log_dir {logdir} \
     --model selfattention \
     --abs_positions_within_attention true \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true  \
     --seed {seed}""",

                """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+R" \
     --logging true \
     --ex_id {seed},{lan},sar \
     --log_dir {logdir} \
     --model selfattention \
     --rel_positions_within_attention true \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true  \
     --seed {seed}""",

                ]

COMMANDS = [

    """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+PE[add]" \
     --logging true \
     --ex_id {seed},{lan},sapeadd \
     --log_dir {logdir} \
     --model selfattention \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --positions embeddings \
     --position_embed_dim 128 \
     --positions_mode add \
     --n_hidden_units 192 \
     --seed {seed}""",

    """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+P+R" \
     --logging true \
     --ex_id {seed},{lan},sapr \
     --log_dir {logdir} \
     --model selfattention \
     --rel_positions_within_attention true \
     --abs_positions_within_attention true \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true  \
     --seed {seed}""",

    """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+PE[add]+Conv2d" \
     --logging true \
     --ex_id {seed},{lan},sapeconv2d \
     --log_dir {logdir} \
     --model selfattention_experimental2d \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --positions embeddings \
     --position_embed_dim 128 \
     --positions_mode add \
     --n_hidden_units 192 \
     --seed {seed}""",

    """python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+PE[add]+Conv" \
     --logging true \
     --ex_id {seed},{lan},sapeconv \
     --log_dir {logdir} \
     --model selfattention_experimental \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --positions embeddings \
     --position_embed_dim 128 \
     --positions_mode add \
     --n_hidden_units 192 \
     --seed {seed}""",

    """
python main.py  \
     --gpu_id {gpu} \
     --comment "{seed},{lan},SAN+PE[add]+Temp" \
     --logging true \
     --ex_id {seed},{lan},satemp \
     --log_dir {logdir} \
     --model selfattention \
     --positions embeddings \
     --data_path {udpath}{lan}/ \
     --is_univ_dep true \
     --positions_mode add \
     --position_embed_dim 128 \
     --weight_normalization true \
     --n_hidden_units 192 \
     --seed {seed}"""
]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=None, type=str, required=True, help="")
    parser.add_argument("--udpath", default=None, type=str, required=True, help="")
    parser.add_argument("--seeds", default="0,1,42", type=str, required=False, help="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", type=str, required=False, help="")
    parser.add_argument("--argument2", action="store_true", help="")
    args = parser.parse_args()

    gpus = {int(i): [] for i in args.gpus.split(",")}
    max_per_gpu = 1

    print("STARTED")

    names = ["UD_Arabic-PADT", "UD_Latin-PROIEL", "UD_Old_French-SRCMF", "UD_French-Spoken", "UD_Portuguese-Bosque", "UD_Bulgarian-BTB", "UD_Estonian-EDT", "UD_French-GSD", "UD_English-GUM", "UD_Afrikaans-AfriBooms", "UD_Korean-Kaist", "UD_Norwegian-Bokmaal", "UD_Czech-FicTree", "UD_Polish-LFG", "UD_Polish-SZ", "UD_Serbian-SET", "UD_Gothic-PROIEL", "UD_Old_Church_Slavonic-PROIEL", "UD_Dutch-Alpino", "UD_Swedish-LinES", "UD_Slovenian-SSJ", "UD_French-Sequoia", "UD_Ukrainian-IU",
             "UD_English-LinES", "UD_Greek-GDT", "UD_Hindi-HDTB", "UD_Finnish-TDT", "UD_Vietnamese-VTB", "UD_Dutch-LassySmall", "UD_Latin-ITTB", "UD_Slovak-SNK", "UD_Hungarian-Szeged", "UD_Swedish-Talbanken", "UD_Persian-Seraji", "UD_Hebrew-HTB", "UD_Chinese-GSD", "UD_Croatian-SET", "UD_German-GSD", "UD_Urdu-UDTB", "UD_Finnish-FTB", "UD_Ancient_Greek-Perseus", "UD_Basque-BDT", "UD_Danish-DDT", "UD_Ancient_Greek-PROIEL", "UD_Norwegian-Nynorsk", "UD_Spanish-AnCora", "UD_Czech-CAC", ]

    names_lower = [x.lower() for x in names]

    # take care of different casing
    names = []
    for file in os.listdir(args.udpath):
        if file.lower() in names_lower:
            names.append(file)

    seeds = [int(seed) for seed in args.seeds.split(",")]

    for seed in seeds:
        for command in COMMANDS:
            for name in names:
                print(name)
                config = {}
                config['lan'] = name
                config['logdir'] = args.logdir
                config['udpath'] = args.udpath
                config['seed'] = seed
                try:
                    found_gpu = False
                    while not found_gpu:
                        for k, v in gpus.items():
                            if len(v) < max_per_gpu:
                                print("... found gpu " + str(k))
                                found_gpu = True
                                config["gpu"] = k
                                proc = subprocess.Popen([command.format(
                                    **config)], shell=True)
                                print(command.format(
                                    **config))
                                gpus[k].append(proc)
                                break
                            # update list of processes
                            gpus[k] = [p for p in v if p.poll() is None]
                        if not found_gpu:
                            time.sleep(30)
                except:
                    print("... failed.")


if __name__ == '__main__':
    main()
