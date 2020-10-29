#######################
# PTB

EMBEDDINGS=/mounts/work/philipp/data/fast_text_embeddings/wiki-news-300d-1M-subword.vec
LOGDIR=/mounts/work/philipp/positions/mylogs/
LOGDIRUD=/mounts/work/philipp/positions/mylogsud
UDPATH="/mounts/work/philipp/data/universal_dependencies/ud-treebanks-v2.2/"
mkdir -p $LOGDIR
mkdir -p $LOGDIRUD

SEED=42
GPUID=2


python main.py \
	--gpu_id ${GPUID} \
	--comment "SAN" \
	--logging true \
	--ex_id $SEED,ptb,sa \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+PE[add]" \
	--logging true \
	--ex_id $SEED,ptb,sapeadd \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--positions embeddings \
	--position_embed_dim 300 \
	--positions_mode add \
	--n_hidden_units 364 \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+PE[con]" \
	--logging true \
	--ex_id $SEED,ptb,sapeconc \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--positions embeddings \
	--position_embed_dim 64	\
	--positions_mode concatenate \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+P" \
	--logging true \
	--ex_id $SEED,ptb,sap \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--abs_positions_within_attention true \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+R" \
	--logging true \
	--ex_id $SEED,ptb,sar \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--rel_positions_within_attention true \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+P+R" \
	--logging true \
	--ex_id $SEED,ptb,sapr \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--abs_positions_within_attention true \
	--rel_positions_within_attention true \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+PE[add]+Conv" \
	--logging true \
	--ex_id $SEED,ptb,sapeconv \
	--log_dir $LOGDIR \
	--model selfattention_experimental \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--positions embeddings \
	--position_embed_dim 300 \
	--positions_mode add \
	--n_hidden_units 364 \
	--seed $SEED


python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+PE[add]+Temp" \
	--logging true \
	--ex_id $SEED,ptb,satemp \
	--log_dir $LOGDIR \
	--model selfattention \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--positions embeddings \
	--position_embed_dim 300 \
	--positions_mode add \
	--weight_normalization true \
	--n_hidden_units 364 \
	--seed $SEED

python main.py \
	--gpu_id ${GPUID}  \
	--comment "SAN+PE[add]+Conv2d" \
	--logging true \
	--ex_id $SEED,ptb,sapeconv2d \
	--log_dir $LOGDIR \
	--model selfattention_experimental2d \
	--embed_dim 300 \
	--num_voc 20000 \
	--pretrained_embeddings $EMBEDDINGS \
	--finetune_embeddings false \
	--positions embeddings \
	--position_embed_dim 300 \
	--positions_mode add \
	--n_hidden_units 364 \
	--seed $SEED
done



#######################
# UD

python ud_parallel.py \
--logdir $LOGDIRUD \
--udpath $UDPATH \
--seeds 42 \
--gpus 2,3,4,5,6



