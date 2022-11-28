torch-model-archiver --model-name BERTForSEQ_en  \
	--version 1.0 --serialized-file ./seq_en/pytorch_model.bin \
	--handler ./deploy/handler_seq.py \
	--extra-files "./seq_en/config.json,./seq_en/setup_confi