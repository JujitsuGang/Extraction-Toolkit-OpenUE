python setup.py install
python main.py --gpus "0," --max_epochs 1  --num_workers=16 \
    --data_class REDataset \
    --litmodel_class INFERLitModel \
    --model_class Inference \
    --task_name interactive \
    --batch_size 16 \
    --model_name_or_path bert-base-chinese \
    --ner_model_name_or_path output/