python setup.py install
python main.py --gpus "0," --max_epochs 5 --num_workers 0 \
    --data_class REDataset \
    --litmodel_class SEQLitModel \
    --model_class B