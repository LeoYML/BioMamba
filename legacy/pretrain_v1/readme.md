# Pretrain Example

## Environment Setup

```bash
conda env create --file environment.yml
```

## Continue Pretrain on Pubmed

Below is an example of continue pretraining on the PubMed dataset, follow these steps:

1. Preprocess the data:
```bash
python data_preprocess.py
```

2. Run the pretraining script:
```bash
python pretrain_pubmed.py
```

3. Evaluate the pretrained models:
```bash
python models_eval.py
```

