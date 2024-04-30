# ArXiv_NLP
NLP Project, to generate titles from abstracts of research papers


To First Download the dataset run

```bash
bash download_dataset.sh
```

After which you can train using:

```python
python3 train_bart.py
```

Or Test trained model using your own custom abstracts:

```python
python3 test_bart.py
```

To try the above please navigate to test_bart.py and enter you abstract into the abstracts list.

