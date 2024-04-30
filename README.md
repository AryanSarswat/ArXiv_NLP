# ArXiv_NLP
NLP Project, to generate titles from abstracts of research papers


To First Download the dataset run

```bash
bash download_dataset.sh
```

[RNN]

Navigate to the respective Jupyeter Notebooks in the baselineRNN folders.

```
python3 baselineRNN/abstract_to_title_vanilla_rnn.py
```

[BART]

After which you can train using:

```python
python3 train_bart.py
```

Or Test trained model using your own custom abstracts:

```python
python3 test_bart.py
```

To try the above please navigate to test_bart.py and enter your abstract into the abstracts list.

