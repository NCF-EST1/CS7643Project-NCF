# CS7643Project-NCF

Usage:
python main.py -d [movielens/lastfm] -m [NCF/GMF/NeuMF] -c [config.yaml] -s [threshold]

Note: -s is used for development purpose which create a subset of provided dataset using len(dataset) * threshold number of \
instances. Threshold value is between 0.0 and 1.0.

Exp Usage:
python main.py -d movielens -m NCF -c configs\ncf.yaml -s 0.5

Repo architecture:
- main.py

- /models
    - NCF.py
    - GMF.py
    - NeuMF.py

- /configs

- /data
    - process_data.py
    - trainloader.py

- saved_models
    Contain saved model .pth, generated plots, saved pickle incldue loss/hr/ndcg values.

- rq1.py
- rq2.py
- rq3.py
- rq4.py



