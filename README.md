# Diagnose Alzheimer

## Enviroment Installation

If you use conda, all you need to do is
````
conda env create -f conda.yaml && conda activate alzai
````

Or, if you don't prefer to use conda, then you need to create an enviroment with pip and if you use unix based operating system, you need to run:
````
# On unix based systems
python -m pip install virtualenv
python -m venv venv
source venv/bin/activate
`````
If you use windows as an operating system, you need to run:
````
# On windows
py -m pip install virtualenv
py -m venv venv
.\venv\Scripts\activate
````

and then, you need to install required packages with
````
pip install -r requirements.txt
````

# Data Loading

We use Data Version Control System because we don't want to include data in our github repository. If you need data for your job, you need to run:

````
dvc pull -r adni
`````
and you will see that dvc gives you an url and wants token from you. Copy that url and paste it into browser tab and go that url. Give permission with your account to dvc and google will give you a token. Take that token and paste it into terminal, and go on. Data will be downloading and you are ready to do your job.
