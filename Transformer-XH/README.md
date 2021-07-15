# Transformer-XH
The source code is a modification of the source code provided for  the paper "Transformer-XH: Multi-evidence Reasoning with Extra Hop Attention (ICLR 2020)".
The original code can be found on: https://github.com/microsoft/Transformer-XH

# Dependency Installation
First, Run python setup.py develop to install required dependencies for transformer-xh.  
Optionally you can install apex (for distributed training) following official documentation [here](https://github.com/NVIDIA/apex).

You can find politiop and liar plus datasets in the data folder here.  
In order to train the models on FEVER, you can find instructions on how to get it in the Transformer-XH repo linked above.  
Note that the pre-trained Transformer-XH available in that repo will not work with our code, since we've made some modifications to the architecture.

## Model Training and Evaluation
All the data should be placed in data directory.  
You can train and evaluate a model from this folder by running transformer-xh/main.py. You can specify different arguments,  
such as model name, number of hops, dataset. Note that in case of evaluation the number of hops specified must be the same as in the loaded model.  
Some of the most relevant arguments are described below:

--out_model (default=None): Name of the trained model (will be saved with that name). Used for training only.  
--in_model (default=None): Path to the model to load for training/eval. Required for eval. Optional for training.  
--epochs (default=2): Number of training epochs  
--checkpoint (default=1000): Number of training set examples to evaluate and save the model after.  
                    
--test(default=False): Whether to evaluate an existing model or train a new one. Don't use if you want to train  
--arch(default=trans\_xh): Model architecture. Either trans\_xh or bert.  
--dataset(default='politihop): Dataset to use in training/eval. One of: politihop, liar, fever, politihop\_all, liar\_all, adv  
--hops(default=3): Number of hops of the trained or loaded model.

For instance this command:

python transformer-xh/main.py --cf transformer-xh/configs/config\_fever.json --out\_model trans3.pt --checkpoint 300 --epochs 4 --hops 3 --dataset adv

Will train a tranformer\_xh 3-hop model for 4 epochs on the adversarial dataset, with a checkpoint every 300 training examples. It will use transformer-xh/configs/config\_fever.json and save it as transformer-xh/experiments/trans3.pt file.
For more details on the available arguments, see transformer-xh/main.py

After evaluating a model you can run results_model.sh to get more detailed results. The script is called like this:

bash results_model.sh model dataset [all|adv]

where the last argument is optional. If it's not provided, then the even_split version of the dataset is used.
If all or adv is provided then the full article or adversarial version of the dataset is used, respectively.  
For instance:
bash results_model.sh trans3.pt politihop all  
Will yield more detailed results of evaluating tran3.pt on politihop dataset with "all" settings (whole articles).  
NOTE: You have to have the outputs from evaluating the same model on the same dataset (the outputs are saved in outputs folder).
