# Requirements
  - Python 2.7.12
  - Tensorflow 1.3.0: `pip install tensorflow==1.3.0`
  - Numpy 1.13.1
  - Scipy 0.17.0
  
# Usages
## download repo
    $ git clone https://github.com/Ha0Tang/GestureGAN
    $ cd ./scripts/evaluation/IS

## launch repo
  - change this [line](https://github.com/Ha0Tang/GestureGAN/blob/db5a420d2a3dce1e7f7b6d1a416f05daa0c6aea8/scripts/evaluation/IS/inception_score.py#L102) according to your case
  - run 
  
    $ python inception_score.py

# References
  - Code is derived from [openai/improved-gan](https://github.com/openai/improved-gan). Thanks all the way.
  
