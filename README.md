# LongMIL
WSI analysis with local-global linear attention and extrapolation positional embedding

step1: generate predefined 2-d alibi attention bias matrix by
```
python alibi.py
```

step2: use [CLAM](https://github.com/mahmoodlab/CLAM) or other packages to preprocess your WSIs dataset.

step3: replace their WSI model during training by 
```
LongMIL.py
``` 
