v1.0.0: 

First Major Update.

Implemented UNet from Initial_experiments into the current project framework.

1. Introduced scheduler in training.py to update training rates through each epoch. Needs more experimentation to make it work.
2. Transforms are in pre_processing.py. Grayscale is now implemented with transforms.Grayscale(). Ideally we can include all colorspace/normalizations we want to try from here 
3. Validation losses are now calculated as well for each epoch. Train and Test datasets will have to defined clearly. Validation shows that Unet cannot guess actual colors if it has not seen it before but colorizing the pixels that may be similar in color with similar values.
4. Added a few Loss functions in criterion. Not yet tested out others beyond MSE.  

Next sets of training with COCO datasets will be performed and loss function combinations with/without.

More models can be added directly to models.py for experimentation. make sure to update setup_model as well if you add a new model.

Note:
CONFIG is a dictionary which will contain information for loss function, scheduler, model etc.
Ideally, this can be modified to expand different color spaces, models etc for keeping track.
Models are saved as *ModelName*_timesamp.pth. 
CONFIG and loss curves for each traning also are stored in the checkpoint file. So, you can deal with it accordingly when you do you analysis (see analysis.ipynb for the basic UNet example)


Cheers,
Sushruth.
