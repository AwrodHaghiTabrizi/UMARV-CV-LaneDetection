# Working With Models

In each model folder, there are 3 sub folders.
- The content/ folder contains some info for the model and performance metrics
- The docs/ folder contains any documentation for the model
- The src/ folder contains all of the code related to the model.

In the src/ folder, there are multiply .py files which include functinonal classes and methods. In the src/notebooks/ folder, there are all of the jupyter notebook environments that can be worked out of. The classes and methods defined in the .py files are called in these jupyter notebook files. An example of this dynamic is the methods.py file would include the methods to train a model, and then those methods would be called in the jupyter notebook to actually train the model.

You have full freedom to add/edit/delete any code in any of these files for your own personal model. If there is another model you wish to start from, istead of creating a new model, you can create a copy of that model running the "create_copy_of_model.py" script

## Creating Models

1. Navigate to src/scripts
2. Right click on either "create_model.py" or "create_copy_of_model.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal