- Build Module Model

- Build Module Dataloader

- Callback
    + Save checkpoint
    + Early stoping
    + Model Sumarize

- Logger
    + mlflow save model artifacts and hypeparameter

- Debug:
    + Use 10% training dataset and 16.6% val dataset
    + Use 8 batches of train and 2 batches of val
    + Show input size and output size for each layer
![Model Sumarize](images/sumarize_model.png "Model Sumarize")

- Understanding your model:
    + Find bottlenecks in your code
    + Logging and Tracking

- Optimization model
    + Simple implement
        ```
            model = ModuleModel()
            model.eval()
            dropout = nn.Dropout()
            dropout.train()
            dropout(model(x))
        ```
    + Drop layers
        ``` 
            model_weights = checkpoint["state_dict"]
            for key in list(model_weights):
                model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)
            model.load_state_dict(model_weights)
            model.eval()
            model(x)

        ```
