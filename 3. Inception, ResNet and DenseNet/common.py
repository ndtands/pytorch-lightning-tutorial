from googlenet import GoogleNet
model_dict={
    'GoogleNet': GoogleNet,
}
def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](model_hparams)
    else:
        assert False, f"Unkown model name '{model_name}'. Available models are: {str(model_dict.keys())}"