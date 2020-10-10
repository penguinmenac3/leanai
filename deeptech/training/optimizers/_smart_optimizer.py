"""doc
# deeptech.training.optimizers._smart_optimizer

> Automatically create an optimizer with the parameters of the model.
"""

def smart_optimizer(optimizer, *args, **kwargs):
    """
    Convert a pytorch optimizer into a lambda function that expects the config, model and loss as parameters, to instantiate the optimizer with all trainable parameters.

    :param optimizer: A pytorch optimizer that should be made smart.
    :param *args: Any ordered arguments the original optimizer expects.
    :param **kwargs: Any named arguments the original optimizer expects.
    """
    def _join_parameters(model, loss):
        model_params = list(model.parameters())
        loss_params = list(loss.parameters())
        return model_params + loss_params
    return lambda config, model, loss: optimizer(_join_parameters(model, loss), config.training_initial_lr, *args, **kwargs)
