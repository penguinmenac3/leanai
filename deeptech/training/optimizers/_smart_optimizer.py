def smart_optimizer(optimizer, *args, **kwargs):
    def _join_parameters(model, loss):
        model_params = list(model.parameters())
        loss_params = list(loss.parameters())
        return model_params + loss_params
    return lambda config, model, loss: optimizer(_join_parameters(model, loss), config.training_initial_lr, *args, **kwargs)
