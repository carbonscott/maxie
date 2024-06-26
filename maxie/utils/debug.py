# -----------------------------------------------------------------------
#  Monitoring classes
# -----------------------------------------------------------------------
class ActivationMonitor:
    """ Forward hook is used """
    def __init__(self, model, layers_to_monitor = None):
        self.model             = model
        self.layers_to_monitor = layers_to_monitor
        self.activations       = {}
        self.hooks             = []

        self.add_hooks()

    def hook_fn(self, name):
        """ Closure to parameterize a function: f(params)(input) """
        def hook(module, input, output):
            self.activations[name] = {
                'pre' : input[0].detach().cpu(),  # For some reasons, it's always a tuple
                'pos' : output.detach().cpu(),
            }
        return hook

    def add_hooks(self):
        for name, module in self.model.named_modules():
            if self.layers_to_monitor is None or name in self.layers_to_monitor:
                getattr(module, 'remove_hooks', lambda : None)()
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)  # For clean-up

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradientMonitor:
    """ Backward hook (more specifically, param hook) is used """
    def __init__(self, model, layers_to_monitor = None):
        self.model             = model
        self.layers_to_monitor = set(layers_to_monitor) if layers_to_monitor is not None else None
        self.gradients         = {}
        self.hooks             = []

        self.add_hooks()

    def hook_fn(self, name):
        def hook(grad):
            """ https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook """
            self.gradients[name] = grad.clone().cpu()
        return hook

    def add_hooks(self):
        for name, param in self.model.named_parameters():
            if self.layers_to_monitor is None or any(layer in name for layer in self.layers_to_monitor):
                getattr(param, 'remove_hooks', lambda : None)()
                hook = param.register_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ----------------------------------------------------------------------- #
#  Helper functions
# ----------------------------------------------------------------------- #
def get_param(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param
    raise ValueError(f"No parameter named '{param_name}' found in the model.")


def get_param_grad(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param.grad
    raise ValueError(f"No parameter named '{param_name}' found in the model.")


def track_metrics(model, lr, layers_to_monitor = None):
    """
    Users need to decide whether to include .weight or .bias.

    Example:
        track_metrics(model, lr, ['features.0.weight', 'features.0.bias', 'features.2.weight',])
    """
    metrics = {
        'percent_param_update': {},
        'grad_mean_std': {}
    }

    for name, param in model.named_parameters():
        if layers_to_monitor is not None and name not in layers_to_monitor:
            continue

        if param.grad is not None:
            # Percent param update
            metrics['percent_param_update'][name] = (
                (param.grad * lr).std().cpu() /
                param.detach().std().cpu()
            ).log10().item()

            # Mean and Std of gradients
            metrics['grad_mean_std'][name] = (
                param.grad.mean().cpu().item(),
                param.grad.std().cpu().item()
            )

    return metrics
