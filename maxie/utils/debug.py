# -----------------------------------------------------------------------
#  Monitoring classes
# -----------------------------------------------------------------------
class ActivationMonitor:
    """
    A class for monitoring activations in PyTorch neural network modules.

    This class uses forward hooks to capture input and output tensors of specified layers.
    It performs a depth-first search through the model's modules using `named_modules()`.

    Key behaviors:
    1. Monitors all layers if no specific layers are specified.
    2. Captures both pre-activation (input) and post-activation (output) tensors.
    3. Includes the root module (entire network) with an empty string key ('').

    The activations dictionary keys correspond to module names:
    - 'layer_name': For specific layers
    - 'container_name': For container modules like nn.Sequential, nn.ModuleList, nn.ModuleDict
    - '': For the root module (entire network)

    Each activation entry contains:
    - 'pre': Input tensor to the module
    - 'pos': Output tensor from the module

    Note: The root module ('') provides the overall network input and final output.

    Args:
        model (nn.Module): The PyTorch model to monitor.
        modules_to_monitor (list, optional): Specific modules to monitor. If None, all modules are monitored.

    Example:
        monitor = ActivationMonitor(model)
        output = model(input_data)
        activations = monitor.activations
    """
    def __init__(self, model, modules_to_monitor = None):
        self.model              = model
        self.modules_to_monitor = modules_to_monitor
        self.activations        = {}
        self.hooks              = []

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
            if self.modules_to_monitor is None or name in self.modules_to_monitor:
                getattr(module, 'remove_hooks', lambda : None)()
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)  # For clean-up

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradientMonitor:
    """
    A class for monitoring gradients in PyTorch neural network parameters.

    This class uses backward hooks to capture gradient tensors of specified parameters
    during the backward pass.  It iterates through the model's parameters using `named_parameters()`.

    Key behaviors:
    1. Monitors gradients of all parameters if no specific layers are specified.
    2. Uses parameter hooks to capture gradients after they've been computed.
    3. Stores gradients for each monitored parameter.

    The gradients dictionary keys correspond to parameter names:
    - 'layer_name.weight': For weight parameters
    - 'layer_name.bias': For bias parameters

    Note: Unlike ActivationMonitor, there's no root module concept here. Only individual
    parameter gradients are captured.

    Args:
        model (nn.Module): The PyTorch model to monitor.
        params_to_monitor (list, optional): Specific parameters to monitor. If None, all parameters are monitored.
                                            Partial matches are allowed (e.g., 'conv' will match 'conv1', 'conv2', etc.)

    Attributes:
        gradients (dict): A dictionary storing the gradients of monitored parameters.
                          Keys are parameter names, values are gradient tensors.

    Example:
        monitor = GradientMonitor(model)
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        gradients = monitor.gradients

    Note:
    - Gradients are cloned and moved to CPU to avoid interfering with the original computation.
    - Hooks should be removed (using remove_hooks()) when no longer needed to free up memory.
    """
    def __init__(self, model, params_to_monitor = None):
        self.model             = model
        self.params_to_monitor = set(params_to_monitor) if params_to_monitor is not None else None
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
            if self.params_to_monitor is None or any(layer in name for layer in self.params_to_monitor):
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
