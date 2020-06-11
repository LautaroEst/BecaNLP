import torch
import math


def constant_init(tensor,val):
	with torch.no_grad():
		tensor.fill_(val)
	return tensor


def ones_init(tensor):
	with torch.no_grad():
		tensor.fill_(1.)
	return tensor


def zeros_init(tensor):
	with torch.no_grad():
		tensor.zero_()
	return tensor


def diagonal_init(tensor,fill_value,wrap=False):
	with torch.no_grad():
		tensor.fill_diagonal_(fill_value,wrap)
	return tensor


def uniform_init(tensor,a=0.0,b=1.0,random_state=None):
	if random_state is None:
		with torch.no_grad():
			tensor.uniform_(a,b)
	else:
		gen = torch.Generator()
		gen.manual_seed(random_state)
		with torch.no_grad():
			tensor.copy_(torch.rand(tensor.size(),generator=gen))
	return tensor


def normal_init(tensor,mean=0.0,std=1.0,random_state=None):
	if random_state is None:
		with torch.no_grad():
			tensor.normal_(mean,std)
	else:
		gen = torch.Generator()
		gen.manual_seed(random_state)
		with torch.no_grad():
			tensor.copy_(torch.randn(tensor.size(),generator=gen))
	return tensor


def dirac_init(tensor, groups=1):
    r"""Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
    delta function. Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.size()

    if sizes[0] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    with torch.no_grad():
        tensor.zero_()

        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2,
                           tensor.size(3) // 2] = 1
                else:  # Volumetric convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2,
                           tensor.size(3) // 2, tensor.size(4) // 2] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_init(tensor, gain=1., random_state=None):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    uniform_init(tensor, -a, a, random_state)
    return tensor

def xavier_normal_init(tensor, gain=1., random_state=None):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    normal_init(tensor, 0., std, random_state)
    return tensor


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))



def kaiming_uniform_init(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', random_state=None):
	r"""Fills the input `Tensor` with values according to the method
	described in `Delving deep into rectifiers: Surpassing human-level
	performance on ImageNet classification` - He, K. et al. (2015), using a
	uniform distribution. The resulting tensor will have values sampled from
	:math:`\mathcal{U}(-\text{bound}, \text{bound})` where

	.. math::
		\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

	Also known as He initialization.

	Args:
	tensor: an n-dimensional `torch.Tensor`
	a: the negative slope of the rectifier used after this layer (only 
	used with ``'leaky_relu'``)
	mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
		preserves the magnitude of the variance of the weights in the
		forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
		backwards pass.
	nonlinearity: the non-linear function (`nn.functional` name),
		recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

	Examples:
		>>> w = torch.empty(3, 5)
		>>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
	"""
	fan = _calculate_correct_fan(tensor, mode)
	gain = calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

	uniform_init(tensor,-bound,bound,random_state)
	return tensor



def kaiming_normal_init(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', random_state=None):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    normal_init(tensor,0,std,random_state)
    return tensor



def orthogonal_init(tensor, gain=1, random_state=None):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = normal_init(tensor.new(rows, cols),0,1,random_state)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)

    return tensor


def sparse_init(tensor, sparsity, std=0.01, random_state=None):
	r"""Fills the 2D input `Tensor` as a sparse matrix, where the
	non-zero elements will be drawn from the normal distribution
	:math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
	Hessian-free optimization` - Martens, J. (2010).

	Args:
		tensor: an n-dimensional `torch.Tensor`
		sparsity: The fraction of elements in each column to be set to zero
		std: the standard deviation of the normal distribution used to generate
			the non-zero values

	Examples:
		>>> w = torch.empty(3, 5)
		>>> nn.init.sparse_(w, sparsity=0.1)
	"""
	if tensor.ndimension() != 2:
		raise ValueError("Only tensors with 2 dimensions are supported")

	rows, cols = tensor.shape
	num_zeros = int(math.ceil(sparsity * rows))

	if random_state is not None:
		gen = torch.Generator()
		gen.manual_seed(random_state)
	else:
		gen = None

	with torch.no_grad():
		normal_init(tensor,0,std,random_state)
		for col_idx in range(cols):
			if gen:
				row_indices = torch.randperm(rows,generator=gen)
			else:
				row_indices = torch.randperm(rows)
			zero_indices = row_indices[:num_zeros]
			tensor[zero_indices, col_idx] = 0

	return tensor