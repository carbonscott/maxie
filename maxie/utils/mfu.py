"""
All arithemtic assumes the use of torch tensor.
"""

def estimate_conv_flops(
        H_input, W_input, C_input, C_output,
        H_kernel, W_kernel, H_stride, W_stride,
        H_padding, W_padding, count_multiply_add_as=1
    ):
    """
    Estimate the number of FLOPs for a convolutional layer.

    This function implements the following logic:
    - Calculate the number of times the kernel is applied (n_H * n_W).
    - Calculate the FLOPs for a single 2D kernel application.
    - Multiply by the number of input and output channels.

    The calculation takes into account:
    - Input dimensions (height, width, channels)
    - Output channels
    - Kernel dimensions
    - Stride
    - Padding

    Count each multiply-add as this many FLOPs (default is 1)
    """
    # Calculate flops for a single 2D kernel application
    flops_kernel2d = H_kernel * W_kernel * count_multiply_add_as

    # Calculate n_H and n_W (number of times kernel is applied in each dimension)
    n_H = (H_input + 2*H_padding - H_kernel) // H_stride + 1
    n_W = (W_input + 2*W_padding - W_kernel) // W_stride + 1

    # Calculate total number of kernel applications
    num_kernel_travel = n_H * n_W

    # Calculate flops for all channels
    total_flops = C_input * C_output * num_kernel_travel * flops_kernel2d

    return total_flops


def estimate_linear_flops(in_features, out_features, count_multiply_add_as=2):
    return count_multiply_add_as * in_features * out_features


def estimate_transformer_mfu(model_hidden_size, num_heads, num_layers, context_length):
    """
    | Operation                   | Input shape           | Output shape | Ops         | Reshape                                   | FLOPs         |
    |-----------------------------|-----------------------|--------------|-------------|-------------------------------------------|---------------|
    | Embedding (deepmind style)  | (B,T,V);(V,E)         | (B,T,E)      | matmul      | N/A                                       | (2V)(BTE)     |
    | Embedding (lookup table)    | (B,T)                 | (B,T,E)      | look-up     | N/A                                       | 0             |
    | KQV Proj                    | (B,T,1,E);(B,T,E,3E)  | (B,T,1,3E)   | matmul      | (B,T,1,3E) -> (B,T,1,3HN) -> (B,N,T,3H)   | (2E)(3HNBT)   |
    | KQ                          | (B,N,T,H);(B,N,H,T)   | (B,N,T,T)    | matmul      | N/A                                       | (2H)(BNTT)    |
    | Softmax                     | (B,N,T,T)             | (B,N,T,T)    | elementwise | N/A                                       | 3(BNTT)       |
    | Update V                    | (B,N,T,T);(B,N,T,H)   | (B,N,T,H)    | matmul      | N/A                                       | (2T)(BNTH)    |
    | Final Linear (Proj V)       | (B,T,NH);(NH,E)       | (B,T,E)      | matmul      | (B,N,T,H)->(B,T,NH)                       | (2NH)(BTE)    |
    | Feedforward (FF)            | (B,T,E);(E,4E)        | (B,T,4E)     | matmul      | N/A                                       | (2E)(4EBT)    |
    """
    head_hidden_size = model_hidden_size / num_heads

    mfu_in_kqv_proj = (2*model_hidden_size)*(3*head_hidden_size*num_heads)               # (2E)(3HNBT)
    mfu_in_kq       = (2*head_hidden_size)*(num_heads*context_length*context_length)     # (2H)(BNTT)
    mfu_in_softmax  = 3*(num_heads*context_length*context_length)                        # 3(BNTT)
    mfu_in_update_v = (2*context_length)*(num_heads*context_length*head_hidden_size)     # (2T)(BNTH)
    mfu_in_proj_v   = (2*num_heads*head_hidden_size)*(context_length*model_hidden_size)  # (2NH)(BTE)
    mfu_in_ff       = (2*model_hidden_size)*(4*model_hidden_size*context_length)         # (2E)(4EBT)

    return num_layers * (mfu_in_kqv_proj+mfu_in_kq+mfu_in_softmax+mfu_in_update_v+mfu_in_proj_v+mfu_in_ff)
