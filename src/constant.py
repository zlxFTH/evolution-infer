hidden_size       = 5120
intermediate_size = 17408
rms_norm_eps      = 1e-6
max_positions     = 262144
base              = 10000000.0
rotary_dim        = 64
num_heads         = 24
num_kv_heads      = 4
head_dim          = 256

linear_num_kernel   = 4
linear_num_kq_heads = 16
linear_num_v_heads  = 48
linear_head_dim     = 128

block             = 4
vocab_size        = 248320
num_hidden_layers = 64

qk_dim      = linear_num_kq_heads * linear_head_dim
v_dim       = linear_num_v_heads * linear_head_dim
conv_dim    = qk_dim * 2 + v_dim
kv_shape    = (num_kv_heads, head_dim)
conv_shape  = (conv_dim, linear_num_kernel - 1)
recur_shape = (linear_num_v_heads, linear_head_dim, linear_head_dim)