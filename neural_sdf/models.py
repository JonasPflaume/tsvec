import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from neural_sdf.sdf_jax import hash_encoding
from typing import Callable, Sequence
import flax.linen as nn

###############################################################
# Instant Neural Graphics Primitives with 
# a Multiresolution Hash Encoding (Müller et al, SIGGRAPH 22)
###############################################################

class MLP(nn.Module):
    dims: Sequence[int]
    act: Callable

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1
        for dim in self.dims:
            x = nn.Dense(features=dim)(x)
            x = self.act(x)
        y = nn.Dense(features=1)(x)
        return y[0]

class HashEmbedding(nn.Module):
    levels: int = 16
    hashmap_size_log2: int = 20
    features_per_entry: int = 2
    nmin: int = 16
    nmax: int = 512

    def setup(self):
        hashmap_size = 1 << self.hashmap_size_log2
        
        def param_init(key, shape, dtype=jnp.float32):
            return jrandom.uniform(
                key, 
                shape, 
                dtype,
                minval=-0.0001, 
                maxval=0.0001
            )

        self.theta = self.param(
            'theta',
            param_init,
            (self.levels, hashmap_size, self.features_per_entry)
        )

    def __call__(self, x):
        assert x.ndim == 1
        y = hash_encoding.encode(x, self.theta, self.nmin, self.nmax)
        return y.reshape(-1)


def build_hash_mlp(emb_kwargs, hidden, act):
    model = nn.Sequential([
        HashEmbedding(**emb_kwargs),
        MLP([hidden, hidden], act),
    ])
    return model

###############################################################
# Implicit Geometric Regularization for Learning Shapes
# (Gropp et al, ICML 2020)
###############################################################

def softplus(x, beta=100):
    return jnp.logaddexp(beta*x, 0) / beta

class IGRModel(nn.Module):
    input_dim: int
    depth: int
    hidden: int
    act: Callable = softplus
    radius_init: float = 1.0

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1

        # x is assumed in [0,1]^d
        # we now rescale to [-1,1]^d for geometric init to work
        x_scaled = 2*x - 1.0

        y = x_scaled
        for i in range(self.depth):
            # prepare skip connection one layer earlier
            if i == 2:
                h = self.hidden - self.input_dim 
            else: 
                h = self.hidden

            # geometric initialization
            layer = nn.Dense(
                features=h,
                kernel_init=jax.nn.initializers.normal(np.sqrt(2) / np.sqrt(h)),
                bias_init=jax.nn.initializers.constant(0.0),
            )
            y = layer(y)
            y = self.act(y)
            
            # skip connection to the fourth layer
            if i + 1 == 3: # 0-indexed, so layer 4 is at index 3
                y = jnp.concatenate([y, x_scaled])

        def final_kernel_init(key, shape, dtype=jnp.float32):
            # h is from the last loop iteration
            mean = np.sqrt(np.pi) / np.sqrt(self.hidden)
            stddev = 0.00001
            return mean + stddev * jrandom.normal(key, shape, dtype)

        final_layer = nn.Dense(
            features=1, 
            kernel_init=final_kernel_init,
            bias_init=jax.nn.initializers.constant(-self.radius_init),
        )
        y = final_layer(y)

        # rescaling values too to fit [0,1]^d instead of [-1,1]^d
        y = y / 2.0

        return y[0]
    
class IGRGrayModel(nn.Module):
    input_dim: int
    depth: int
    hidden: int
    act: Callable = softplus
    radius_init: float = 1.0

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1

        # x is assumed in [0,1]^d
        # we now rescale to [-1,1]^d for geometric init to work
        x_scaled = 2*x - 1.0

        y = x_scaled
        for i in range(self.depth):
            # prepare skip connection one layer earlier
            if i == 2:
                h = self.hidden - self.input_dim 
            else: 
                h = self.hidden

            # geometric initialization
            layer = nn.Dense(
                features=h,
                kernel_init=jax.nn.initializers.normal(np.sqrt(2) / np.sqrt(h)),
                bias_init=jax.nn.initializers.constant(0.0),
            )
            y = layer(y)
            y = self.act(y)
            
            # skip connection to the fourth layer
            if i + 1 == 3: # 0-indexed, so layer 4 is at index 3
                y = jnp.concatenate([y, x_scaled])

        # Modified final layer for grayscale output [0,1]
        def final_kernel_init(key, shape, dtype=jnp.float32):
            # Xavier/Glorot initialization for sigmoid activation
            fan_in = self.hidden
            stddev = jnp.sqrt(2.0 / fan_in)
            return stddev * jrandom.normal(key, shape, dtype)

        final_layer = nn.Dense(
            features=1, 
            kernel_init=final_kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),  # Neutral bias for sigmoid
        )
        y = final_layer(y)
        
        # Apply sigmoid to ensure output is in [0,1] for grayscale values
        y = jax.nn.sigmoid(y)

        return y[0]


###############################################################
# Geometry-consistent Neural Shape Representation With
# Implicit Displacement Fields (Yifan et al, ICLR 2022)
###############################################################

def chi(fx, nu):
    return 1.0 / (1.0 + (fx / nu)**4)

class IDFModel(nn.Module):
    base_model_fn: Callable[[], nn.Module]
    disp_model_fn: Callable[[], nn.Module]
    nu: float

    def setup(self):
        self.base = self.base_model_fn()
        self.disp = self.disp_model_fn()

    def __call__(self, x):
        fx, gradfx = jax.value_and_grad(self.base)(x)
        norm_gradfx = jnp.linalg.norm(gradfx)
        # Avoid division by zero
        safe_norm_gradfx = jnp.where(norm_gradfx == 0, 1.0, norm_gradfx)
        x2 = x + chi(fx, self.nu) * self.disp(x) * gradfx / safe_norm_gradfx
        return self.base(x2)

###############################################################
# Tests
###############################################################

if __name__ == '__main__':
    key = jrandom.PRNGKey(42)
    
    # --- Test Hash MLP ---
    print("--- Testing Hash MLP ---")
    emb_kwargs = {'nmin': 16, 'nmax': 2048, 'hashmap_size_log2': 19}
    hash_mlp = build_hash_mlp(emb_kwargs, hidden=64, act=nn.relu)
    
    x_test = jnp.ones(3) * 0.5
    key, subkey = jrandom.split(key)
    params = hash_mlp.init(subkey, x_test)['params']
    
    print("Hash MLP initialized successfully.")
    
    # Test forward pass
    output = hash_mlp.apply({'params': params}, x_test)
    print(f"Forward pass output: {output}")

    # Test gradient
    grad_fn = jax.grad(lambda p, i: hash_mlp.apply({'params': p}, i))
    grads = grad_fn(params, x_test)
    print("Gradients computed successfully.")
    print("-" * 25, "\n")

    # --- Test IGR Model ---
    print("--- Testing IGR Model ---")
    igr_model = IGRModel(input_dim=3, depth=4, hidden=128)
    
    key, subkey = jrandom.split(key)
    params_igr = igr_model.init(subkey, x_test)['params']
    print("IGR Model initialized successfully.")

    # Test forward pass
    output_igr = igr_model.apply({'params': params_igr}, x_test)
    print(f"Forward pass output: {output_igr}")

    # Test gradient
    grad_fn_igr = jax.grad(lambda p, i: igr_model.apply({'params': p}, i))
    grads_igr = grad_fn_igr(params_igr, x_test)
    print("Gradients computed successfully.")
    print("-" * 25, "\n")

    # --- Test IDF Model ---
    print("--- Testing IDF Model ---")
    # Using IGR as the base model and a simple MLP as the displacement field
    base_fn = lambda: IGRModel(input_dim=3, depth=4, hidden=128)
    disp_fn = lambda: MLP(dims=[64, 64], act=nn.relu)
    
    idf_model = IDFModel(base_model_fn=base_fn, disp_model_fn=disp_fn, nu=0.01)

    key, subkey = jrandom.split(key)
    params_idf = idf_model.init(subkey, x_test)['params']
    print("IDF Model initialized successfully.")

    # Test forward pass
    output_idf = idf_model.apply({'params': params_idf}, x_test)
    print(f"Forward pass output: {output_idf}")

    # Test gradient
    grad_fn_idf = jax.grad(lambda p, i: idf_model.apply({'params': p}, i))
    grads_idf = grad_fn_idf(params_idf, x_test)
    print("Gradients computed successfully.")
    print("-" * 25, "\n")
