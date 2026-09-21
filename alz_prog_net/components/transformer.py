import keras
from keras import ops

@keras.saving.register_keras_serializable(
    package="AlzProgNet"
)
class RMSNorm(keras.layers.Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        dim = input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            shape=(dim,),
            initializer="ones",
            trainable=True,
        )

    def call(self, x):
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
        rms = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale

@keras.saving.register_keras_serializable(
    package="AlzProgNet"
)
class TransformerBlock(keras.layers.Layer):
    """Pre-norm Transformer block with attention and SwiGLU FFN."""

    def __init__(
            self,
            dim,
            num_heads,
            ff_dim,
            attention_dropout=0.0,
            residual_dropout=0.05,
            **kwargs):
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.norm1 = RMSNorm()

        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            use_bias=False,
            dropout=attention_dropout
        )

        self.norm2 = RMSNorm()

        self.residual_dropout = keras.layers.Dropout(residual_dropout)

        # SwiGLU
        self.gate = keras.layers.Dense(ff_dim, use_bias=False)
        self.up = keras.layers.Dense(ff_dim, use_bias=False)
        self.down = keras.layers.Dense(dim, use_bias=False)

    def call(self, x, training=None):
        # Attention
        h = self.norm1(x)
        h = self.attn(h, h, training=training)
        x = x + h

        # Feed-forward network
        h = self.norm2(x)

        # SwiGLU:
        # SiLU(W_gate x) * (W_up x)
        h = ops.silu(self.gate(h)) * self.up(h)
        h = self.down(h)

        # Residual connection
        return x + self.residual_dropout(h, training=training)

@keras.saving.register_keras_serializable(
    package="AlzProgNet"
)
class AttentionPooling(keras.layers.Layer):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.dropout_rate = dropout_rate
        self.score_projection = keras.layers.Dense(
            self.hidden_dim,
            activation="tanh",
            name="score_projection"
        )

        self.dropout = keras.layers.Dropout(
            self.dropout_rate,
            name="score_dropout"
        )

        self.score_output = keras.layers.Dense(
            1, use_bias=False, name="score_output"
        )

    def call(
        self,
        inputs,
        training=None,
        return_attention=False
    ):
        x = self.score_projection(inputs)
        x = self.dropout(x, training=training)

        scores = self.score_output(x)
        attention_weights = ops.softmax(
            scores, axis=1
        )

        pooled = ops.sum(
            attention_weights * inputs,
            axis=1
        )

        if return_attention:
            return pooled, attention_weights

        return pooled

    def get_config(self):
        config = super().get_config()

        config.update({
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        })

        return config