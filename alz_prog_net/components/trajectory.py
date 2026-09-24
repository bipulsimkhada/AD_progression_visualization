import keras
from keras import ops, layers
from alz_prog_net.components.progression import SwiGLU

@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class TrajectoryPrediction(keras.layers.Layer):
    """
    Auxiliary trajectory prediction branch.

    Predicts:
    1. Trajectory class
        0 = stable
        1 = forward_progressive
        2 = reverter
        3 = fluctuate
    
    2. Discrete-time first-conversion hazard
        interval 0: 0 -> 6 months
        interval 1: 6 -> 12 months
        interval 2: 12 -> 24 months
    """

    def __init__(
        self,
        hidden_dims=(128, 64, 32),
        num_trajectory_classes=4,
        num_conversion_intervals=3,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dims = tuple(hidden_dims)
        self.num_trajectory_classes = int(num_trajectory_classes)
        self.num_conversion_intervals = int(num_conversion_intervals)
        self.dropout_rate = float(dropout_rate)

        self.hidden_layers = []

        for i, hidden_dim in enumerate(self.hidden_dims):
            block = keras.Sequential(
                [
                    SwiGLU(output_dim=hidden_dim, name=f"swiglu_{i}"),
                    layers.Dropout(self.dropout_rate, name=f"dropout_{i}"),
                ],
                name=f"trajectory_block_{i}",
            )

            self.hidden_layers.append(block)

        self.trajectory_classifier = layers.Dense(self.num_trajectory_classes, name="trajectory_logits")
        self.conversion_hazard_head = layers.Dense(self.num_conversion_intervals, activation="sigmoid", name="conversion_hazard")

    def call(self, inputs, training=None):
        x = inputs
        hidden_states = []

        for layer in self.hidden_layers:
            x = layer(x, training=training)
            hidden_states.append(x)

        representation = x
        trajectory_logits = self.trajectory_classifier(representation)
        trajectory_probabilities = ops.softmax(trajectory_logits, axis=-1)

        hazard = self.conversion_hazard_head(representation)
        survival = ops.cumprod(
            1.0 - hazard, 
            axis=-1
        )

        survival_before = ops.concatenate(
            [
                ops.ones_like(survival[:, :1]),
                survival[:, :-1],
            ],
            axis=-1
        )

        event_probability = survival_before * hazard

        return {
            "representation": representation,
            "hidden_states": hidden_states,
            "trajectory_probabilities": trajectory_probabilities,
            "conversion_event_probability": event_probability,
            "conversion_hazard": hazard
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "num_trajectory_classes":self.num_trajectory_classes,
            "num_conversion_intervals":self.num_conversion_intervals,
            "dropout_rate":self.dropout_rate,
        })

        return config

        