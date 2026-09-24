import keras
from keras import ops

@keras.saving.register_keras_serializable(package="AlzProgNet")
class LongitudinalTransitionLoss(keras.losses.Loss):
    def __init__(
        self,
        class_weights,
        time_weights=None,
        severity_matrix=None,
        severity_weight=0.25,
        transition_weight=0.25,
        transition_loss="huber",
        huber_delta=1.0,
        stable_transition_weight=1.0,
        converter_transition_weight=1.0,
        converter_sample_weight=1.0,
        from_logits=False,
        name="longitudinal_transition_loss",
        reduction="sum_over_batch_size",
        **kwargs
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)

        #class weights [CN, MCI, AD]
        if class_weights is None:
            raise ValueError("class_weights must be passed")
        self.class_weights = class_weights

        #time weights [T0, T6, T12, T24]
        if time_weights is None:
            time_weights = [1.0, 1.0, 1.0, 1.0]

        self.time_weights = list(time_weights)

        #severity matrix where rows = true class, cols = predicted class
        if severity_matrix is None:
            severity_matrix = [
                [0.0, 0.5, 2.0],
                [0.5, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ]

        self.severity_matrix = [
            list(row) for row in severity_matrix 
        ]

        self.severity_weight = float(severity_weight)
        self.transition_weight = float(transition_weight)
        self.stable_transition_weight = float(stable_transition_weight)
        self.converter_transition_weight = float(converter_transition_weight)
        self.converter_sample_weight = float(converter_sample_weight)
        self.transition_loss = transition_loss
        self.huber_delta = float(huber_delta)
        self.from_logits = from_logits

        if transition_loss not in {"mae", "mse", "huber"}:
            raise ValueError("transition_loss must of one of: 'mae', 'mse', or 'huber'")

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        class_weights = ops.convert_to_tensor(self.class_weights, dtype="float32")
        time_weights = ops.convert_to_tensor(self.time_weights, dtype="float32")
        severity_matrix = ops.convert_to_tensor(self.severity_matrix, dtype="float32")
        class_values = ops.convert_to_tensor(
            [0.0, 1.0, 2.0],
            dtype="float32"
        )

        #normalize time weights
        # time_weights = time_weights / ops.mean(time_weights)

        #probabilities
        if self.from_logits:
            probabilities = ops.softmax(y_pred, axis=-1)
        else:
            probabilities = y_pred

        # true class (N, 4)
        true_class = ops.argmax(y_true, axis=-1)

        # cross entropy (N, 4)
        cce = ops.categorical_crossentropy(
            target=y_true,
            output=y_pred,
            from_logits=self.from_logits,
            axis=-1
        )

        #class balancing
        sample_class_weights = ops.sum(
            y_true * ops.expand_dims(class_weights, axis=0),
            axis=-1
        )
        weighted_cce = cce * sample_class_weights

        # severity penalty
        # (N, 4, 3)
        true_severity_cost = ops.take(
            severity_matrix,
            true_class,
            axis=0
        )

        severity_loss = ops.sum(
            probabilities * true_severity_cost,
            axis=-1
        )

        #apply class balancing
        severity_loss = severity_loss * sample_class_weights

        #expected disease state
        expected_state = ops.sum(
            probabilities * class_values,
            axis=-1
        )

        true_state = ops.cast(true_class, "float32")

        # transitions (N, 3)
        true_delta = true_state[:, 1:] - true_state[:, :-1]

        # predicted transitions
        predicted_delta = expected_state[:, 1:] - expected_state[:, :-1]

        transition_error = true_delta - predicted_delta

        # transition loss
        if self.transition_loss == "mae":
            transition_loss = ops.abs(transition_error)
        elif self.transition_loss == "mse":
            transition_loss = ops.square(transition_error)
        else:
            abs_error = ops.abs(transition_error)
            quadratic = ops.minimum(abs_error, self.huber_delta)
            linear = abs_error - quadratic
            transition_loss = 0.5 * ops.square(quadratic) + self.huber_delta * linear

        # transiting event weighing
        is_conversion_transition = ops.cast(
            ops.abs(true_delta) > 0.0,
            "float32",
        )

        transition_event_weights = (
            self.stable_transition_weight +
            is_conversion_transition * (
                self.converter_transition_weight - self.stable_transition_weight
            )
        )

        transition_loss = (
            transition_loss * transition_event_weights
        )

        converter_sample = ops.max(is_conversion_transition, axis=-1) #(N,)
        sample_weights = (
            1.0 + 
            converter_sample * (
                self.converter_sample_weight - 1.0
            )
        )

        # convert transition_loss (N, 3) to (N, 4)
        zero_transition = ops.zeros_like(transition_loss[:, :1])
        transition_loss = ops.concatenate([
            zero_transition,
            transition_loss
        ], axis=1)

        total_loss = (
            weighted_cce
            +
            self.severity_weight * severity_loss
            +
            self.transition_weight * transition_loss
        )

        # time weighing
        total_loss = total_loss * ops.expand_dims(time_weights, axis=0)

        # reduce over time (N, 4) to (N, )
        loss_per_sample = ops.mean(
            total_loss,
            axis=1
        )
        loss_per_sample = loss_per_sample * sample_weights

        return loss_per_sample

    def get_config(self):
        config = super().get_config()

        config.update({
            "class_weights": self.class_weights,
            "time_weights": self.time_weights,
            "severity_matrix": self.severity_matrix,
            "severity_weight": self.severity_weight,
            "transition_weight": self.transition_weight,
            "transition_loss": self.transition_loss,
            "huber_delta": self.huber_delta,
            "stable_transition_weight": self.stable_transition_weight,
            "converter_transition_weight": self.converter_transition_weight,
            "converter_sample_weight": self.converter_sample_weight,
            "from_logits": self.from_logits
        })

        return config


@keras.saving.register_keras_serializable(
    package="AlzProgNet"
)
class DiscreteTimeConversionLoss(
    keras.losses.Loss
):
    # y_true is (B, 3, 2)
    # [..., 0]: event indicator
    # [..., 1]: at-risk indicator
    # y_pred: (B, intervals) first conversion probabilities
    def __init__(
        self,
        epsilon=1e-7,
        name="discrete-time-conversion_loss",
        reduction="sum_over_batch_size",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "flaot32")

        event = y_true[..., 0]
        at_risk = y_true[..., 1]

        hazard = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        interval_loss = -(
            event * ops.log(hazard) +
            (1.0 - event) * ops.log(1.0 - hazard)
        )

        # ignore intervals after first conversion
        interval_loss = interval_loss * at_risk

        denominator = ops.maximum(
            ops.sum(at_risk, axis=-1),
            1.0
        )

        loss_per_sample = (
            ops.sum(interval_loss, axis=-1) / denominator
        )

        return loss_per_sample

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })

        return config

