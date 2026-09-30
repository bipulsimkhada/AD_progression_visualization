RANDOM_STATE = 42

MODALITIES = {
    "mri": (6, slice(None, 6)),
    "pet": (2, slice(6, 8)),
    "cog": (11, slice(8, 19)),
    "csf": (3, slice(19, 22)),
    "rf":  (4, slice(22, 26)),
}

CONFIGS = [
    # {
    #     "name": "proposed_median_min-max",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "median",
    #         "scaling": "min-max",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_mean_standarization_base_loss",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 1,
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1,
    #         "converter_transition_weight": 1,
    #         "converter_sample_weight": 1,
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_base_and_transition_loss_weight",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4,
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1,
    #         "converter_transition_weight": 1,
    #         "converter_sample_weight": 1,
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_base_and_converter_sample_weight",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 1,
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1,
    #         "converter_transition_weight": 1,
    #         "converter_sample_weight": 3,
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_base_and_transition_weight",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 1,
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2,
    #         "converter_sample_weight": 1,
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
#     {
#         "name": "proposed_with_mean_min-max",
#         "modalities": ("mri", "pet", "cog", "csf", "rf"),
#         "model": {
#             "imputer": "mean",
#             "scaling": "min-max",
#             "batch_size": 32,
#             "temporal_levels": (True, True, False),
#             "num_transformer_layers": 3,
#             "use_time": True,
#             "use_gate": True,
#             "use_interaction": True,
#         },
#         "loss": {
#             "transition_loss_weight": 4, #s1_loss_7
#             "transition_loss": "mse",
#             "stable_transition_weight": 1.5,
#             "converter_transition_weight": 2.0,
#             "converter_sample_weight": 3, #s2_loss_5
#             "huber_delta": 1.0,
#             "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
#             "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
#         }
#     },
    # {
    #     "name": "proposed_with_mean_standarization",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #         "name": "proposed_with_median_standarization",
    #         "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #         "model": {
    #             "imputer": "median",
    #             "scaling": "standarization",
    #             "batch_size": 32,
    #             "temporal_levels": (True, True, False),
    #             "num_transformer_layers": 3,
    #             "use_time": True,
    #             "use_gate": True,
    #             "use_interaction": True,
    #         },
    #         "loss": {
    #             "transition_loss_weight": 4, #s1_loss_7
    #             "transition_loss": "mse",
    #             "stable_transition_weight": 1.5,
    #             "converter_transition_weight": 2.0,
    #             "converter_sample_weight": 3, #s2_loss_5
    #             "huber_delta": 1.0,
    #             "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #             "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         }
    #     },
    # {
    #     "name": "proposed_with_mean_standarization_wo_transformer",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 0,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_mean_standarization_wo_temporal",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (False, False, False),
    #         "num_transformer_layers": 3,
    #         "use_time": True,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_mean_standarization_wo_time",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (True, True, False),
    #         "num_transformer_layers": 3,
    #         "use_time": False,
    #         "use_gate": True,
    #         "use_interaction": True,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    # {
    #     "name": "proposed_with_mean_standarization_base_model",
    #     "modalities": ("mri", "pet", "cog", "csf", "rf"),
    #     "model": {
    #         "imputer": "mean",
    #         "scaling": "standarization",
    #         "batch_size": 32,
    #         "temporal_levels": (False, False, False),
    #         "num_transformer_layers": 0,
    #         "use_time": False,
    #         "use_gate": False,
    #         "use_interaction": False,
    #     },
    #     "loss": {
    #         "transition_loss_weight": 4, #s1_loss_7
    #         "transition_loss": "mse",
    #         "stable_transition_weight": 1.5,
    #         "converter_transition_weight": 2.0,
    #         "converter_sample_weight": 3, #s2_loss_5
    #         "huber_delta": 1.0,
    #         "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
    #         "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
    #     }
    # },
    {
        "name": "proposed_with_mean_standarization_wo_mri",
        "modalities": ("pet", "cog", "csf", "rf"),
        "model": {
            "imputer": "mean",
            "scaling": "standarization",
            "batch_size": 32,
            "temporal_levels": (True, True, False),
            "num_transformer_layers": 3,
            "use_time": True,
            "use_gate": True,
            "use_interaction": True,
        },
        "loss": {
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        }
    },
    {
        "name": "proposed_with_mean_standarization_wo_pet",
        "modalities": ("mri", "cog", "csf", "rf"),
        "model": {
            "imputer": "mean",
            "scaling": "standarization",
            "batch_size": 32,
            "temporal_levels": (True, True, False),
            "num_transformer_layers": 3,
            "use_time": True,
            "use_gate": True,
            "use_interaction": True,
        },
        "loss": {
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        }
    },
    {
        "name": "proposed_with_mean_standarization_wo_cog",
        "modalities": ("mri", "pet", "csf", "rf"),
        "model": {
            "imputer": "mean",
            "scaling": "standarization",
            "batch_size": 32,
            "temporal_levels": (True, True, False),
            "num_transformer_layers": 3,
            "use_time": True,
            "use_gate": True,
            "use_interaction": True,
        },
        "loss": {
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        }
    },
    {
        "name": "proposed_with_mean_standarization_wo_csf",
        "modalities": ("mri", "pet", "cog", "rf"),
        "model": {
            "imputer": "mean",
            "scaling": "standarization",
            "batch_size": 32,
            "temporal_levels": (True, True, False),
            "num_transformer_layers": 3,
            "use_time": True,
            "use_gate": True,
            "use_interaction": True,
        },
        "loss": {
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        }
    },
    {
        "name": "proposed_with_mean_standarization_wo_rf",
        "modalities": ("mri", "pet", "cog", "csf"),
        "model": {
            "imputer": "mean",
            "scaling": "standarization",
            "batch_size": 32,
            "temporal_levels": (True, True, False),
            "num_transformer_layers": 3,
            "use_time": True,
            "use_gate": True,
            "use_interaction": True,
        },
        "loss": {
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        }
    },

]

LOSS_SEARCH_STAGES = {
    "stage_1_transition": [
        {
            "name": "s1_loss_1",
            "transition_loss_weight": 0.25,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_2",
            "transition_loss_weight": 0.5,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_3",
            "transition_loss_weight": 1,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_4",
            "transition_loss_weight": 1.5,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_5",
            "transition_loss_weight": 2,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_6",
            "transition_loss_weight": 3,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_7",
            "transition_loss_weight": 4,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_8",
            "transition_loss_weight": 5,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s1_loss_9",
            "transition_loss_weight": 6,
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
    ],
    "stage_2_converter_sample":[
        {
            "name": "s2_loss_1",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s2_loss_2",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 1.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s2_loss_3",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 2.0,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s2_loss_4",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 2.5,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s2_loss_5",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s2_loss_6",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 4,
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
    ],
    "stage_3_transition_weight": [
        {
            "name": "s3_loss_1",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s3_loss_2",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.0,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s3_loss_3",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s3_loss_4",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 2.0,
            "converter_transition_weight": 2.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s3_loss_5",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.0,
            "converter_transition_weight": 1.0,
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
    ],
    "stage_4_time": [
        {
            "name": "s4_loss_1",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5, #s3_loss_3
            "converter_transition_weight": 2.0, #s3_loss_3
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.0, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s4_loss_2",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5, #s3_loss_3
            "converter_transition_weight": 2.0, #s3_loss_3
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.5, 1.0, 1.0, 1.25],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s4_loss_3",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5, #s3_loss_3
            "converter_transition_weight": 2.0, #s3_loss_3
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.2, 1.0, 1.0, 1.1],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "s4_loss_4",
            "transition_loss_weight": 4, #s1_loss_7
            "transition_loss": "mse",
            "stable_transition_weight": 1.5, #s3_loss_3
            "converter_transition_weight": 2.0, #s3_loss_3
            "converter_sample_weight": 3, #s2_loss_5
            "huber_delta": 1.0,
            "stable_time_weights": [1.25, 1.0, 1.0, 1.0],
            "converter_time_weights": [1.0, 1.0, 1.0, 1.0],
        },
    ]
}