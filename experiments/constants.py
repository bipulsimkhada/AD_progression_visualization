RANDOM_STATE = 42

MODALITIES = {
    "mri": (6, slice(None, 6)),
    "pet": (2, slice(6, 8)),
    "cog": (11, slice(8, 19)),
    "csf": (3, slice(19, 22)),
    "rf":  (4, slice(22, 26)),
}

LOSS_SEARCH_STAGES = {
    "stage_1_severity": [
        {
            "name": "s1_severity_000",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_025",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.25,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_050",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.5,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_075",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.75,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_100",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 1.0,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },

    ],
    "stage_2_transition": [
        {
            "name": "s2_transition_000",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_025",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 0.25,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_050",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 0.50,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_075",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 0.75,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_100",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 1.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_150",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 1.5,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_200",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 2.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_300",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 3.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_400",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 4.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_500",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 5.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_600",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, 
            "transition_weight": 6.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_700",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0,
            "transition_weight": 7.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_1000",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0,
            "transition_weight": 10.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
    ],
    "stage_3_time_weights": [
        {
            "name": "s3_time_equal",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # from stage1
            "transition_weight": 7.0, # from stage2
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s3_time_progressive",
            "time_weights": [1.0, 1.25, 1.5, 1.75],
            "severity_weight": 0.0, # from stage1
            "transition_weight": 7.0, # from stage2
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s3_time_progressive_mild",
            "time_weights": [1.0, 1.5, 2, 2.5],
            "severity_weight": 0.0, # from stage1
            "transition_weight": 7.0, # from stage2
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s3_time_late_focus_0_24",
            "time_weights": [2.0, 1.0, 1.0, 2.0],
            "severity_weight": 0.0, # from stage1
            "transition_weight": 7.0, # from stage2
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        # {
        #     "name": "s3_time_late_focus_equal",
        #     "time_weights": [1.0, 1.0, 3, 3],
        #     "severity_weight": 0.0, # from stage1
        #     "transition_weight": 7.0, # from stage2
        #     "transition_loss": "huber",
        #     "huber_delta": 1.0,
        # },
    ],
    "stage_4_loss_type": [
        # {
        #     "name": "s4_loss_mae",
        #     "time_weights": [1.0, 1.0, 1.0, 1.0], # frome stage3
        #     "severity_weight": 0.0, # from stage1
        #     "transition_weight": 2.0, # from stage2
        #     "transition_loss": "mae",
        #     "stable_transition_weight": 0.5,
        #     "converter_transition_weight": 3.0,
        #     "converter_sample_weight": 2,
        #     "huber_delta": 1.0,
        # },
        {
            "name": "s4_loss_mse", 
            "time_weights": [1.0, 1.0, 1.0, 1.0], # frome stage3
            "severity_weight": 0.0, # from stage1
            "transition_weight": 2.0, # from stage2
            "transition_loss": "mse",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 3.0,
            "converter_sample_weight": 2,
            "huber_delta": 1.0,
        },
        {
            "name": "s4_loss_huber",
            "time_weights": [1.0, 1.0, 1.0, 1.0], # frome stage3
            "severity_weight": 0.0, # from stage1
            "transition_weight": 2.0, # from stage2
            "transition_loss": "huber",
            "stable_transition_weight": 0.5,
            "converter_transition_weight": 3.0,
            "converter_sample_weight": 2,
            "huber_delta": 1.0,
        },
    ]
}