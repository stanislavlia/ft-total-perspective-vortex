from enum import Enum

class TaskParadigm(Enum):
    LEFT_RIGHT_HAND = "left_right_hand"
    HANDS_FEET = "hands_feet"

class TaskType(Enum):
    MOTOR_EXECUTION = "motor_execution"
    MOTOR_IMAGERY = "motor_imagery"

MOTOR_CHANNELS = [
    'C3..',   # Left motor cortex (primary)
    'Cz..',   # Central motor area (feet)
    'C4..',   # Right motor cortex (primary)
    'Fc3.',   # Left frontal-central (premotor)
    'Fc4.',   # Right frontal-central (premotor)
    'Cp3.',   # Left central-parietal (sensorimotor)
    'Cp4.',   # Right central-parietal (sensorimotor)
    'C5..',   # Left lateral motor
    'C1..',   # Left medial motor
    'C2..',   # Right medial motor
    'C6..',   # Right lateral motor
    'Fc1.',   # Left medial frontal-central
    'Fc2.',   # Right medial frontal-central
    'Fc5.',   # Left lateral frontal-central
    'Fc6.',   # Right lateral frontal-central
    'Cp1.',   # Left medial central-parietal
    'Cp2.',   # Right medial central-parietal
    'Cp5.',   # Left lateral central-parietal
    'Cp6.'    # Right lateral central-parietal
]

RUN_TYPE_TO_TASK = {
    "R01": {
        "name": "Baseline - Eyes Open",
        "task_type": "baseline",
        "labels": None
    },
    "R02": {
        "name": "Baseline - Eyes Closed",
        "task_type": "baseline",
        "labels": None
    },
    "R03": {
        "name": "Task 1 - Real Left/Right Fist",
        "task_type": "motor_execution",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R04": {
        "name": "Task 2 - Imagine Left/Right Fist",
        "task_type": "motor_imagery",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R05": {
        "name": "Task 3 - Real Fists/Feet",
        "task_type": "motor_execution",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    },
    "R06": {
        "name": "Task 4 - Imagine Fists/Feet",
        "task_type": "motor_imagery",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    },
    "R07": {
        "name": "Task 1 - Real Left/Right Fist",
        "task_type": "motor_execution",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R08": {
        "name": "Task 2 - Imagine Left/Right Fist",
        "task_type": "motor_imagery",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R09": {
        "name": "Task 3 - Real Fists/Feet",
        "task_type": "motor_execution",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    },
    "R10": {
        "name": "Task 4 - Imagine Fists/Feet",
        "task_type": "motor_imagery",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    },
    "R11": {
        "name": "Task 1 - Real Left/Right Fist",
        "task_type": "motor_execution",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R12": {
        "name": "Task 2 - Imagine Left/Right Fist",
        "task_type": "motor_imagery",
        "paradigm": "left_right_hand",
        "labels": {
            "T1": "left_fist",
            "T2": "right_fist"
        }
    },
    "R13": {
        "name": "Task 3 - Real Fists/Feet",
        "task_type": "motor_execution",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    },
    "R14": {
        "name": "Task 4 - Imagine Fists/Feet",
        "task_type": "motor_imagery",
        "paradigm": "hands_feet",
        "labels": {
            "T1": "both_fists",
            "T2": "both_feet"
        }
    }
}
