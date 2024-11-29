const behaviorTree = {
    "default": {
        "action": "wander",
        "transitions": [
            {
                "condition": "player_nearby",
                "action": "flee"
            },
            {
                "condition": "player_approaching",
                "action": "keep_distance"
            },
            {
                "condition": "obstacle_nearby",
                "action": "avoid_obstacle"
            },
            {
                "condition": "out_of_bounds",
                "action": "stay_within_bounds"
            },
            {
                "condition": "capture_object_nearby",
                "action": "attempt_capture"
            },
            {
                "condition": "capturing_npc_nearby",
                "action": "patrol_nearby"
            }
        ]
    }
  };
  