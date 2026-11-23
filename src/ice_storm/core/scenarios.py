
SCENARIOS = {
    "initial_response": {
        "priority_multipliers": {
            "Residential": 1,
            "Shelter": 1,
            "Clinic": 1,
            "Depot": 1,
        },
        "nu_0": 0.04,
        "rho": 0,
        "a": 5,
        "b": 5,
    },

    "evening_pressure": {
        "priority_multipliers": {
            "Residential": 1,
            "Shelter": 2,
            "Clinic": 2,
            "Depot": 1,
        },
        "nu_0": 0.04,
        "rho": 3,
        "a": 5,
        "b": 5,
    },

    "northern_outage": {
        "priority_multipliers": {
            "Residential": 2,
            "Shelter": 1,
            "Clinic": 2,
            "Depot": 1,
        },
        "nu_0": 0.06,
        "rho": 2,
        "a": 7,
        "b": 8,
    },
}
