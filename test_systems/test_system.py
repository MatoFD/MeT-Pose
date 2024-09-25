from test_systems.test_mediapipe_holistic import test_holistic


def test_system(met_rules, sys_test, results_filepath, kwargs):
    """call the corresponding test class according to sys_test, and pass extra kwargs to it"""

    if sys_test == "holistic":
        return test_holistic(met_rules, results_filepath, kwargs)
    else:
        raise "this system test is not implemented yet, please check your spelling or add it to the framework"
