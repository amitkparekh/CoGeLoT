from pytest_cases import fixture, param_fixture

mission_task = param_fixture("mission_task", ["sweep", "follow_order"], scope="session")
mission_id = param_fixture("mission_id", ["000000"], scope="session")


@fixture(scope="session")
def pretrained_model() -> str:
    return "hf-internal-testing/tiny-random-t5"
