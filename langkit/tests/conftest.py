import pytest


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(
        "--load", action="store_true", default=False, help="run load tests"
    )


def pytest_configure(config) -> None:  # type: ignore
    config.addinivalue_line(
        "markers", "load: mark test as load to skip running with unit tests"
    )


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore
    if config.getoption("--load"):
        # --integ specified on command line: do not skip integ tests
        return
    skip_load_test = pytest.mark.skip(reason="need --load option to run")
    for item in items:
        if "load" in item.keywords:
            item.add_marker(skip_load_test)


@pytest.fixture(scope="session")
def long_response():
    pair = {
        "prompt": "How do I befriend a racoon?",
        "response": "Ah, indeed, the prospect of forging a bond with the elusive and enigmatic raccoon, a creature of astute cunning and unwavering curiosity, is a formidable undertaking. Verily, let us embark upon an intellectual journey, beset by intricate knowledge and labyrinthine discourse, to unravel the mystery of befriending these captivating beings. To initiate this harmonious entanglement with a raccoon, one must first grasp the essence of their nature and comport oneself accordingly. Pray, let us contemplate the significance of patience, persistence, and a profound comprehension of their intricate behavioral patterns. The raccoon, Procyon lotor, of the family Procyonidae, is an embodiment of opportunism, adaptability, and resourcefulness. Its affinity for urban environments, combined with its prodigious dexterity and inquisitive disposition, often attracts the attention of intrepid individuals such as yourself, yearning to establish an amicable bond. Perchance, the acme of your endeavor lies in understanding the raccoon's intrinsic needs, and endeavoring to provide for them. Raccoons are omnivorous in their dietary proclivities, displaying an eclectic gustatory inclination toward both animal and plant matter. Therefore, incorporating an array of palatable edibles into your offerings, such as fruits, vegetables, nuts, and even the occasional proteinaceous delicacy, may engender a favorable response from our masked companions. Furthermore, the raccoon is an exemplar of nocturnality, shrouded in the veils of darkness, when the luminescent orb of the moon heralds their venturesome expeditions. Thus, it would be judicious to align your temporal rhythms with their preferred crepuscular inclinations, facilitating the establishment of a convivial connection during their active periods. Nonetheless, one must remain cognizant of the necessity for caution and the preservation of personal boundaries. Raccoons, though captivating, retain an inherent wildness, and attempting to intrude upon their personal space without due regard may elicit a defensive response, potentially jeopardizing the prospect of friendship. Permit the raccoon to approach you at its own volition, appreciating the gradual development of trust, as trust is the scaffold upon which the structure of companionship is erected. Engaging in activities that foster a sense of mutual amusement and intellectual stimulation may also serve to strengthen the rapport between yourself and this creature of untamed allure. Dispensing playthings, crafting puzzling contrivances, or engaging in captivating games of hide and seek could, perchance, ignite the fires of camaraderie and intellectual synergy. In conclusion, the path to befriending a raccoon is an arduous and intricate odyssey, fraught with nuance and subtlety. Patience, empathy, and a profound understanding of the raccoon's comportment and proclivities shall undoubtedly enhance your prospects of engendering a bond of companionship. May your journey be filled with intellectual fascination and the sublime joy of inter-species camaraderie.",  # noqa E501
    }
    return pair
