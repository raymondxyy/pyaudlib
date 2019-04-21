"""Test suite for ESC50 class."""
from audlib.data.esc50 import ESC50


def test_esc50():
    dataset = ESC50('/home/xyy/data/ESC-50')
    print(dataset)
    harmonicset = ESC50('/home/xyy/data/ESC-50',
                        categories=[
                            'crying_baby',
                            'clock_alarm',
                            'door_wood_creaks',
                            'cow',
                            'cat',
                            'church_bells',
                            'rooster',
                            'siren',
                        ])
    print(harmonicset)

    return


if __name__ == "__main__":
    test_esc50()
