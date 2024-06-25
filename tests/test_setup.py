from unittest import skipIf

from typing import Union, Callable

class DisableTestSetup(dict):
    def __init__(self, disable:Union[bool, Callable]=False, reason:str="this test is disabled"):
        c=disable if isinstance(disable, bool) else disable()
        super(DisableTestSetup, self).__init__(
            condition=c,
            reason=reason
        )


DISABLE_TEST_SETUP_ONE = DisableTestSetup() # this test is enabled
DISABLE_TEST_SETUP_TWO = DisableTestSetup(True) # this test is disabled
# more DISABLE_* will come in here

@skipIf(**DISABLE_TEST_SETUP_ONE)
def test_setup_one():
    pass

@skipIf(**DISABLE_TEST_SETUP_TWO)
def test_setup_two():
    pass