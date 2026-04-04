from unittest import TestCase

from src.personal_llm.react import search


class Test(TestCase):
    def test_search(self):
        s = search("马来西亚和印度尼西亚首都是哪？")
        print(s)
